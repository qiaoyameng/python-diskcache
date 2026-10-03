"""Explicit migration tool for redistributing a :class:`FanoutCache` layout.

Changing the ``shards`` argument alone is not enough: keys are routed by
``hash(key) % shards`` so every row may move to a different shard. This
module performs an offline, resumable migration:

1. **Maintenance mode** -- an exclusive migration lock is taken and a
   write transaction is held open on every source shard, blocking writers
   using the old layout. Reads continue to work. Opening the fanout cache
   while a migration is in progress raises
   :class:`FanoutMigrationInProgress`.

2. **Redistribution** -- every row of every source shard is copied into a
   staging layout with ``shards=target_shards``. Key encoding (``key`` and
   ``raw`` columns), value columns, absolute ``expire_time`` and ``tag``
   are preserved byte-for-byte. Values stored in files are physically
   copied into the target shard directory; the target never references
   files in the old directory.

3. **Verification** -- rows are re-read from both layouts and compared,
   value files are checked (existence, size, sha256), target shards pass
   :meth:`Cache.check`, counts match and routing matches the new modulus.

4. **Cutover** -- the layout generation is bumped, old shard directories
   are moved aside atomically, staging shards are moved into place, and a
   new :data:`fanout-manifest.json` is published. Processes that still
   hold the old layout fail their next write transaction with
   :class:`GenerationMismatchError`.

A :data:`fanout-migration.json` manifest sits next to the cache and
records the source layout, target layout, progress (per source shard),
verification results and cutover state. Re-running :func:`migrate` resumes
an interrupted migration. Until cutover the source layout is fully
intact and readable; cutover only succeeds when verification passed.

"""

import contextlib as cl
import hashlib
import json
import os
import os.path as op
import shutil
import sqlite3
import tempfile
import time

from .core import DEFAULT_SETTINGS, Cache, Disk
from .fanout import (
    MANIFEST_NAME,
    MIGRATION_NAME,
    FanoutLayoutError,
    _atomic_json,
    _bootstrap_manifest,
    _disk_name,
    _legacy_shards,
    _read_manifest,
    _write_manifest,
)

STAGING_PREFIX = '.fanout-staging'
QUARANTINE_PREFIX = '.fanout-old'
MIGRATION_LOCK_NAME = '.fanout-migration.lock'

STATE_VERSION = 1

BATCH_SIZE = 256
CHUNK_SIZE = 2**20

# Free space requested on top of the bytes actually copied.
FREE_SPACE_MARGIN = 1.10


class MigrationError(Exception):
    """Base class for migration failures."""


class MigrationLockError(MigrationError):
    """Raised when the exclusive migration lock cannot be acquired."""


class MigrationDiskSpaceError(MigrationError):
    """Raised when the destination filesystem lacks free space."""


class MigrationValueFileError(MigrationError):
    """Raised when a value file referenced by the database is missing."""


class MigrationVerificationError(MigrationError):
    """Raised when the redistributed layout fails verification."""


def layout_info(directory):
    """Return information about the fanout layout at `directory`.

    :param str directory: fanout cache directory
    :return: dict with ``shards``, ``generation``, ``disk`` and migration
        status (when present)

    """
    directory = op.expandvars(op.expanduser(str(directory)))
    result = {'directory': directory, 'migration': None}

    manifest = _read_manifest(directory)

    if manifest is not None:
        result.update(
            shards=manifest.get('shards'),
            generation=manifest.get('generation'),
            disk=manifest.get('disk'),
        )
    else:
        legacy = _legacy_shards(directory)

        if legacy:
            result.update(shards=legacy, generation=None, disk=None)
        else:
            result.update(shards=None, generation=None, disk=None)

    migration_path = op.join(directory, MIGRATION_NAME)

    if op.exists(migration_path):
        with cl.suppress(OSError, ValueError):
            with open(migration_path, 'r', encoding='utf-8') as reader:
                result['migration'] = json.load(reader)

    return result


def _new_state(directory, source_manifest, source_shards, target_shards,
               disk_name, keep_source, started, free_space_margin):
    source_generation = source_manifest.get('generation')

    return {
        'version': STATE_VERSION,
        'directory': directory,
        'status': 'copying',
        'phase': 'copy',
        'started_at': started,
        'updated_at': started,
        'finished_at': None,
        'error': None,
        'source': {
            'shards': source_shards,
            'generation': source_generation,
            'disk': disk_name,
        },
        'target': {
            'shards': target_shards,
            'generation': source_generation + 1,
            'disk': disk_name,
        },
        'keep_source': keep_source,
        'free_space_margin': free_space_margin,
        'progress': {
            'shards': [
                {
                    'rows_copied': 0,
                    'files_copied': 0,
                    'bytes_copied': 0,
                    'done': False,
                }
                for _ in range(source_shards)
            ],
            'rows_total': 0,
            'files_total': 0,
            'bytes_total': 0,
        },
        'verification': {'status': 'pending', 'shards': [], 'errors': []},
        'cutover': {
            'status': 'pending',
            'generation_bumped': False,
            'moved': [],
            'manifest_published': False,
            'quarantine': None,
        },
    }


def _save_state(state):
    state['updated_at'] = time.time()
    path = op.join(state['directory'], MIGRATION_NAME)
    _atomic_json(path, state)


def _set_error(state, exc):
    state['error'] = {
        'type': type(exc).__name__,
        'message': str(exc),
        'at': time.time(),
    }


@cl.contextmanager
def _maintenance_lock(directory, timeout):
    """Exclusive cross-process lock for the whole migration."""
    path = op.join(directory, MIGRATION_LOCK_NAME)
    start = time.time()
    fd = None

    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except FileExistsError:
            lock_age = None

            with cl.suppress(OSError):
                lock_age = time.time() - op.getmtime(path)

            holder = _read_lock_pid(path)
            holder_dead = holder is not None and not _pid_alive(holder)

            # A killed process (SIGKILL, power loss) cannot remove its
            # lock file. If the recorded PID is dead and the lock is old
            # enough to rule out a live process mid-startup, reclaim it.

            if holder_dead and lock_age is not None and lock_age > 1.0:
                with cl.suppress(OSError):
                    os.remove(path)
                continue

            if lock_age is not None and lock_age > timeout:
                raise MigrationLockError(
                    'migration lock %r is held by another process'
                    ' (older than %.0fs); if no migration is running,'
                    ' remove the stale lock file (holder pid=%r)'
                    % (path, timeout, holder)
                )

            if time.time() - start > timeout:
                raise MigrationLockError(
                    'timed out after %.0fs waiting for migration lock %r'
                    % (timeout, path)
                )

    time.sleep(0.05)


    try:
        os.write(fd, ('%d\n' % os.getpid()).encode('ascii'))
        os.fsync(fd)
        yield
    finally:
        with cl.suppress(OSError):
            os.close(fd)
        with cl.suppress(OSError):
            os.remove(path)


def _read_lock_pid(path):
    try:
        with open(path, 'r', encoding='ascii') as reader:
            return int(reader.read().strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False

    return True


def _connect_shard(shard_dir, timeout=60):
    path = op.join(shard_dir, 'cache.db')
    con = sqlite3.connect(path, timeout=timeout, isolation_level=None)
    con.execute('PRAGMA busy_timeout = %d' % int(timeout * 1000))
    return con


def _lock_sources(state, directory, source_shards, timeout):
    """Begin IMMEDIATE transactions on all source shards.

    The transactions are held until the migration finishes (or aborts), so
    no process holding the old layout can write concurrently.
    """
    cons = []

    try:
        for num in range(source_shards):
            con = _connect_shard(op.join(directory, '%03d' % num), timeout)
            con.execute('BEGIN IMMEDIATE')
            cons.append(con)
    except sqlite3.OperationalError:
        for con in cons:
            with cl.suppress(sqlite3.Error):
                con.execute('ROLLBACK')
            con.close()
        raise MigrationLockError(
            'could not lock all source shards for maintenance; another'
            ' writer is active'
        )

    return cons


def _read_source_settings(con):
    rows = con.execute('SELECT key, value FROM Settings').fetchall()
    settings = {key: value for key, value in rows}
    return settings


def _settings_from_shard(shard_dir):
    """Read reopen settings (including custom disk_* keys) from a shard."""
    con = sqlite3.connect(op.join(shard_dir, 'cache.db'))
    try:
        settings = {
            key: value
            for key, value in con.execute(
                'SELECT key, value FROM Settings'
            ).fetchall()
        }
    finally:
        con.close()

    result = _target_cache_settings(settings)
    result.pop('fanout_generation', None)
    return result


def _target_cache_settings(settings):
    # Keep built-in settings and any custom Disk settings (disk_*), e.g.
    # JSONDisk's disk_compress_level. These must be identical on the target
    # or key encoding/hashing changes.
    result = {
        key: value
        for key, value in settings.items()
        if (
            key in DEFAULT_SETTINGS
            or key.startswith('disk_')
        )
        and key != 'size_limit'
    }

    for key in ('count', 'size', 'hits', 'misses', 'fanout_generation'):
        result.pop(key, None)

    return result


def _shard_bytes(shard_dir):
    total = 0

    for dirpath, _, filenames in os.walk(shard_dir):
        for name in filenames:
            with cl.suppress(OSError):
                total += op.getsize(op.join(dirpath, name))

    return total


def _check_disk_space(state, directory, source_shards, margin):
    required = 0

    for num in range(source_shards):
        required += _shard_bytes(op.join(directory, '%03d' % num))

    required = int(required * margin)

    usage = shutil.disk_usage(directory)

    if usage.free < required:
        raise MigrationDiskSpaceError(
            'not enough free space to stage %d shard(s): need ~%d bytes'
            ' (%0.1f%% margin) but only %d bytes available at %r'
            % (source_shards, required, (margin - 1) * 100, usage.free,
               directory)
        )

    return required, usage.free


def _sha256(path):
    digest = hashlib.sha256()

    with open(path, 'rb') as reader:
        for chunk in iter(lambda: reader.read(CHUNK_SIZE), b''):
            digest.update(chunk)

    return digest.hexdigest()


def _copy_value_file(src_dir, dst_dir, filename, progress):
    src_path = op.join(src_dir, filename)

    if not op.exists(src_path):
        raise MigrationValueFileError(
            'value file referenced by database is missing: %s' % src_path
        )

    dst_path = op.join(dst_dir, filename)
    dst_dirname = op.dirname(dst_path)
    os.makedirs(dst_dirname, exist_ok=True)

    # If a previous attempt left a target file (same key, re-run after
    # interruption), replace it via a temp file so no reader ever sees a
    # partial file.
    fd, tmp_path = tempfile.mkstemp(
        prefix='.migrating-', dir=dst_dirname
    )

    try:
        size = 0

        with os.fdopen(fd, 'wb') as writer:
            with open(src_path, 'rb') as reader:
                while True:
                    chunk = reader.read(CHUNK_SIZE)

                    if not chunk:
                        break

                    writer.write(chunk)
                    size += len(chunk)

            writer.flush()
            os.fsync(writer.fileno())

        os.replace(tmp_path, dst_path)
    except BaseException:
        with cl.suppress(OSError):
            os.remove(tmp_path)
        raise

    return size


SELECT_ROWS = (
    'SELECT rowid, key, raw, store_time, expire_time, access_time,'
    ' access_count, tag, size, mode, filename, value FROM Cache'
    ' ORDER BY rowid LIMIT ? OFFSET ?'
)

INSERT_ROW = (
    'INSERT OR REPLACE INTO Cache (key, raw, store_time,'
    ' expire_time, access_time, access_count, tag, size, mode, filename,'
    ' value) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'
)

# Selected row layout is (rowid, key, raw, store_time, expire_time,
# access_time, access_count, tag, size, mode, filename, value). Drop the
# rowid (target assigns a new one) to get the INSERT parameter tuple.


def _copy_shard(num, con, directory, targets, target_disk, state, hooks):
    src_dir = op.join(directory, '%03d' % num)
    progress = state['progress']['shards'][num]
    offset = progress['rows_copied']
    total_progress = state['progress']

    while True:
        rows = con.execute(SELECT_ROWS, (BATCH_SIZE, offset)).fetchall()

        if not rows:
            break

        grouped = {}

        for row in rows:
            db_key, raw = row[1], row[2]
            key = target_disk.get(db_key, raw)
            target_num = target_disk.hash(key) % len(targets)
            grouped.setdefault(target_num, []).append(row)

        for target_num, target_rows in sorted(grouped.items()):
            target = targets[target_num]
            target_dir = target.directory

            with target._transact(retry=True) as (sql, cleanup):
                for row in target_rows:
                    db_key, raw = row[1], row[2]
                    filename = row[10]

                    # On resume the row may already exist from a checkpointed
                    # batch. Remove its old value file so the replacement
                    # cannot leave an orphaned file behind.
                    old_rows = sql(
                        'SELECT filename FROM Cache WHERE key = ? AND raw = ?',
                        (db_key, raw),
                    ).fetchall()

                    for (old_filename,) in old_rows:
                        if old_filename is not None:
                            cleanup(old_filename)

                    if filename is not None:
                        _copy_value_file(
                            src_dir, target_dir, filename, progress
                        )

                    # Drop the source rowid: rowids from different source
                    # shards collide. The target assigns a fresh rowid and
                    # (key, raw) stays unique.
                    sql(INSERT_ROW, row[1:])

        offset += len(rows)
        progress['rows_copied'] = offset
        total_progress['rows_total'] = sum(
            item['rows_copied'] for item in total_progress['shards']
        )
        total_progress['files_total'] = sum(
            item['files_copied'] for item in total_progress['shards']
        )
        total_progress['bytes_total'] = sum(
            item['bytes_copied'] for item in total_progress['shards']
        )

        _save_state(state)

        hook = hooks.get('after_batch')

        if hook is not None:
            hook('after_batch', num, offset, state)

    progress['done'] = True
    ((file_count,),) = con.execute(
        'SELECT COUNT(*) FROM Cache WHERE filename IS NOT NULL'
    ).fetchall()
    ((byte_count,),) = con.execute(
        'SELECT COALESCE(SUM(size), 0) FROM Cache WHERE filename IS NOT NULL'
    ).fetchall()
    progress['files_copied'] = file_count
    progress['bytes_copied'] = byte_count
    _save_state(state)


def _open_targets(state, directory, disk, timeout, settings):
    target_shards = state['target']['shards']
    generation = state['target']['generation']
    staging = _staging_dir(directory)
    os.makedirs(staging, exist_ok=True)

    per_shard_limit = settings['size_limit'] / target_shards
    targets = [None] * target_shards

    for num in range(target_shards):
        shard_dir = op.join(staging, '%03d' % num)

        if not op.isdir(shard_dir):
            os.makedirs(shard_dir)

        target = Cache(
            directory=shard_dir,
            timeout=timeout,
            disk=disk,
            size_limit=per_shard_limit,
            fanout_generation=generation,
            **_target_cache_settings(settings),
        )
        targets[num] = target

    return targets


def _staging_dir(directory):
    return op.join(directory, STAGING_PREFIX)


def _verify(state, directory, source_cons, targets, target_disk):
    """Compare every source row against the redistributed target."""
    errors = []
    per_shard = state['verification']['shards'] = []
    source_shards = state['source']['shards']
    target_count = state['target']['shards']
    seen = [set() for _ in range(target_count)]

    for num, con in enumerate(source_cons):
        rows = con.execute(
            'SELECT rowid, key, raw, store_time, expire_time, access_time,'
            ' access_count, tag, size, mode, filename, value FROM Cache'
        ).fetchall()
        shard_errors = []

        for row in rows:
            (_rowid, db_key, raw, store_time, expire_time, access_time,
             access_count, tag, size, mode, filename, value) = row
            key = target_disk.get(db_key, raw)
            target_num = target_disk.hash(key) % target_count
            target = targets[target_num]

            actual = target._sql(
                'SELECT key, raw, store_time, expire_time,'
                ' access_time, access_count, tag, size, mode, filename,'
                ' value FROM Cache WHERE key = ? AND raw = ?',
                (db_key, raw),
            ).fetchall()

            if len(actual) != 1:
                shard_errors.append(
                    'key %r from shard %03d missing in target %03d'
                    % (key, num, target_num)
                )
                continue

            other = actual[0]

            expected = (
                db_key, raw, store_time, expire_time, access_time,
                access_count, tag, size, mode, filename, value,
            )

            if other != expected:
                shard_errors.append(
                    'key %r from shard %03d differs in target %03d'
                    % (key, num, target_num)
                )

            if filename is not None:
                dst_path = op.join(target.directory, filename)

                if not op.exists(dst_path):
                    shard_errors.append(
                        'value file missing in target: %s' % dst_path
                    )
                else:
                    if op.getsize(dst_path) != size:
                        shard_errors.append(
                            'value file size mismatch: %s' % dst_path
                        )
                    else:
                        src_path = op.join(
                            op.join(directory, '%03d' % num), filename
                        )

                        if op.exists(src_path) and _sha256(
                            src_path
                        ) != _sha256(dst_path):
                            shard_errors.append(
                                'value file sha256 mismatch: %s' % dst_path
                            )

                    # Large values must have been physically copied: a
                    # reference to the old directory would fail here.
                    if not op.realpath(dst_path).startswith(
                        op.realpath(target.directory)
                    ):
                        shard_errors.append(
                            'value file escapes target directory: %s'
                            % dst_path
                        )

            seen[target_num].add((db_key, raw))

        per_shard.append(
            {
                'source': num,
                'rows': len(rows),
                'errors': shard_errors,
            }
        )
        errors.extend(shard_errors)

    # No extra rows and counts agree.

    for num, target in enumerate(targets):
        ((count,),) = target._sql(
            'SELECT COUNT(*) FROM Cache'
        ).fetchall()

        if count != len(seen[num]):
            errors.append(
                'target shard %03d has %d rows but %d were routed to it'
                % (num, count, len(seen[num]))
            )

    warnings = []

    for num, target in enumerate(targets):
        warns = target.check(fix=False, retry=True)
        warnings.extend(
            {'shard': num, 'message': str(item)} for item in warns
        )

    state['verification'].update(
        status='failed' if errors else 'passed',
        errors=errors,
        warnings=warnings,
        checked_at=time.time(),
    )
    _save_state(state)

    if errors:
        raise MigrationVerificationError(
            '%d verification error(s): %s'
            % (len(errors), '; '.join(errors[:5]))
        )

    return errors


def _bump_generation(con, new_generation):
    """Bump the generation on a shard on an already write-locked `con`."""
    con.execute(
        'UPDATE Settings SET value = ? WHERE key = ?',
        (new_generation, 'fanout_generation'),
    )


def _checkpoint(shard_dir, timeout=60):
    con = sqlite3.connect(op.join(shard_dir, 'cache.db'), timeout=timeout)
    try:
        con.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    finally:
        con.close()


def _cutover(state, directory, source_cons, targets, hooks=None,
             disk=Disk, target_settings=None):
    cut = state['cutover']
    source_shards = state['source']['shards']
    target_shards = state['target']['shards']
    new_generation = state['target']['generation']
    staging = _staging_dir(directory)
    manifest_published = cut.get('manifest_published', False)

    if not cut.get('targets_checkpointed') and not manifest_published:
        # Flush and close all staging databases before moving directories.

        for target in targets:
            target.close()

        for num in range(target_shards):
            _checkpoint(op.join(staging, '%03d' % num))

        cut['targets_checkpointed'] = True
        _save_state(state)
    elif targets:
        for target in targets:
            target.close()

    # Bump the generation inside every source shard while the maintenance
    # write locks are still held: any stale process attempting a write now
    # fails with GenerationMismatchError.
    def _remaining_sources():
        return [
            num for num in range(source_shards)
            if op.isdir(op.join(directory, '%03d' % num))
        ]

    if not cut['generation_bumped'] and not manifest_published:
        # Re-lock shards whose directories already moved in a previous
        # (interrupted) attempt -- nothing to bump there.
        remaining = _remaining_sources()

        for num in remaining:
            _bump_generation(source_cons[num], new_generation)

        cut['generation_bumped'] = True
        _save_state(state)

        # Commit the maintenance transactions before renaming the
        # directories (a new connection cannot checkpoint while the write
        # transaction is held by this process).
        if source_cons is not None:
            for con in source_cons:
                with cl.suppress(sqlite3.Error):
                    con.execute('COMMIT')
                con.close()

            source_cons[:] = []

        for num in remaining:
            _checkpoint(op.join(directory, '%03d' % num))

    moved = set(cut.get('moved', []))

    if cut.get('quarantine') is None:
        quarantine = op.join(
            directory,
            '%s-gen%s'
            % (QUARANTINE_PREFIX, state['source']['generation']),
        )

        if not op.isdir(quarantine):
            os.mkdir(quarantine)

        cut['quarantine'] = quarantine
        _save_state(state)
    else:
        quarantine = cut['quarantine']

    def _move(src, dst, token):
        if token in moved:
            return

        os.rename(src, dst)
        moved.add(token)
        cut['moved'] = sorted(moved)
        _save_state(state)

    for num in range(source_shards):
        src = op.join(directory, '%03d' % num)
        dst = op.join(quarantine, '%03d' % num)

        if op.isdir(src):
            _move(src, dst, 'old-%03d' % num)

    for num in range(target_shards):
        src = op.join(staging, '%03d' % num)
        dst = op.join(directory, '%03d' % num)

        if op.isdir(src):
            _move(src, dst, 'new-%03d' % num)

    with cl.suppress(OSError):
        os.rmdir(staging)

    if not manifest_published:
        hook = hooks.get('before_manifest_publish') if hooks else None

        if hook is not None:
            hook('before_manifest_publish', state)

        _write_manifest(
            directory,
            target_shards,
            state['target']['disk'],
            new_generation,
        )
        cut['manifest_published'] = True
        _save_state(state)

    # Re-open using the new layout and run a full consistency check.

    if not cut.get('reopened_checked'):
        warnings = []
        reopen_settings = target_settings or {}
        reopen_generation = state['target']['generation']

        for num in range(target_shards):
            shard = Cache(
                op.join(directory, '%03d' % num),
                disk=disk,
                fanout_generation=reopen_generation,
                **reopen_settings,
            )
            try:
                warns = shard.check(fix=False, retry=True)
                warnings.extend(
                    {'shard': num, 'message': str(item)} for item in warns
                )
            finally:
                shard.close()

        cut['warnings'] = warnings
        cut['reopened_checked'] = True
        _save_state(state)
    cut['status'] = 'done'
    _save_state(state)

    return quarantine


def migrate(
    directory,
    shards,
    disk=Disk,
    timeout=60,
    keep_source=False,
    lock_timeout=60,
    free_space_margin=FREE_SPACE_MARGIN,
    resume=True,
    hooks=None,
):
    """Migrate a :class:`FanoutCache` at `directory` to `shards` shards.

    Resumes automatically when an interrupted migration state is found.

    :param str directory: fanout cache directory
    :param int shards: target number of shards
    :param disk: Disk type used by the cache (must match the source)
    :param float timeout: SQLite timeout in seconds
    :param bool keep_source: keep the old shard directories after cutover
        (they are moved under ``.fanout-old-genN`` instead of removed)
    :param float lock_timeout: seconds to wait for the migration lock and
        source shard write locks
    :param float free_space_margin: required free-space multiplier
    :param bool resume: when False, refuse to resume existing state
    :param dict hooks: callbacks used to inject interruptions in tests
    :return: migration report (also stored in ``fanout-migration.json``)

    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
    # pylint: disable=too-many-statements
    directory = op.expandvars(op.expanduser(str(directory)))
    hooks = hooks or {}
    os.makedirs(directory, exist_ok=True)

    disk_name = _disk_name(disk)
    state_path = op.join(directory, MIGRATION_NAME)
    existing = None

    # Fast path: a finished migration can be reported without the lock.

    if op.exists(state_path):
        with open(state_path, 'r', encoding='utf-8') as reader:
            candidate = json.load(reader)

        if candidate.get('status') == 'complete':
            return MigrationResult(candidate)

    with _maintenance_lock(directory, lock_timeout):
        # Re-read inside the lock: a concurrent migration may have created
        # the state after this process passed the fast path above.

        if op.exists(state_path):
            with open(state_path, 'r', encoding='utf-8') as reader:
                existing = json.load(reader)

            status = existing.get('status')

            if status == 'complete':
                return MigrationResult(existing)

            if status == 'aborted':
                existing = None
            elif not resume:
                raise MigrationError(
                    'interrupted migration state found at %r; re-run with'
                    ' resume=True (default) or resolve it manually'
                    % state_path
                )

        if existing is None:
            manifest = _read_manifest(directory)

            if manifest is None:
                legacy = _legacy_shards(directory)

                if not legacy:
                    raise FanoutLayoutError(
                        '%r is not a fanout cache directory' % directory
                    )

                manifest = _bootstrap_manifest(
                    directory, legacy, disk_name, generation=1
                )

            source_shards = manifest['shards']

            if manifest.get('disk') != disk_name:
                raise FanoutLayoutError(
                    'cache at %r uses disk %r but %r given'
                    % (directory, manifest.get('disk'), disk_name)
                )

            if source_shards == shards:
                raise MigrationError(
                    'cache already uses %d shard(s); nothing to migrate'
                    % shards
                )

            started = time.time()
            state = _new_state(
                directory, manifest, source_shards, shards, disk_name,
                keep_source, started, free_space_margin,
            )
            _save_state(state)
        else:
            state = existing

            if state['target']['shards'] != shards:
                raise MigrationError(
                    'existing migration targets %d shard(s) but %d requested'
                    % (state['target']['shards'], shards)
                )

            source_shards = state['source']['shards']

            # An interruption (process killed, power loss) left no target
            # connections open. Copy is checkpointed per batch and resumes
            # idempotently (INSERT OR REPLACE plus temp-file rename). If the
            # staging directory is gone, copy restarts from zero.

            if state['phase'] in ('copy', 'verify') and not op.isdir(
                _staging_dir(directory)
            ):
                for item in state['progress']['shards']:
                    item['rows_copied'] = 0
                    item['files_copied'] = 0
                    item['bytes_copied'] = 0
                    item['done'] = False

                state['progress'].update(
                    rows_total=0, files_total=0, bytes_total=0
                )
                state['phase'] = 'copy'
                state['status'] = 'copying'
                state['verification'] = {
                    'status': 'pending', 'shards': [], 'errors': []
                }
                state['error'] = None
                _save_state(state)
            elif state['phase'] in ('verify', 'cutover'):
                staging = _staging_dir(directory)

                if not op.isdir(staging):
                    # Staging was already moved; continue the cutover.
                    state['phase'] = 'cutover'
                    state['status'] = 'cutting-over'
                    _save_state(state)

        # Block normal opens for the duration of the maintenance window.
        _save_state(state)

        cutover_only = state['phase'] == 'cutover' and not op.isdir(
            _staging_dir(directory)
        )
        source_cons = []

        if not cutover_only:
            try:
                required, free = _check_disk_space(
                    state, directory, source_shards, free_space_margin
                )
            except MigrationDiskSpaceError as error:
                _set_error(state, error)
                _save_state(state)
                raise

            state['progress']['space_required'] = required
            state['progress']['space_free'] = free

            source_cons = _lock_sources(
                state, directory, source_shards, lock_timeout
            )
        else:
            source_cons = []

        try:
            hook = hooks.get('after_lock')

            if hook is not None:
                hook('after_lock', state)

            if cutover_only:
                targets = []
                first_new = next(
                    num
                    for num in range(state['target']['shards'])
                    if op.isdir(op.join(directory, '%03d' % num))
                )
                reopen_settings = _settings_from_shard(
                    op.join(directory, '%03d' % first_new)
                )
            else:
                settings = _read_source_settings(source_cons[0])
                targets = _open_targets(
                    state, directory, disk, timeout, settings
                )
                router = targets[0].disk
                reopen_settings = _target_cache_settings(settings)
                reopen_settings.pop('fanout_generation', None)

                if state['phase'] == 'copy':
                    for num in range(source_shards):
                        _copy_shard(
                            num, source_cons[num], directory, targets,
                            router, state, hooks,
                        )

                    state['phase'] = 'verify'
                    state['status'] = 'verifying'
                    _save_state(state)

                if state['phase'] == 'verify':
                    _verify(
                        state, directory, source_cons, targets, router
                    )
                    state['phase'] = 'cutover'
                    state['status'] = 'cutting-over'
                    _save_state(state)

                    hook = hooks.get('before_cutover')

                    if hook is not None:
                        hook('before_cutover', state)

            quarantine = _cutover(
                state, directory, source_cons, targets, hooks,
                disk=disk, target_settings=reopen_settings,
            )

            if not keep_source:
                shutil.rmtree(quarantine, ignore_errors=True)

            with cl.suppress(OSError):
                os.rmdir(_staging_dir(directory))

            state['status'] = 'complete'
            state['finished_at'] = time.time()
            _save_state(state)

            hook = hooks.get('after_cutover')

            if hook is not None:
                hook('after_cutover', state)

            return MigrationResult(state)
        except BaseException as exc:
            _set_error(state, exc)

            with cl.suppress(BaseException):
                _save_state(state)

            for con in source_cons:
                with cl.suppress(sqlite3.Error):
                    con.execute('ROLLBACK')
                con.close()

            raise



def abort_migrate(directory, lock_timeout=60):
    """Cancel an unfinished migration and restore normal operation.

    Only callable before cutover completes. The source layout is never
    modified during copy/verify, so aborting simply discards the staging
    area and marks the migration state aborted. If the cutover already
    finished, :class:`MigrationError` is raised and the new layout remains
    in place.

    :param str directory: fanout cache directory
    :param float lock_timeout: seconds to wait for the migration lock
    :return: dict describing the aborted state

    """
    directory = op.expandvars(op.expanduser(str(directory)))
    state_path = op.join(directory, MIGRATION_NAME)

    if not op.exists(state_path):
        raise MigrationError('no migration state found at %r' % state_path)

    with open(state_path, 'r', encoding='utf-8') as reader:
        state = json.load(reader)

    if state.get('status') == 'complete':
        raise MigrationError(
            'migration of %r already completed; cannot abort' % directory
        )

    if state.get('phase') == 'cutover' and state.get(
        'cutover', {}
    ).get('manifest_published'):
        raise MigrationError(
            'migration of %r already cut over; cannot abort' % directory
        )

    with _maintenance_lock(directory, lock_timeout):
        cut = state.get('cutover', {})
        quarantine = cut.get('quarantine')

        # A cutover may have renamed directories (but cannot have published
        # the manifest -- that is rejected above). Undo those renames.

        if cut.get('moved') or (quarantine and op.isdir(quarantine)):
            for num in range(state['target']['shards']):
                new_here = op.join(directory, '%03d' % num)
                token = 'new-%03d' % num

                if token in cut.get('moved', []) and op.isdir(new_here):
                    # Move the new shard back into staging (then discarded).
                    back = op.join(_staging_dir(directory), '%03d' % num)
                    os.makedirs(op.dirname(back), exist_ok=True)
                    os.rename(new_here, back)

            for num in range(state['source']['shards']):
                old_here = op.join(directory, '%03d' % num)
                old_saved = op.join(quarantine, '%03d' % num)

                if op.isdir(old_saved) and not op.isdir(old_here):
                    os.rename(old_saved, old_here)

            with cl.suppress(OSError):
                os.rmdir(quarantine)

        shutil.rmtree(_staging_dir(directory), ignore_errors=True)
        shutil.rmtree(_staging_dir(directory), ignore_errors=True)

        state['status'] = 'aborted'
        state['finished_at'] = time.time()
        state['error'] = {'type': 'Aborted', 'message': 'manual abort',
                          'at': state['finished_at']}
        _save_state(state)

    return state


class MigrationResult:
    """Summary of a finished migration."""

    def __init__(self, state):
        self._state = state

    def __getitem__(self, key):
        return self._state[key]

    def get(self, key, default=None):
        return self._state.get(key, default)

    @property
    def state(self):
        return self._state

    @property
    def source_shards(self):
        return self._state['source']['shards']

    @property
    def target_shards(self):
        return self._state['target']['shards']

    @property
    def generation(self):
        return self._state['target']['generation']

    @property
    def rows(self):
        return self._state['progress']['rows_total']

    @property
    def files(self):
        return self._state['progress']['files_total']

    @property
    def bytes(self):
        return self._state['progress']['bytes_total']

    def as_dict(self):
        return dict(self._state)

    def __repr__(self):
        return (
            'MigrationResult(source=%d, target=%d, generation=%s,'
            ' rows=%d, files=%d, bytes=%d)'
            % (
                self.source_shards,
                self.target_shards,
                self.generation,
                self.rows,
                self.files,
                self.bytes,
            )
        )
