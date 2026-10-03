"""Resharding tool for fanout caches."""

import contextlib as cl
import hashlib
import os
import os.path as op
import shutil
import sqlite3
import tempfile
import time
import traceback

from .core import DEFAULT_SETTINGS, DBNAME, Cache, Disk
from .routes import (
    MAINTENANCE_LOCK_FILENAME,
    ROUTE_FILENAME,
    existing_legacy_shards,
    layout_from_route,
    load_migration_state,
    load_route,
    remove_migration_state,
    shard_path,
    write_migration_state,
    write_route,
)

MIGRATION_STATE_VERSION = 1
COPY_COLUMNS = (
    'key',
    'raw',
    'store_time',
    'expire_time',
    'access_time',
    'access_count',
    'tag',
    'size',
    'mode',
    'filename',
    'value',
)


class MigrationError(Exception):
    """Base class for fanout cache migration errors."""


class MigrationInProgress(MigrationError):
    """Another process is performing or holding a migration."""


class MigrationStateError(MigrationError):
    """Stored migration state is inconsistent."""


class MigrationDiskSpaceError(MigrationError):
    """Not enough disk space is available to continue migration."""


class MigrationValueFileError(MigrationError):
    """A value file required by migration is missing or invalid."""


class MigrationValidationError(MigrationError):
    """Source and destination data do not match."""


def _as_disk_space_error(error):
    if not isinstance(error, sqlite3.OperationalError):
        return error
    message = str(error).lower()
    if 'disk' not in message or 'full' not in message:
        return error
    return MigrationDiskSpaceError(str(error))


def _relative_paths(directory, layout):
    return [
        op.relpath(
            shard_path(directory, layout, index),
            directory,
        )
        for index in range(layout['shards'])
    ]


def _layout_with_paths(directory, layout):
    result = dict(layout)
    result['paths'] = _relative_paths(directory, layout)
    return result


def _initial_state(directory, source_layout, target_layout, batch_size):
    now = time.time()
    return {
        'version': MIGRATION_STATE_VERSION,
        'status': 'copying',
        'created': now,
        'updated': now,
        'source': _layout_with_paths(directory, source_layout),
        'target': _layout_with_paths(directory, target_layout),
        'batch_size': batch_size,
        'progress': [],
        'validation': {
            'status': 'pending',
            'source_rows': 0,
            'target_rows': 0,
            'validated_rows': 0,
            'last_source_shard': 0,
        },
        'error': None,
    }


def _save(directory, state, status=None):
    if status is not None:
        state['status'] = status
    state['updated'] = time.time()
    write_migration_state(directory, state)


def _record_error(directory, state, error, code=None):
    state['status'] = 'failed'
    error_record = {
        'type': type(error).__name__,
        'code': code,
        'message': str(error),
        'traceback': traceback.format_exc(),
        'time': time.time(),
    }
    for name in ('required_bytes', 'available_bytes'):
        value = getattr(error, name, None)
        if value is not None:
            error_record[name] = value
    state['error'] = error_record
    _save(directory, state)


@cl.contextmanager
def maintenance_lock(directory, timeout=60):
    """Acquire an exclusive cross-process migration lock."""
    os.makedirs(directory, exist_ok=True)
    connection = sqlite3.connect(
        op.join(directory, MAINTENANCE_LOCK_FILENAME),
        timeout=timeout,
        isolation_level=None,
    )
    try:
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('CREATE TABLE IF NOT EXISTS Lock(locked INTEGER)')
        try:
            connection.execute('BEGIN IMMEDIATE')
        except sqlite3.OperationalError:
            raise MigrationInProgress(
                'another fanout migration is in progress'
            ) from None
        yield connection
        connection.execute('COMMIT')
    finally:
        connection.close()


def _cache_settings(cache, total_shards):
    rows = cache._sql('SELECT key, value FROM Settings').fetchall()
    settings = dict(rows)
    size_limit = settings.pop('size_limit', DEFAULT_SETTINGS['size_limit'])
    settings['size_limit'] = size_limit * total_shards
    for key in ('count', 'size', 'hits', 'misses'):
        settings.pop(key, None)
    return settings


def _shard_progress(state, source_shard):
    progress = state['progress']
    while len(progress) <= source_shard:
        progress.append(
            {
                'source_shard': len(progress),
                'last_rowid': 0,
                'copied_rows': 0,
                'status': 'pending',
            }
        )
    return progress[source_shard]


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_join(base, filename):
    base_real = op.realpath(base)
    full_path = op.realpath(op.join(base_real, filename))
    try:
        if op.commonpath((base_real, full_path)) != base_real:
            raise MigrationValueFileError('value path escapes shard directory')
    except ValueError:
        raise MigrationValueFileError('invalid value file path') from None
    return full_path


def _copy_value_file(source_dir, target_dir, filename, source_shard, rowid):
    source_path = _safe_join(source_dir, filename)
    if op.islink(op.join(source_dir, filename)):
        raise MigrationValueFileError('source value is a symbolic link')
    if not op.isfile(source_path):
        raise MigrationValueFileError('missing value file: %s' % source_path)

    digest = hashlib.sha256()
    temporary = None
    target_filename = None
    try:
        with open(source_path, 'rb') as reader:
            chunk = reader.read(1024 * 1024)
            while chunk:
                digest.update(chunk)
                chunk = reader.read(1024 * 1024)

        hex_digest = digest.hexdigest()
        target_filename = op.join(
            'migrated',
            hex_digest[:2],
            hex_digest[2:4],
            '%03d-%020d-%s.val' % (source_shard, rowid, hex_digest[4:28]),
        )
        target_path = _safe_join(target_dir, target_filename)
        os.makedirs(op.dirname(target_path), exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix='.migrating-',
            dir=op.dirname(target_path),
        )
        copied = 0
        with os.fdopen(descriptor, 'wb') as writer, open(
            source_path, 'rb'
        ) as reader:
            while True:
                chunk = reader.read(1024 * 1024)
                if not chunk:
                    break
                writer.write(chunk)
                copied += len(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        if copied != op.getsize(source_path):
            raise MigrationValueFileError('value file changed during migration')
        os.replace(temporary, target_path)
        temporary = None
        _fsync_directory(op.dirname(target_path))
        if op.getsize(target_path) != copied:
            raise MigrationValueFileError('copied value file has wrong size')
        return target_filename
    except OSError as error:
        if temporary is not None:
            with cl.suppress(OSError):
                os.remove(temporary)
        if error.errno == 28:
            raise MigrationDiskSpaceError(str(error)) from error
        raise


def _database_file_size(path):
    total = 0
    for suffix in ('', '-wal', '-shm'):
        candidate = path + suffix
        if op.exists(candidate):
            total += op.getsize(candidate)
    return total


def _check_space(source_caches, target_caches, state):
    required = 0
    available = None
    for index, cache in enumerate(source_caches):
        progress = (
            state['progress'][index]
            if index < len(state['progress'])
            else {'last_rowid': 0}
        )
        rows = cache._sql(
            'SELECT COALESCE(SUM(size), 0) FROM Cache'
            ' WHERE rowid > ? AND filename IS NOT NULL',
            (progress['last_rowid'],),
        ).fetchone()
        required += int(rows[0])
        required += _database_file_size(
            op.join(cache.directory, DBNAME)
        )

    for cache in target_caches:
        usage = shutil.disk_usage(cache.directory)
        available = usage.free if available is None else min(
            available, usage.free
        )

    required = int(required * 1.1)
    if available is not None and required > available:
        error = MigrationDiskSpaceError(
            'migration requires about %d bytes but only %d bytes are free'
            % (required, available)
        )
        error.required_bytes = required
        error.available_bytes = available
        raise error


def _row_columns(row):
    return dict(zip(COPY_COLUMNS, row))


def _copy_shard(
    directory,
    state,
    source_cache,
    target_caches,
    source_shard,
    batch_size,
):
    progress = _shard_progress(state, source_shard)
    progress['status'] = 'copying'
    _save(directory, state)

    while True:
        _check_space(
            (source_cache,),
            target_caches,
            {
                'progress': [
                    {'last_rowid': progress['last_rowid']},
                ],
            },
        )
        rows = source_cache._sql(
            'SELECT rowid, %s FROM Cache WHERE rowid > ?'
            ' ORDER BY rowid LIMIT ?' % ', '.join(COPY_COLUMNS),
            (progress['last_rowid'], batch_size),
        ).fetchall()
        if not rows:
            progress['status'] = 'copied'
            _save(directory, state)
            return

        records = []
        target_batches = {}
        last_rowid = progress['last_rowid']
        for row in rows:
            rowid = row[0]
            record = _row_columns(row[1:])
            last_rowid = rowid
            key = source_cache.disk.get(record['key'], record['raw'])
            target_index = source_cache.disk.hash(key) % len(target_caches)
            target_cache = target_caches[target_index]
            if record['filename'] is not None:
                record['target_filename'] = _copy_value_file(
                    source_cache.directory,
                    target_cache.directory,
                    record['filename'],
                    source_shard,
                    rowid,
                )
            records.append(record)
            target_batches.setdefault(target_index, []).append(record)

        for target_index, target_records in target_batches.items():
            with target_caches[target_index]._transact(
                retry=True
            ) as (sql, _):
                for record in target_records:
                    target_filename = record.pop('target_filename', None)
                    if target_filename is not None:
                        record['filename'] = target_filename
                    sql(
                        'INSERT OR REPLACE INTO Cache (%s) VALUES (%s)'
                        % (
                            ', '.join(COPY_COLUMNS),
                            ', '.join('?' for _ in COPY_COLUMNS),
                        ),
                        tuple(record[column] for column in COPY_COLUMNS),
                    )

        progress['last_rowid'] = last_rowid
        progress['copied_rows'] += len(records)
        _save(directory, state)


def _file_hash(cache, filename):
    path = _safe_join(cache.directory, filename)
    if op.islink(path) or not op.isfile(path):
        raise MigrationValueFileError('missing value file: %s' % path)
    digest = hashlib.sha256()
    with open(path, 'rb') as reader:
        while True:
            chunk = reader.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.digest()


def _comparable_record(cache, row):
    record = _row_columns(row)
    file_hash = None
    if record['filename'] is not None:
        file_hash = _file_hash(cache, record['filename'])
    record['file_hash'] = file_hash
    record.pop('filename')
    return record


def _validate_migration(directory, state, source_caches, target_caches):
    validation = state['validation']
    validation['status'] = 'validating'
    _save(directory, state)

    validated = 0

    for target_cache in target_caches:
        integrity = target_cache._sql('PRAGMA integrity_check').fetchall()
        if len(integrity) != 1 or integrity[0][0] != 'ok':
            raise MigrationValidationError(
                'target database integrity check failed'
            )

    for source_shard, source_cache in enumerate(source_caches):
        last_rowid = 0
        while True:
            rows = source_cache._sql(
                'SELECT rowid, %s FROM Cache WHERE rowid > ?'
                ' ORDER BY rowid LIMIT ?' % ', '.join(COPY_COLUMNS),
                (last_rowid, 100),
            ).fetchall()
            if not rows:
                break

            for source_rowid, *source_values in rows:
                source_record = _row_columns(source_values)
                key = source_cache.disk.get(
                    source_record['key'], source_record['raw']
                )
                target_index = target_caches[0].disk.hash(
                    key
                ) % len(target_caches)
                target_cache = target_caches[target_index]
                target_rows = target_cache._sql(
                    'SELECT %s FROM Cache WHERE key = ? AND raw = ?'
                    % ', '.join(COPY_COLUMNS),
                    (source_record['key'], source_record['raw']),
                ).fetchall()
                if len(target_rows) != 1:
                    raise MigrationValidationError(
                        'target row missing for source %d/%d'
                        % (source_shard, source_rowid)
                    )
                target_record = _comparable_record(
                    target_cache, target_rows[0]
                )
                source_comparable = _comparable_record(
                    source_cache, source_values
                )
                if target_record != source_comparable:
                    raise MigrationValidationError(
                        'mismatched row for source %d/%d'
                        % (source_shard, source_rowid)
                    )
                validated += 1

            last_rowid = rows[-1][0]
            validation.update(
                {
                    'last_source_shard': source_shard,
                    'last_rowid': last_rowid,
                    'validated_rows': validated,
                }
            )
            _save(directory, state)

    source_total = sum(
        cache._sql('SELECT COUNT(*) FROM Cache').fetchone()[0]
        for cache in source_caches
    )
    target_total = sum(
        cache._sql('SELECT COUNT(*) FROM Cache').fetchone()[0]
        for cache in target_caches
    )
    if source_total != target_total or validated != source_total:
        raise MigrationValidationError(
            'row count mismatch: source=%d target=%d validated=%d'
            % (source_total, target_total, validated)
        )

    validation.update(
        {
            'status': 'complete',
            'source_rows': source_total,
            'target_rows': target_total,
            'validated_rows': validated,
            'last_source_shard': len(source_caches),
            'last_rowid': 0,
        }
    )
    _save(directory, state)


def _checkpoint(caches):
    for cache in caches:
        cache._sql('PRAGMA wal_checkpoint(TRUNCATE)').fetchall()


def _chmod_tree(root, file_mode, dir_mode):
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            os.chmod(op.join(dirpath, filename), file_mode)
        for dirname in dirnames:
            os.chmod(op.join(dirpath, dirname), dir_mode)
    os.chmod(root, dir_mode)


def _seal_source(directory, layout):
    for index in range(layout['shards']):
        root = shard_path(directory, layout, index)
        if op.exists(root):
            _chmod_tree(root, 0o444, 0o555)


def _delete_source(directory, layout):
    for index in range(layout['shards']):
        root = shard_path(directory, layout, index)
        if op.exists(root):
            _chmod_tree(root, 0o600, 0o700)
            shutil.rmtree(root)


def _delete_target_layout(directory, layout):
    _delete_source(directory, layout)
    if layout['layout_version'] == 2:
        generation_dir = op.join(
            directory,
            'shards',
            '%03d' % layout['generation'],
        )
        if op.isdir(generation_dir):
            shutil.rmtree(generation_dir)
        with cl.suppress(OSError):
            os.rmdir(op.join(directory, 'shards'))


def _target_layout(source_layout, target_shards):
    if target_shards <= 0:
        raise ValueError('target shards must be a positive integer')
    if target_shards == source_layout['shards']:
        raise ValueError('target shards must differ from source shards')
    return {
        'layout_version': 2,
        'generation': source_layout['generation'] + 1,
        'shards': target_shards,
    }


def _open_caches(
    directory,
    layout,
    disk=None,
    settings=None,
    timeout=60,
    read_only=False,
):
    caches = []
    common_settings = dict(settings or {})
    total_size_limit = common_settings.pop(
        'size_limit', DEFAULT_SETTINGS['size_limit']
    )
    for index in range(layout['shards']):
        cache_settings = common_settings.copy()
        cache_settings['size_limit'] = total_size_limit / layout['shards']
        caches.append(
            Cache(
                shard_path(directory, layout, index),
                timeout=timeout,
                disk=disk or Disk,
                read_only=read_only,
                **cache_settings,
            )
        )
    return caches


def _normalize_directory(directory):
    return op.abspath(op.expanduser(op.expandvars(str(directory))))


def _state_layout_matches(state, source_layout, target_layout):
    source_state = state.get('source', {})
    target_state = state.get('target', {})
    for key in ('layout_version', 'generation', 'shards'):
        if source_state.get(key) != source_layout[key]:
            return False
        if target_state.get(key) != target_layout[key]:
            return False
    return True


def _source_route(directory):
    route = load_route(directory)
    if route is not None:
        return route
    shards = existing_legacy_shards(directory)
    if not shards:
        raise MigrationStateError('no fanout cache found in directory')
    return {
        'version': 1,
        'layout_version': 1,
        'generation': 1,
        'shards': shards,
        'maintenance': False,
    }


def _release(caches):
    for cache in caches:
        cache.close()


def _unseal_source(directory, layout):
    for index in range(layout['shards']):
        root = shard_path(directory, layout, index)
        if op.exists(root):
            _chmod_tree(root, 0o600, 0o700)


def _source_is_sealed(directory, layout):
    root = shard_path(directory, layout, 0)
    db_path = op.join(root, DBNAME)
    return op.exists(db_path) and not os.access(db_path, os.W_OK)


def migrate_shards(
    directory,
    target_shards,
    disk=None,
    timeout=60,
    batch_size=100,
    delete_source=False,
    progress_callback=None,
):
    """Redistribute a fanout cache into a new shard layout.

    The migration is crash recoverable. Source shards remain complete and
    readable until validation and route installation complete. They are
    sealed read-only by default after the switch.
    """
    directory = _normalize_directory(directory)
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')

    with maintenance_lock(directory, timeout=timeout):
        source_route = _source_route(directory)
        source_layout = layout_from_route(source_route)
        target_layout = _target_layout(source_layout, target_shards)
        existing_state = load_migration_state(directory)

        if existing_state is not None:
            status = existing_state.get('status')
            if status == 'complete':
                raise MigrationStateError(
                    'migration to %d shards is already complete'
                    % existing_state['target']['shards']
                )
            if not _state_layout_matches(
                existing_state, source_layout, target_layout
            ):
                raise MigrationStateError(
                    'existing migration uses a different target layout'
                )
            state = existing_state
            state['error'] = None
            batch_size = state.get('batch_size', batch_size)
        else:
            state = _initial_state(
                directory, source_layout, target_layout, batch_size
            )
            _save(directory, state)

        active_route = load_route(directory)
        if active_route is not None and active_route['generation'] == (
            target_layout['generation']
        ):
            target_route = dict(active_route)
            target_route['maintenance'] = False
            write_route(directory, target_route)

            if delete_source:
                _delete_source(directory, source_layout)
                state['source_action'] = 'deleted'
            else:
                _seal_source(directory, source_layout)
                state['source_action'] = 'sealed'
            state['status'] = 'complete'
            _save(directory, state)
            return dict(state)

        maintenance_route = dict(source_route)
        maintenance_route['maintenance'] = True
        write_route(directory, maintenance_route)

        sealed_before_resume = _source_is_sealed(directory, source_layout)
        if (
            sealed_before_resume
            and state.get('validation', {}).get('status') == 'complete'
        ):
            target_route = {
                'version': 1,
                'layout_version': target_layout['layout_version'],
                'generation': target_layout['generation'],
                'shards': target_layout['shards'],
                'maintenance': False,
                'size_limit': source_route.get('size_limit'),
            }
            write_route(directory, target_route)
            state['status'] = 'switched'
            state['route_file'] = ROUTE_FILENAME
            _save(directory, state)
            if delete_source:
                _delete_source(directory, source_layout)
                state['source_action'] = 'deleted'
            else:
                state['source_action'] = 'sealed'
            state['status'] = 'complete'
            _save(directory, state)
            return dict(state)

        source_caches = None
        target_caches = None
        try:
            first_cache = Cache(
                shard_path(directory, source_layout, 0),
                timeout=timeout,
                disk=disk or Disk,
                read_only=sealed_before_resume,
            )
            disk_type = type(first_cache.disk) if disk is None else disk
            settings = _cache_settings(
                first_cache, source_layout['shards']
            )
            size_limit_total = settings['size_limit']
            maintenance_route['size_limit'] = size_limit_total
            write_route(directory, maintenance_route)
            first_cache.close()

            source_caches = _open_caches(
                directory,
                source_layout,
                disk=disk_type,
                settings=settings,
                timeout=timeout,
                read_only=sealed_before_resume,
            )
            target_caches = _open_caches(
                directory,
                target_layout,
                disk=disk_type,
                settings=settings,
                timeout=timeout,
            )

            for status in state['progress']:
                if status['status'] == 'copying':
                    status['status'] = 'pending'

            _check_space(source_caches, target_caches, state)

            if not sealed_before_resume:
                _checkpoint(source_caches)

            with cl.ExitStack() as stack:
                for source_cache in source_caches:
                    stack.enter_context(
                        source_cache._transact(retry=True)
                    )

                for source_shard, source_cache in enumerate(source_caches):
                    _copy_shard(
                        directory,
                        state,
                        source_cache,
                        target_caches,
                        source_shard,
                        batch_size,
                    )
                    if progress_callback is not None:
                        progress_callback(dict(state))

                _validate_migration(
                    directory, state, source_caches, target_caches
                )
                _checkpoint(target_caches)
                target_route = {
                    'version': 1,
                    'layout_version': target_layout['layout_version'],
                'generation': target_layout['generation'],
                'shards': target_layout['shards'],
                'maintenance': False,
                'size_limit': size_limit_total,
            }
                write_route(directory, target_route)
                state['status'] = 'switched'
                state['route_file'] = ROUTE_FILENAME
                _save(directory, state)

            _release(target_caches)
            _release(source_caches)
            target_caches = source_caches = None

            if delete_source:
                _delete_source(directory, source_layout)
                state['source_action'] = 'deleted'
            else:
                _seal_source(directory, source_layout)
                state['source_action'] = 'sealed'
            state['status'] = 'complete'
            _save(directory, state)
            return dict(state)
        except BaseException as error:
            disk_error = _as_disk_space_error(error)
            if disk_error is not error:
                _record_error(
                    directory,
                    state,
                    disk_error,
                    getattr(disk_error, 'errno', None),
                )
                raise disk_error from error
            _record_error(
                directory, state, error, getattr(error, 'errno', None)
            )
            raise
        finally:
            if target_caches is not None:
                _release(target_caches)
            if source_caches is not None:
                _release(source_caches)


def cancel_migration(directory, timeout=60):
    """Cancel a failed or interrupted migration and restore writes."""
    directory = _normalize_directory(directory)
    with maintenance_lock(directory, timeout=timeout):
        state = load_migration_state(directory)
        if state is None:
            raise MigrationStateError('no migration state found')
        if state.get('status') in ('switched', 'complete'):
            raise MigrationStateError('migration cannot be canceled after switch')

        source_layout = {
            key: state['source'][key]
            for key in ('layout_version', 'generation', 'shards')
        }
        target_layout = {
            key: state['target'][key]
            for key in ('layout_version', 'generation', 'shards')
        }
        _unseal_source(directory, source_layout)
        _delete_target_layout(directory, target_layout)
        source_route = load_route(directory) or {}
        route = {
            'version': 1,
            'layout_version': source_layout['layout_version'],
            'generation': source_layout['generation'],
            'shards': source_layout['shards'],
            'maintenance': False,
            'size_limit': source_route.get('size_limit'),
        }
        write_route(directory, route)
        remove_migration_state(directory)
