"""Tests for the explicit FanoutCache shard migration tool."""

import json
import os
import os.path as op
import shutil
import sqlite3
import tempfile
import time

import pytest

import diskcache as dc
from diskcache import migration as mg


@pytest.fixture
def directory():
    path = tempfile.mkdtemp(prefix='diskcache-migrate-')
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _populate(directory, shards=2, rows=120, min_file_size=64):
    cache = dc.FanoutCache(
        directory,
        shards=shards,
        size_limit=10**9,
        disk_min_file_size=min_file_size,
        tag_index=True,
    )
    payload = {}

    for num in range(rows):
        key = 'key-%04d' % num

        if num % 3 == 0:
            value = b'B' * 200
        elif num % 3 == 1:
            value = num
        else:
            value = {'n': num, 'blob': b'v' * 100}

        tag = 'even' if num % 2 == 0 else 'odd'
        expire = 3600.0 if num % 5 == 0 else None
        cache.set(key, value, expire=expire, tag=tag)
        payload[key] = (value, tag)

    cache.set(b'bytes-key', 'text-value' * 20, tag='bytes')
    cache.set(987654321, 42, tag='int')
    cache.close()
    return payload


def _absolute_expires(directory, shards):
    result = {}

    for num in range(shards):
        path = op.join(directory, '%03d' % num, 'cache.db')
        con = sqlite3.connect(path)

        try:
            for key, expire in con.execute(
                'SELECT key, expire_time FROM Cache'
                ' WHERE expire_time IS NOT NULL'
            ):
                result[repr(bytes(key) if isinstance(key, (bytes, bytearray))
                       else key)] = expire
        finally:
            con.close()

    return result


def test_migrate_preserves_data_and_layout(directory):
    payload = _populate(directory, shards=2)
    before_expires = _absolute_expires(directory, 2)

    info = mg.layout_info(directory)
    assert info['shards'] == 2
    assert info['generation'] == 1
    assert info['migration'] is None

    result = mg.migrate(directory, shards=4)

    assert result.source_shards == 2
    assert result.target_shards == 4
    assert result.generation == 2
    assert result.rows == len(payload) + 2
    assert result.files > 0
    assert result.bytes > 0

    manifest = json.load(
        open(op.join(directory, 'fanout-manifest.json'))
    )
    assert manifest['shards'] == 4
    assert manifest['generation'] == 2

    # Absolute expire times survive unchanged.

    after_expires = _absolute_expires(directory, 4)
    assert after_expires == before_expires

    cache = dc.FanoutCache(
        directory, shards=4, size_limit=10**9,
        disk_min_file_size=64, tag_index=True,
    )

    try:
        assert len(cache) == len(payload) + 2

        for key, (value, tag) in payload.items():
            got, got_tag = cache.get(
                key, default=object(), tag=True
            )
            assert got == value, key
            assert got_tag == tag, key

        assert cache[b'bytes-key'] == 'text-value' * 20
        assert cache[987654321] == 42
        assert cache.check() == []

        cache.set('after-cutover', b'z' * 200)
        assert cache['after-cutover'] == b'z' * 200
    finally:
        cache.close()


def test_shards_argument_alone_is_an_error(directory):
    _populate(directory, shards=2)

    with pytest.raises(dc.FanoutLayoutError):
        dc.FanoutCache(directory, shards=4)

    # Source still readable with the correct shard count.

    cache = dc.FanoutCache(directory, shards=2)
    try:
        assert cache['key-0000'] == b'B' * 200
    finally:
        cache.close()


def test_idempotent_noop_when_complete(directory):
    _populate(directory, shards=2)
    first = mg.migrate(directory, shards=4)
    second = mg.migrate(directory, shards=4)
    assert second.state == first.state


def test_resume_after_interruption_in_copy(directory):
    payload = _populate(directory, shards=2, rows=80)
    calls = {'n': 0}

    def interrupt(event, shard, offset, state):
        calls['n'] += 1

        if calls['n'] == 2:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        mg.migrate(
            directory, shards=4,
            hooks={'after_batch': interrupt},
        )

    state_path = op.join(directory, 'fanout-migration.json')
    state = json.load(open(state_path))
    assert state['status'] == 'copying'
    assert state['error']['type'] == 'KeyboardInterrupt'

    # Source layout is intact and readable while in maintenance.

    cache = dc.FanoutCache(
        directory, shards=2, size_limit=10**9,
        disk_min_file_size=64, tag_index=True,
    )
    assert cache['key-0000'] == b'B' * 200
    cache.close()

    # The source layout can be opened while copy/verify is in progress;
    # writes are blocked by the maintenance write locks.

    reader = dc.FanoutCache(directory, shards=2)
    assert reader['key-0000'] == b'B' * 200
    reader.close()

    # Resume and finish.

    result = mg.migrate(directory, shards=4)
    assert result.rows == len(payload) + 2

    cache = dc.FanoutCache(
        directory, shards=4, size_limit=10**9,
        disk_min_file_size=64, tag_index=True,
    )

    try:
        for key, (value, tag) in payload.items():
            assert cache[key] == value, key
        assert cache.check() == []
    finally:
        cache.close()


def test_resume_after_interruption_before_cutover(directory):
    _populate(directory, shards=2, rows=40)

    def interrupt(event, state):
        raise SystemExit(1)

    with pytest.raises(SystemExit):
        mg.migrate(
            directory, shards=4,
            hooks={'before_cutover': interrupt},
        )

    state = json.load(
        open(op.join(directory, 'fanout-migration.json'))
    )
    assert state['phase'] == 'cutover'
    assert state['verification']['status'] == 'passed'

    # Source still complete: old directories untouched.

    assert op.isdir(op.join(directory, '000'))
    assert op.isdir(op.join(directory, '001'))

    result = mg.migrate(directory, shards=4)
    assert result['status'] == 'complete'

    cache = dc.FanoutCache(
        directory, shards=4, size_limit=10**9,
        disk_min_file_size=64, tag_index=True,
    )

    try:
        assert cache.check() == []
        assert cache['key-0000'] == b'B' * 200
    finally:
        cache.close()


def test_writes_blocked_while_migration_runs(directory):
    _populate(directory, shards=2)
    seen = {}

    # A process already holding the old layout:
    stale = [
        dc.Cache(op.join(directory, '%03d' % num), timeout=0.2)
        for num in range(2)
    ]

    def try_write(event, state):
        for shard in stale:
            # Writers holding the old layout fail while the migration
            # holds the maintenance write locks.
            with pytest.raises(
                (dc.Timeout, sqlite3.OperationalError)
            ):
                shard.set('racer', b'r' * 10, retry=False)

        seen['blocked'] = True

    try:
        mg.migrate(directory, shards=4, hooks={'after_lock': try_write})
    finally:
        for shard in stale:
            shard.close()

    assert seen.get('blocked') is True


def test_stale_writer_fails_generation_check_after_cutover(directory):
    _populate(directory, shards=2)
    stale = dc.FanoutCache(
        directory, shards=2, size_limit=10**9,
        disk_min_file_size=64, tag_index=True,
    )
    # Establish connections and read something.
    assert stale['key-0000'] == b'B' * 200

    mg.migrate(directory, shards=4)

    # The stale process is routed over the new generation and its writes
    # are rejected instead of landing in the wrong shard.

    with pytest.raises(dc.GenerationMismatchError):
        stale.set('stale-write', 1, retry=True)

    with pytest.raises(dc.GenerationMismatchError):
        stale.incr('counter', retry=True)

    stale.close()

    # Re-opening at the new generation works.

    fresh = dc.FanoutCache(directory, shards=4)
    assert fresh.set('fresh-write', 2) is True
    fresh.close()


def test_value_files_physically_copied(directory):
    payload = _populate(directory, shards=2)
    mg.migrate(directory, shards=4)

    for num in range(4):
        shard_dir = op.join(directory, '%03d' % num)
        con = sqlite3.connect(op.join(shard_dir, 'cache.db'))

        try:
            rows = con.execute(
                'SELECT filename FROM Cache WHERE filename IS NOT NULL'
            ).fetchall()

            for (filename,) in rows:
                full = op.join(shard_dir, filename)
                assert op.isfile(full), full
                assert op.realpath(full).startswith(
                    op.realpath(shard_dir)
                )
        finally:
            con.close()


def test_missing_value_file_reports_and_preserves_source(directory):
    _populate(directory, shards=2)
    victim = None

    for num in range(2):
        shard_dir = op.join(directory, '%03d' % num)
        con = sqlite3.connect(op.join(shard_dir, 'cache.db'))
        rows = con.execute(
            'SELECT filename FROM Cache WHERE filename IS NOT NULL'
            ' LIMIT 1'
        ).fetchall()
        con.close()

        if rows:
            victim = op.join(shard_dir, rows[0][0])
            break

    os.remove(victim)

    with pytest.raises(mg.MigrationValueFileError):
        mg.migrate(directory, shards=4)

    state = json.load(
        open(op.join(directory, 'fanout-migration.json'))
    )
    assert state['error']['type'] == 'MigrationValueFileError'

    # Source still fully present and openable at the old layout.

    cache = dc.FanoutCache(directory, shards=2)
    cache.close()
    assert op.isdir(op.join(directory, '000'))
    assert op.isdir(op.join(directory, '001'))


def test_disk_space_preflight(directory):
    _populate(directory, shards=2)

    with pytest.raises(mg.MigrationDiskSpaceError):
        mg.migrate(directory, shards=4, free_space_margin=10**9)

    state = json.load(
        open(op.join(directory, 'fanout-migration.json'))
    )
    assert state['error']['type'] == 'MigrationDiskSpaceError'
    assert op.isdir(op.join(directory, '000'))


def test_migration_lock_is_exclusive(directory):
    _populate(directory, shards=2)
    lock = op.join(directory, mg.MIGRATION_LOCK_NAME)

    def hold(event, state):
        assert op.exists(lock)

        with pytest.raises(mg.MigrationLockError):
            with mg._maintenance_lock(directory, timeout=0.1):
                pass

    mg.migrate(directory, shards=4, hooks={'after_lock': hold})
    assert not op.exists(lock)


def test_keep_source_moves_old_shards_aside(directory):
    _populate(directory, shards=2)
    result = mg.migrate(directory, shards=4, keep_source=True)

    quarantine = result['cutover']['quarantine']
    assert op.isdir(quarantine)
    assert op.isdir(op.join(quarantine, '000'))
    assert op.isdir(op.join(quarantine, '001'))

    old = dc.Cache(op.join(quarantine, '000'))
    try:
        old.check()
    finally:
        old.close()


def test_resume_mismatch_target_rejected(directory):
    _populate(directory, shards=2)

    with pytest.raises(KeyboardInterrupt):
        mg.migrate(
            directory, shards=4,
            hooks={'after_batch': lambda *a: (_ for _ in ()).throw(
                KeyboardInterrupt())},
        )

    with pytest.raises(mg.MigrationError):
        mg.migrate(directory, shards=8)


def test_legacy_cache_bootstrap_and_migrate(directory):
    # Build a cache, then strip the manifest and generation settings to
    # emulate a cache created before the routing manifest existed.

    _populate(directory, shards=3)
    os.remove(op.join(directory, 'fanout-manifest.json'))

    for num in range(3):
        con = sqlite3.connect(op.join(directory, '%03d' % num, 'cache.db'))
        con.execute(
            "DELETE FROM Settings WHERE key = 'fanout_generation'"
        )
        con.commit()
        con.close()

    assert mg.layout_info(directory)['shards'] == 3
    result = mg.migrate(directory, shards=6)
    assert result.source_shards == 3
    assert result.target_shards == 6

    cache = dc.FanoutCache(
        directory, shards=6, size_limit=10**9,
        disk_min_file_size=64, tag_index=True,
    )

    try:
        assert cache['key-0000'] == b'B' * 200
        assert cache.check() == []
    finally:
        cache.close()


CHILD_SCRIPT = r"""
import os, signal, sys, time
import diskcache as dc
from diskcache import migration as mg

directory = sys.argv[1]
target = int(sys.argv[2])
calls = {'n': 0}

sys.stderr.write('PID:%d\n' % os.getpid())
sys.stderr.flush()

def interrupt(event, shard, offset, state):
    calls['n'] += 1
    if calls['n'] == 3:
        os.kill(os.getpid(), signal.SIGKILL)

mg.migrate(directory, shards=target, hooks={'after_batch': interrupt})
"""


def test_resume_after_sigkill_subprocess(directory):
    payload = _populate(directory, shards=2, rows=1200)

    import sys
    import subprocess

    proc = subprocess.run(
        [sys.executable, '-c', CHILD_SCRIPT, directory, '5'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert proc.returncode < 0  # killed by signal
    child_pid = int(
        proc.stderr.decode('ascii').strip().split('PID:')[1]
    )

    # The lock file survives a hard kill (it is not an OS-level lock), but
    # the resume detects the dead holder PID and reclaims it.

    lock = op.join(directory, mg.MIGRATION_LOCK_NAME)
    assert op.exists(lock)
    assert mg._read_lock_pid(lock) == child_pid
    assert not mg._pid_alive(child_pid)

    state = json.load(
        open(op.join(directory, 'fanout-migration.json'))
    )
    assert state['status'] == 'copying'

    result = mg.migrate(directory, shards=5)
    assert result.rows == len(payload) + 2

    cache = dc.FanoutCache(
        directory, shards=5, size_limit=10**9,
        disk_min_file_size=64, tag_index=True,
    )

    try:
        assert len(cache) == len(payload) + 2

        for key, (value, tag) in payload.items():
            assert cache[key] == value, key

        assert cache.check() == []
    finally:
        cache.close()


def test_abort_in_copy_restores_normal_operation(directory):
    _populate(directory, shards=2, rows=60)

    with pytest.raises(KeyboardInterrupt):
        mg.migrate(
            directory, shards=4,
            hooks={
                'after_batch': lambda *a: (_ for _ in ()).throw(
                    KeyboardInterrupt())
            },
        )

    state = mg.abort_migrate(directory)
    assert state['status'] == 'aborted'
    assert not op.exists(op.join(directory, mg.STAGING_PREFIX))
    assert not op.exists(op.join(directory, mg.MIGRATION_LOCK_NAME))

    cache = dc.FanoutCache(
        directory, shards=2, size_limit=10**9,
        disk_min_file_size=64, tag_index=True,
    )

    try:
        assert cache['key-0000'] == b'B' * 200
        assert cache.set('new-write', 1) is True
        assert cache['new-write'] == 1
    finally:
        cache.close()

    # A fresh migration can start after abort.

    result = mg.migrate(directory, shards=6)
    assert result.target_shards == 6


def test_abort_before_cutover(directory):
    _populate(directory, shards=2, rows=30)

    with pytest.raises(SystemExit):
        mg.migrate(
            directory, shards=4,
            hooks={'before_cutover': lambda *a: (_ for _ in ()).throw(
                SystemExit(1))},
        )

    state = mg.abort_migrate(directory)
    assert state['status'] == 'aborted'
    assert op.isdir(op.join(directory, '000'))
    assert op.isdir(op.join(directory, '001'))

    cache = dc.FanoutCache(directory, shards=2, size_limit=10**9,
                           disk_min_file_size=64)
    try:
        assert cache['key-0000'] == b'B' * 200
        assert cache.set('x', 2) is True
    finally:
        cache.close()


CUTOVER_CHILD_SCRIPT = r"""
import os, signal, sys
import diskcache as dc
from diskcache import migration as mg

directory = sys.argv[1]
target = int(sys.argv[2])
calls = {'done': False}

def hook(event, state):
    # Kill after the new manifest is published (routing switched) but
    # before the migration marks itself complete.
    cut = state.get('cutover', {})
    if cut.get('manifest_published') and not calls['done']:
        calls['done'] = True
        os.kill(os.getpid(), signal.SIGKILL)

sys.stderr.write('PID:%d\n' % os.getpid())
sys.stderr.flush()
mg.migrate(directory, shards=target, hooks={'after_cutover': hook})
"""


def test_resume_after_sigkill_during_cutover(directory):
    payload = _populate(directory, shards=2, rows=60)

    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, '-c', CUTOVER_CHILD_SCRIPT, directory, '4'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert proc.returncode < 0

    state = json.load(
        open(op.join(directory, 'fanout-migration.json'))
    )
    assert state['cutover']['manifest_published'] is True

    # New layout is already live; resume just finalizes.

    result = mg.migrate(directory, shards=4)
    assert result['status'] == 'complete'

    cache = dc.FanoutCache(
        directory, shards=4, size_limit=10**9,
        disk_min_file_size=64, tag_index=True,
    )

    try:
        for key, (value, tag) in payload.items():
            assert cache[key] == value, key
        assert cache.check() == []
    finally:
        cache.close()


MID_CUTOVER_CHILD_SCRIPT = r"""
import os, signal, sys
import diskcache.migration as mg

directory = sys.argv[1]
target = int(sys.argv[2])

def hook(event, state):
    raise SystemExit(2)

sys.stderr.write('PID:%d\n' % os.getpid()); sys.stderr.flush()
mg.migrate(
    directory, shards=target,
    hooks={'before_manifest_publish': hook},
)
"""


def test_resume_after_dirs_moved_before_manifest(directory):
    payload = _populate(directory, shards=2, rows=40)

    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, '-c', MID_CUTOVER_CHILD_SCRIPT, directory, '4'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode == 2

    state = json.load(
        open(op.join(directory, 'fanout-migration.json'))
    )
    assert state['phase'] == 'cutover'
    assert state['cutover']['manifest_published'] is False

    result = mg.migrate(directory, shards=4)
    assert result['status'] == 'complete'

    cache = dc.FanoutCache(
        directory, shards=4, size_limit=10**9,
        disk_min_file_size=64, tag_index=True,
    )

    try:
        for key, (value, tag) in payload.items():
            assert cache[key] == value, key
        assert cache.check() == []
    finally:
        cache.close()


def test_custom_jsondisk_settings_preserved(directory):
    cache = dc.FanoutCache(
        directory, shards=3, disk=dc.JSONDisk,
        disk_compress_level=9, size_limit=10**9,
    )
    cache['doc'] = {'a': [1, 2], 'b': 'text' * 20}
    cache['n'] = 7
    cache.close()

    mg.migrate(directory, shards=6, disk=dc.JSONDisk)

    reopened = dc.FanoutCache(
        directory, shards=6, disk=dc.JSONDisk,
        disk_compress_level=9, size_limit=10**9,
    )

    try:
        assert reopened['doc'] == {'a': [1, 2], 'b': 'text' * 20}
        assert reopened['n'] == 7
        assert reopened.check() == []
    finally:
        reopened.close()


def test_plain_cache_without_generation_is_unguarded(directory):
    # A standalone Cache (not a fanout shard) has NULL generation and
    # keeps working after a migration touches unrelated directories.

    plain_dir = op.join(directory, 'standalone')
    plain = dc.Cache(plain_dir)
    plain.set('k', 1)

    _populate(directory, shards=2)
    mg.migrate(directory, shards=4)

    plain.set('k', 2)
    assert plain['k'] == 2
    plain.close()


STALE_WRITER_CHILD = r"""
import os, sys, time
import diskcache as dc

directory = sys.argv[1]
shard = dc.Cache(os.path.join(directory, '000'), timeout=2)
sys.stderr.write('READY\n'); sys.stderr.flush()

# Wait until the parent tells us the migration finished.
line = sys.stdin.readline()
try:
    shard.set('late-write', b'x', retry=True)
except dc.GenerationMismatchError:
    sys.stderr.write('GENERATION_MISMATCH\n')
    sys.stderr.flush()
shard.close()
"""


def test_cross_process_stale_writer_rejected(directory):
    _populate(directory, shards=2)

    import subprocess
    import sys

    child = subprocess.Popen(
        [sys.executable, '-c', STALE_WRITER_CHILD, directory],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Give the child time to open its raw connection.

    line = child.stderr.readline()
    assert line.strip() == b'READY'

    mg.migrate(directory, shards=4)

    child.stdin.write(b'go\n')
    child.stdin.flush()
    _, stderr = child.communicate(timeout=30)

    assert b'GENERATION_MISMATCH' in stderr, stderr
    assert child.returncode == 0


def test_two_concurrent_migrations_exclusive(directory):
    _populate(directory, shards=2, rows=40)

    import threading

    errors = []
    results = []

    def run(hold):
        try:
            hooks = {}

            if hold:
                def sleep_holder(event, state):
                    time.sleep(1.0)

                hooks['after_lock'] = sleep_holder

            results.append(
                mg.migrate(
                    directory, shards=4, lock_timeout=0.2, hooks=hooks
                )
            )
        except BaseException as error:  # noqa
            errors.append(error)

    t1 = threading.Thread(target=run, args=(True,))
    t2 = threading.Thread(target=run, args=(False,))
    t1.start()
    t2.start()
    t1.join(timeout=60)
    t2.join(timeout=60)

    assert len(results) + len(errors) == 2
    assert len(errors) == 1, [type(e).__name__ for e in errors]
    assert isinstance(errors[0], mg.MigrationLockError), errors[0]
