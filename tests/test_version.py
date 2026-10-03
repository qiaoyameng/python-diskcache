"""Optimistic concurrency: get-with-version, compare-and-set/delete."""

import glob
import io
import os.path as op
import shutil
import threading
import time
from unittest import mock

import pytest

import diskcache as dc

pytestmark = pytest.mark.filterwarnings('ignore', category=dc.EmptyDirWarning)


@pytest.fixture
def cache():
    with dc.Cache() as cache:
        yield cache
    shutil.rmtree(cache.directory, ignore_errors=True)


def val_files(cache):
    pattern = op.join(cache.directory, '**', '*.val')
    return set(glob.glob(pattern, recursive=True))


def test_get_returns_version(cache):
    assert cache.set('key', 'value')
    result = cache.get('key', version=True)
    assert isinstance(result, tuple) and len(result) == 2
    value, version = result
    assert value == 'value'
    assert isinstance(version, int) and version > 0


def test_version_tuple_combinations(cache):
    cache.set('key', 'value', tag='tag')

    value, expire, tag, version = cache.get(
        'key', expire_time=True, tag=True, version=True
    )
    assert (value, tag) == ('value', 'tag')
    assert isinstance(version, int)

    value, expire = cache.get('key', expire_time=True)
    assert value == 'value' and expire is None

    value, tag = cache.get('key', tag=True)
    assert (value, tag) == ('value', 'tag')

    value, version = cache.get('key', version=True)
    assert value == 'value' and isinstance(version, int)

    assert cache.get('key') == 'value'
    assert cache.get('missing') is None
    missing = cache.get('missing', version=True)
    assert missing == (None, None)
    missing = cache.get(
        'missing', default='d', expire_time=True, tag=True, version=True
    )
    assert missing == ('d', None, None, None)


def test_version_is_not_content_hash(cache):
    versions = []
    cache.set('key', 'value')
    versions.append(cache.get('key', version=True)[1])
    cache.set('key', 'other')
    versions.append(cache.get('key', version=True)[1])
    cache.set('key', 'value')  # changed back to the original value
    versions.append(cache.get('key', version=True)[1])
    assert len(set(versions)) == 3
    assert versions == sorted(versions)


def test_version_monotonic_global(cache):
    cache.set('a', 1)
    va1 = cache.get('a', version=True)[1]
    cache.set('b', 2)
    vb = cache.get('b', version=True)[1]
    cache.set('a', 3)
    va2 = cache.get('a', version=True)[1]
    assert va1 < vb < va2


def test_cas_success_and_conflict(cache):
    cache.set('key', 'value')
    version = cache.get('key', version=True)[1]

    assert cache.cas('key', 'new', version) is True
    assert cache.get('key') == 'new'

    # The old token is now stale.
    assert cache.cas('key', 'stale', version) is False
    assert cache.get('key') == 'new'

    new_version = cache.get('key', version=True)[1]
    assert new_version != version
    assert cache.cas('key', 'newer', new_version) is True
    assert cache.get('key') == 'newer'


def test_cas_missing_key(cache):
    assert cache.cas('missing', 'value', 1) is False
    assert cache.get('missing') is None


def test_cad_success_and_conflict(cache):
    cache.set('key', 'value')
    version = cache.get('key', version=True)[1]

    assert cache.cad('key', version + 1) is False
    assert cache.get('key') == 'value'
    assert cache.cad('key', version) is True
    assert cache.get('key') is None

    # A deleted version can never be reused.
    cache.set('key', 'value')
    new_version = cache.get('key', version=True)[1]
    assert new_version != version
    assert cache.cad('key', version) is False
    assert cache.get('key') == 'value'
    assert cache.cad('missing', version) is False


def test_delete_and_recreate_invalidates(cache):
    cache.set('key', 'value')
    version = cache.get('key', version=True)[1]
    cache.delete('key')
    cache.set('key', 'value')

    assert cache.cas('key', 'other', version) is False
    assert cache.cad('key', version) is False
    assert cache.get('key') == 'value'


def test_expired_recreate_invalidates(cache):
    cache.set('key', 'value', expire=0.05)
    version = cache.get('key', version=True)[1]
    time.sleep(0.08)

    assert cache.get('key') is None
    assert cache.cas('key', 'other', version) is False
    assert cache.cad('key', version) is False

    cache.set('key', 'recreated')
    assert cache.cas('key', 'other', version) is False
    assert cache.cad('key', version) is False
    assert cache.get('key') == 'recreated'


def test_expiry_and_version_checked_in_one_snapshot(cache):
    # Structural guarantee: cas/cad evaluate the expiry predicate and the
    # version comparison in a single SELECT, and that SELECT is the only
    # statement run against Cache inside the write transaction before the
    # UPDATE/DELETE. SQLite takes the transaction snapshot at BEGIN
    # IMMEDIATE, so there is no window in which a row can expire or be
    # replaced between the two checks.
    cache.set('key', 'value', expire=60)
    version = cache.get('key', version=True)[1]

    logged = []
    real_execute = type(cache)._sql.fget(cache)

    def spy(statement, *args, **kwargs):
        logged.append(' '.join(statement.upper().split()))
        return real_execute(statement, *args, **kwargs)

    with mock.patch.object(
        type(cache), '_sql', new=property(lambda self: spy)
    ):
        assert cache.cas('key', 'other', version) is True

    statements = logged
    begin_index = next(
        index
        for index, line in enumerate(statements)
        if line.startswith('BEGIN IMMEDIATE')
    )
    mutation_index = next(
        index
        for index, line in enumerate(statements[begin_index:], begin_index)
        if line.startswith('UPDATE CACHE')
    )
    pre_mutation = statements[begin_index:mutation_index]
    selects = [line for line in pre_mutation if line.startswith('SELECT') and ' FROM CACHE' in line]
    assert len(selects) == 1, selects
    condition = selects[0]
    assert 'EXPIRE_TIME IS NULL OR EXPIRE_TIME > ?' in condition
    assert 'VERSION' in condition
    assert condition.index('EXPIRE_TIME') < condition.index('VERSION')

    # Same shape for cad.
    logged.clear()
    cache.set('key2', 'value')
    version2 = cache.get('key2', version=True)[1]
    with mock.patch.object(
        type(cache), '_sql', new=property(lambda self: spy)
    ):
        assert cache.cad('key2', version2) is True
    statements = logged
    begin_index = next(
        index
        for index, line in enumerate(statements)
        if line.startswith('BEGIN IMMEDIATE')
    )
    mutation_index = next(
        index
        for index, line in enumerate(statements[begin_index:], begin_index)
        if line.startswith('DELETE FROM CACHE')
    )
    pre_mutation = statements[begin_index:mutation_index]
    selects = [line for line in pre_mutation if line.startswith('SELECT') and ' FROM CACHE' in line]
    assert len(selects) == 1, selects
    assert 'EXPIRE_TIME IS NULL OR EXPIRE_TIME > ?' in selects[0]
    assert 'VERSION' in selects[0]


def test_cas_conditional_ttl_and_tag(cache):
    cache.set('key', 'value', tag='old')
    version = cache.get('key', version=True)[1]

    # A conflict changes neither value, tag, nor expiry.
    assert cache.cas('key', 'x', version + 1, tag='new', expire=100) is False
    value, expire, tag = cache.get(
        'key', expire_time=True, tag=True
    )
    assert (value, tag) == ('value', 'old')
    assert expire is None

    # A successful conditional update changes everything atomically.
    assert cache.cas('key', 'value2', version, tag='new', expire=100) is True
    value, expire, tag = cache.get('key', expire_time=True, tag=True)
    assert (value, tag) == ('value2', 'new')
    assert expire is not None and expire > time.time()


def test_cas_preserves_ttl_and_tag(cache):
    cache.set('key', 'value', tag='keep', expire=100)
    _, before_expire, _, version = cache.get(
        'key', expire_time=True, tag=True, version=True
    )

    assert cache.cas('key', 'value2', version) is True
    value, after_expire, tag, _ = cache.get(
        'key', expire_time=True, tag=True, version=True
    )
    assert value == 'value2'
    assert tag == 'keep'
    assert after_expire == before_expire


def test_cas_can_clear_ttl_and_tag(cache):
    cache.set('key', 'value', tag='tag', expire=100)
    version = cache.get('key', version=True)[1]
    assert cache.cas('key', 'value2', version, expire=None, tag=None) is True
    value, expire, tag = cache.get('key', expire_time=True, tag=True)
    assert (value, expire, tag) == ('value2', None, None)


def test_interleaved_two_cache_instances_aba(tmpdir):
    directory = str(tmpdir)
    cache_a = dc.Cache(directory)
    cache_b = dc.Cache(directory)

    try:
        cache_a.set('key', 'shared')
        _, version_a = cache_a.get('key', version=True)
        _, version_b = cache_b.get('key', version=True)
        assert version_a == version_b

        assert cache_a.cas('key', 'from-a', version_a) is True
        # B's read-copy-update must not overwrite A's change.
        assert cache_b.cas('key', 'from-b', version_b) is False
        assert cache_b.get('key') == 'from-a'

        _, version_b = cache_b.get('key', version=True)
        assert cache_b.cas('key', 'from-b', version_b) is True

        # Delete and recreate with the same value: old tokens stay dead.
        cache_a.delete('key')
        cache_a.set('key', 'shared')
        _, recreated = cache_a.get('key', version=True)
        assert recreated != version_a and recreated != version_b
        assert cache_b.cas('key', 'stale', version_b) is False
        assert cache_b.cad('key', version_b) is False
        assert cache_a.get('key') == 'shared'

        cache_a.check()
        cache_b.check()
    finally:
        cache_a.close()
        cache_b.close()


def test_big_value_cas_success_removes_old_file(cache):
    big_old = b'o' * (2**15 + 1)
    big_new = b'n' * (2**15 + 2)
    cache.set('key', big_old)
    version = cache.get('key', version=True)[1]
    before = val_files(cache)
    assert len(before) == 1

    assert cache.cas('key', big_new, version) is True
    assert cache.get('key') == big_new
    after = val_files(cache)
    assert len(after) == 1
    assert after != before
    cache.check()


def test_big_value_cas_conflict_no_garbage(cache):
    big = b'o' * (2**15 + 1)
    cache.set('key', big)
    version = cache.get('key', version=True)[1]

    # Another writer replaces the value (and its file) before the cas runs.
    cache.set('key', b'p' * (2**15 + 3))
    live = cache.get('key')
    before = val_files(cache)
    assert len(before) == 1

    candidate = b'c' * (2**15 + 4)
    assert cache.cas('key', candidate, version) is False
    assert cache.get('key') == live
    after = val_files(cache)
    assert after == before  # no orphan, no deletion of the live file
    cache.check()


def test_big_value_cad_removes_file(cache):
    cache.set('key', b'o' * (2**15 + 1))
    version = cache.get('key', version=True)[1]
    assert len(val_files(cache)) == 1
    assert cache.cad('key', version) is True
    assert val_files(cache) == set()
    cache.check()


def test_big_value_stale_cad_keeps_live_file(cache):
    cache.set('key', b'o' * (2**15 + 1))
    version = cache.get('key', version=True)[1]
    cache.set('key', b'p' * (2**15 + 2))
    before = val_files(cache)

    assert cache.cad('key', version) is False
    assert cache.get('key') == b'p' * (2**15 + 2)
    assert val_files(cache) == before
    cache.check()


def test_big_value_cas_rollback_ownership(cache):
    big_old = b'o' * (2**15 + 1)
    cache.set('key', big_old)
    version = cache.get('key', version=True)[1]
    before = val_files(cache)

    real_next_version = cache._next_version

    def failing_next_version(sql=None):
        result = real_next_version(sql)
        raise RuntimeError('boom after version allocation')

    with mock.patch.object(cache, '_next_version', failing_next_version):
        with pytest.raises(RuntimeError):
            cache.cas('key', b'n' * (2**15 + 5), version)

    assert cache.get('key') == big_old
    assert cache.get('key', version=True)[1] == version
    assert val_files(cache) == before
    cache.check()


def test_small_value_to_big_value(cache):
    cache.set('key', 'small')
    version = cache.get('key', version=True)[1]
    assert val_files(cache) == set()

    assert cache.cas('key', b'b' * (2**15 + 1), version) is True
    assert cache.get('key') == b'b' * (2**15 + 1)
    assert len(val_files(cache)) == 1

    version = cache.get('key', version=True)[1]
    assert cache.cas('key', 'small-again', version) is True
    assert cache.get('key') == 'small-again'
    assert val_files(cache) == set()
    cache.check()


def test_big_value_file_like_conflict(cache):
    cache.set('key', b'o' * (2**15 + 1))
    version = cache.get('key', version=True)[1]
    cache.set('key', b'p' * (2**15 + 2))
    before = val_files(cache)

    reader = io.BytesIO(b'c' * (2**15 + 3))
    assert cache.cas('key', reader, version, read=True) is False
    assert val_files(cache) == before
    cache.check()


def test_concurrent_cas_only_one_wins(cache):
    cache.set('counter', 0)
    version = cache.get('counter', version=True)[1]
    barrier = threading.Barrier(2)
    results = []

    def attempt(value):
        barrier.wait()
        for _ in range(100):
            current = cache.get('counter', version=True)[1]
            if cache.cas('counter', value, current):
                results.append(value)
                return

    threads = [
        threading.Thread(target=attempt, args=(1,)),
        threading.Thread(target=attempt, args=(2,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == [1, 2]
    assert cache.get('counter') in (1, 2)
    cache.check()


def test_legacy_schema_migration(tmpdir):
    directory = str(tmpdir)
    # Create a cache with the new feature, then drop the version column as if
    # the database had been written by an older release.
    cache = dc.Cache(directory)
    cache.set('key', 'value')
    cache.close()

    import sqlite3

    con = sqlite3.connect(op.join(directory, dc.core.DBNAME))
    rows = con.execute('SELECT key, value FROM Cache').fetchall()
    con.execute(
        'CREATE TABLE OldCache AS SELECT'
        ' rowid, key, raw, store_time, expire_time, access_time,'
        ' access_count, tag, size, mode, filename, value FROM Cache'
    )
    con.execute('DROP TABLE Cache')
    con.execute('ALTER TABLE OldCache RENAME TO Cache')
    con.execute(
        'CREATE UNIQUE INDEX Cache_key_raw ON Cache(key, raw)'
    )
    con.execute('DELETE FROM Settings WHERE key = "version"')
    con.commit()
    con.close()

    cache = dc.Cache(directory)
    try:
        # Legacy rows carry version 0 and never match a compare token.
        assert cache.get('key') == 'value'
        result = cache._sql(
            'SELECT version FROM Cache WHERE key = ?',
            (cache._disk.put('key')[0],),
        ).fetchall()
        assert result == [(0,)]
        assert cache.cas('key', 'new', 0) is False
        assert cache.cad('key', 0) is False
        # Plain writes work and allocate real versions.
        cache.set('key', 'new')
        version = cache.get('key', version=True)[1]
        assert version > 0
        assert cache.cas('key', 'newer', version) is True
        cache.check()
    finally:
        cache.close()
