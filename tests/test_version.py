"""Tests for get-with-version and compare-and-set/delete (optimistic
concurrency control).

Versions are change identities, not content hashes: delete/recreate,
expire/recreate, and change-back-to-same-value all invalidate stale versions.
"""

import os
import shutil
import tempfile
import time

import pytest

import diskcache as dc


@pytest.fixture
def directory():
    path = tempfile.mkdtemp(prefix='diskcache-version-')
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def cache(directory):
    with dc.Cache(directory) as cache:
        yield cache


def test_get_version_missing(cache):
    assert cache.get('missing', version=True) == (None, None)
    assert cache.get('missing', default='d', version=True) == ('d', None)


def test_version_changes_on_set(cache):
    cache.set('k', 'a')
    _, version_a = cache.get('k', version=True)
    cache.set('k', 'b')
    _, version_b = cache.get('k', version=True)
    assert version_a < version_b
    # Setting the same value still produces a new version.
    cache.set('k', 'b')
    _, version_c = cache.get('k', version=True)
    assert version_b < version_c


def test_cas_success_and_conflict(cache):
    cache.set('k', 0)
    _, version = cache.get('k', version=True)

    assert cache.cas('k', 1, version) is True
    assert cache.get('k') == 1

    # Stale version loses.
    assert cache.cas('k', 2, version) is False
    assert cache.get('k') == 1

    _, version2 = cache.get('k', version=True)
    assert cache.cas('k', 3, version2) is True
    assert cache.get('k') == 3


def test_cas_on_missing_key_fails(cache):
    assert cache.cas('nope', 1, 1) is False
    assert 'nope' not in cache


def test_cad_conflict_and_success(cache):
    cache.set('k', 0)
    _, version = cache.get('k', version=True)
    cache.set('k', 1)

    assert cache.cad('k', version) is False
    assert 'k' in cache

    _, version2 = cache.get('k', version=True)
    assert cache.cad('k', version2) is True
    assert 'k' not in cache

    # Deleting a missing key is a conflict.
    assert cache.cad('k', version2) is False


def test_aba_change_back_to_same_value_two_instances(directory):
    """Two independent Cache instances: stale version must not win when the
    value was changed away and then back (classic ABA)."""
    left = dc.Cache(directory)
    right = dc.Cache(directory)

    left.set('counter', 100)
    value, version_left = left.get('counter', version=True)
    assert value == 100

    # Right interleaves A -> B -> A.
    _, version_right = right.get('counter', version=True)
    assert version_right == version_left
    assert right.cas('counter', 200, version_right) is True
    _, version_b = right.get('counter', version=True)
    assert right.cas('counter', 100, version_b) is True  # same value as A

    # The value looks equal but left's version is stale.
    assert left.get('counter') == 100
    assert left.cas('counter', 999, version_left) is False
    assert left.get('counter') == 100  # lost update was rejected

    # Right holds the current version and proceeds.
    _, version_now = right.get('counter', version=True)
    assert right.cas('counter', 1, version_now) is True
    assert right.get('counter') == 1

    left.close()
    right.close()


def test_aba_delete_and_recreate_two_instances(directory):
    left = dc.Cache(directory)
    right = dc.Cache(directory)

    left.set('k', 'v')
    _, version = left.get('k', version=True)

    assert right.cad('k', version) is True
    right.set('k', 'v')  # recreated with identical value

    assert left.get('k') == 'v'  # indistinguishable by value
    assert left.cas('k', 'stale', version) is False
    assert left.cad('k', version) is False
    assert right.get('k') == 'v'

    left.close()
    right.close()


def test_aba_expire_and_recreate_two_instances(directory):
    left = dc.Cache(directory)
    right = dc.Cache(directory)

    left.set('k', 'v', expire=0.1)
    _, version = left.get('k', version=True)

    time.sleep(0.15)

    # Expired items read as missing ...
    assert left.get('k', version=True) == (None, None)
    # ... but a stale conditional update cannot resurrect/overwrite.
    assert left.cas('k', 'stale', version) is False
    assert left.cad('k', version) is False

    # Recreate the key with the same value after expiry.
    right.set('k', 'v')
    assert left.cas('k', 'stale', version) is False
    assert left.cad('k', version) is False

    _, fresh = right.get('k', version=True)
    assert right.cas('k', 'new', fresh) is True
    assert right.get('k') == 'new'

    left.close()
    right.close()


def test_cas_can_update_expire_and_tag(cache):
    cache.set('k', 'v', expire=100, tag='old')
    _, version = cache.get('k', version=True)

    # Conditional update changes value, TTL and tag together.
    assert cache.cas('k', 'w', version, expire=100, tag='new') is True
    value, expire_time, tag, version2 = cache.get(
        'k', expire_time=True, tag=True, version=True
    )
    assert (value, tag) == ('w', 'new')
    assert expire_time is not None
    assert version2 != version

    # A failed conditional update changes nothing at all.
    assert cache.cas('k', 'x', version, expire=None, tag='other') is False
    value, expire_time, tag, _ = cache.get(
        'k', expire_time=True, tag=True, version=True
    )
    assert (value, tag) == ('w', 'new')
    assert expire_time is not None

    # Omitting expire/tag keeps the current ones.
    assert cache.cas('k', 'z', version2) is True
    value, expire_time, tag, _ = cache.get(
        'k', expire_time=True, tag=True, version=True
    )
    assert (value, tag) == ('z', 'new')
    assert expire_time is not None

    # Explicit None clears the TTL and tag.
    _, version3 = cache.get('k', version=True)
    assert cache.cas('k', 'q', version3, expire=None, tag=None) is True
    value, expire_time, tag, _ = cache.get(
        'k', expire_time=True, tag=True, version=True
    )
    assert (value, expire_time, tag) == ('q', None, None)


def test_big_value_conflict_cleans_staged_file(directory):
    """A deliberate conflict on a file-backed (large) value must not leave an
    orphaned file and must keep the current value's file in place."""
    min_file_size = 2**15

    left = dc.Cache(directory, disk_min_file_size=min_file_size)
    right = dc.Cache(directory, disk_min_file_size=min_file_size)

    big = b'x' * (min_file_size * 4)
    left.set('blob', big)

    value, version = left.get('blob', version=True)
    assert value == big
    _, row_mode, row_filename, _ = _row(left, 'blob')
    assert row_filename is not None
    old_path = os.path.join(directory, row_filename)
    assert os.path.exists(old_path)

    # Another writer wins first.
    right.set('blob', b'y' * (min_file_size * 4))

    # Staged large value is written before the transaction ...
    staged = b'z' * (min_file_size * 4)
    assert left.cas('blob', staged, version) is False  # deliberate conflict

    # ... but no garbage file remains and the current value survives.
    assert left.check(fix=False) == []
    # Winner replaced the file; the old one was removed after commit.
    assert not os.path.exists(old_path)
    value2, version2 = right.get('blob', version=True)
    assert value2 == b'y' * (min_file_size * 4)
    _, _, new_filename, _ = _row(right, 'blob')
    assert os.path.exists(os.path.join(directory, new_filename))

    # The fresh version can update with a large value and check() stays clean.
    assert right.cas('blob', big, version2) is True
    assert right.check(fix=False) == []
    assert left.get('blob') == big

    left.close()
    right.close()


def test_big_value_rollback_cleans_staged_file(directory):
    """If the transaction rolls back after Disk.store staged a file, the file
    must be removed and the old value left intact."""
    min_file_size = 2**15
    cache = dc.Cache(directory, disk_min_file_size=min_file_size)

    old = b'a' * (min_file_size * 4)
    cache.set('blob', old)
    _, version = cache.get('blob', version=True)
    _, _, old_filename, _ = _row(cache, 'blob')
    old_path = os.path.join(directory, old_filename)
    assert os.path.exists(old_path)

    new = b'b' * (min_file_size * 4)

    original_row_update = cache._row_update

    def boom(rowid, now, columns, new_version):
        raise RuntimeError('rollback')

    cache._row_update = boom
    try:
        with pytest.raises(RuntimeError):
            cache.cas('blob', new, version)
    finally:
        cache._row_update = original_row_update

    # Old value and its file are untouched; the staged file is gone.
    assert cache.get('blob') == old
    assert os.path.exists(old_path)
    assert cache.check(fix=False) == []
    cache.close()


def test_big_value_small_value_paths(directory):
    """Both SQLite-inline and Disk.store paths follow the version protocol."""
    cache = dc.Cache(directory, disk_min_file_size=2**15)

    cache.set('small', 42)
    value, version = cache.get('small', version=True)
    assert (value,) == (42,)
    assert cache.cas('small', 43, version) is True
    assert cache.get('small') == 43

    cache.set('big', b'q' * 2**16)
    value, version = cache.get('big', version=True)
    assert value == b'q' * 2**16
    assert cache.cas('big', b'r' * 2**16, version) is True
    assert cache.get('big') == b'r' * 2**16
    assert cache.check(fix=False) == []
    cache.close()


def test_versions_are_global_and_monotonic_across_keys(cache):
    seen = []
    for count in range(3):
        cache.set('k%d' % count, count)
        seen.append(cache.get('k%d' % count, version=True)[1])
    assert seen == sorted(seen) and len(set(seen)) == 3

    _, version = cache.get('k0', version=True)
    assert cache.cas('k0', 42, version) is True
    new_version = cache.get('k0', version=True)[1]
    assert new_version > max(seen)


def test_existing_set_get_need_no_version(cache):
    cache['plain'] = {'x': 1}
    assert cache['plain'] == {'x': 1}
    assert cache.get('plain') == {'x': 1}


def test_old_database_schema_migrates(tmp_path):
    # Create a row with the legacy schema (no version column), then reopen.
    import sqlite3

    path = str(tmp_path / 'legacy')
    os.makedirs(path)
    con = sqlite3.connect(os.path.join(path, 'cache.db'))
    con.execute(
        'CREATE TABLE Settings (key TEXT NOT NULL UNIQUE, value)'
    )
    for setting in ('count', 'size', 'hits', 'misses', 'version'):
        con.execute(
            'INSERT OR IGNORE INTO Settings VALUES (?, ?)', (setting, 0)
        )
    con.execute(
        'CREATE TABLE Cache ('
        ' rowid INTEGER PRIMARY KEY,'
        ' key BLOB, raw INTEGER, store_time REAL, expire_time REAL,'
        ' access_time REAL, access_count INTEGER DEFAULT 0, tag BLOB,'
        ' size INTEGER DEFAULT 0, mode INTEGER DEFAULT 0,'
        ' filename TEXT, value BLOB)'
    )
    con.execute(
        "INSERT INTO Cache(key, raw, store_time, access_time, mode, value)"
        " VALUES ('old', 1, 0, 0, 1, 'legacy')"
    )
    con.commit()
    con.close()

    cache = dc.Cache(path)
    assert cache.get('old') == 'legacy'
    assert cache.get('old', version=True)[1] == 0  # migrated row

    # Zero can never be allocated by the sequence, so it never CAS-matches a
    # newly written version.
    cache.set('old', 'new')
    _, version = cache.get('old', version=True)
    assert version >= 1
    assert cache.cas('old', 'x', 0) is False
    cache.close()


def _row(cache, key):
    db_key, raw = cache._disk.put(key)
    rows = cache._sql(
        'SELECT version, mode, filename, value FROM Cache'
        ' WHERE key = ? AND raw = ?',
        (db_key, raw),
    ).fetchall()
    assert len(rows) == 1
    return rows[0]
