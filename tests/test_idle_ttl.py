"""Test idle TTL and hard deadline renewal policy of diskcache.core.Cache."""

import io
import os
import shutil
import sqlite3
import tempfile
from unittest import mock

import pytest

import diskcache as dc


class Clock:
    """Fake clock returning a manually controlled epoch time."""

    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now

    def set(self, now):
        self.now = now

    def advance(self, delta):
        self.now += delta


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def cache(clock):
    cache = dc.Cache(clock=clock)
    # Renew on every read unless a test exercises throttling explicitly.
    cache.idle_refresh_factor = 0
    with cache:
        yield cache
    shutil.rmtree(cache.directory, ignore_errors=True)


def row(cache, key):
    """Return policy columns for `key` directly from the database."""
    db_key, raw = cache._disk.put(key)
    select = (
        'SELECT expire_time, idle_ttl, hard_expire_time, store_time, rowid'
        ' FROM Cache WHERE key = ? AND raw = ?'
    )
    rows = cache._sql(select, (db_key, raw)).fetchall()
    assert len(rows) == 1
    expire_time, idle_ttl, hard_expire_time, store_time, rowid = rows[0]
    return {
        'expire_time': expire_time,
        'idle_ttl': idle_ttl,
        'hard_expire_time': hard_expire_time,
        'store_time': store_time,
        'rowid': rowid,
    }


def test_idle_ttl_expires_when_unused(cache, clock):
    cache.set('key', 'value', idle_ttl=10)
    assert row(cache, 'key')['expire_time'] == clock.now + 10

    clock.advance(9)
    assert cache.get('key') == 'value'

    clock.advance(11)
    assert cache.get('key') is None
    assert 'key' not in cache


def test_read_renews_idle_deadline(cache, clock):
    cache.set('key', 'value', idle_ttl=10)

    for _ in range(5):
        clock.advance(6)
        assert cache.get('key') == 'value'

    # 30 seconds of wall time passed but never 10 idle seconds.
    assert row(cache, 'key')['expire_time'] == clock.now + 10

    clock.advance(11)
    assert cache.get('key') is None


def test_hard_deadline_caps_hot_item(cache, clock):
    cache.set('key', 'value', idle_ttl=10, hard_ttl=25)
    assert row(cache, 'key')['hard_expire_time'] == clock.now + 25

    while clock.now < 1_000_000.0 + 24:
        clock.advance(1)
        assert cache.get('key') == 'value'

    # Renewals are clamped to the hard deadline.
    assert row(cache, 'key')['expire_time'] == 1_000_000.0 + 25

    clock.advance(2)
    assert cache.get('key') is None


def test_effective_deadline_is_earliest(cache, clock):
    # Hard deadline earlier than idle deadline.
    cache.set('hard', 'value', idle_ttl=100, hard_ttl=10)
    assert row(cache, 'hard')['expire_time'] == clock.now + 10

    # Fixed expire earlier than idle deadline; reads must not extend it.
    cache.set('fixed', 'value', expire=10, idle_ttl=100)

    clock.advance(5)
    assert cache.get('hard') == 'value'
    assert cache.get('fixed') == 'value'
    assert row(cache, 'fixed')['expire_time'] == 1_000_000.0 + 10

    clock.advance(6)
    assert cache.get('hard') is None
    assert cache.get('fixed') is None


def test_hard_ttl_without_idle_ttl(cache, clock):
    cache.set('key', 'value', hard_ttl=10)
    assert row(cache, 'key')['idle_ttl'] is None

    clock.advance(5)
    assert cache.get('key') == 'value'
    # No idle policy: reads do not renew.
    assert row(cache, 'key')['expire_time'] == 1_000_000.0 + 10

    clock.advance(6)
    assert cache.get('key') is None


def test_missing_and_expired_reads_do_not_renew(cache, clock):
    cache.set('key', 'value', idle_ttl=10)
    before = row(cache, 'key')['expire_time']

    # Missing key.
    assert cache.get('other') is None

    # Expired key.
    clock.advance(11)
    assert cache.get('key') is None
    assert row(cache, 'key')['expire_time'] == before


def test_failed_value_read_does_not_renew(cache, clock):
    cache.set('key', 'value', idle_ttl=10)
    before = row(cache, 'key')['expire_time']

    fetch = mock.Mock(side_effect=IOError)
    with mock.patch.object(cache._disk, 'fetch', fetch):
        assert cache.get('key') is None

    assert row(cache, 'key')['expire_time'] == before


def test_accessors_consistently_renew(cache, clock):
    accessors = [
        lambda key: cache.get(key),
        lambda key: cache[key],
        lambda key: cache.get(key, expire_time=True),
        lambda key: cache.get(key, tag=True),
    ]

    for index, accessor in enumerate(accessors):
        key = 'key-%d' % index
        cache.set(key, 'value', idle_ttl=10, tag='tag')

        clock.advance(6)
        accessor(key)
        assert row(cache, key)['expire_time'] == clock.now + 10

        clock.advance(6)
        assert cache.get(key) == 'value'

        clock.advance(11)
        assert cache.get(key) is None
        clock.advance(10)


def test_read_accessor_renews(cache, clock):
    cache.set('key', io.BytesIO(b'value'), read=True, idle_ttl=10)

    clock.advance(6)
    with cache.read('key') as handle:
        assert handle.read() == b'value'
    assert row(cache, 'key')['expire_time'] == clock.now + 10

    clock.advance(6)
    assert cache.get('key') == b'value'

    clock.advance(11)
    assert cache.get('key') is None


def test_statistics_slow_path_renews(cache, clock):
    cache.stats(enable=True)
    cache.set('key', 'value', idle_ttl=10)

    clock.advance(6)
    assert cache.get('key') == 'value'
    assert row(cache, 'key')['expire_time'] == clock.now + 10
    assert cache.stats() == (1, 0)


def test_peek_and_peekitem_do_not_renew(cache, clock):
    cache.set('item', 'value', idle_ttl=10)
    cache.push('queued', idle_ttl=10)
    before_item = row(cache, 'item')['expire_time']
    before_queued = row(cache, 500_000_000_000_000)['expire_time']

    clock.advance(6)
    assert cache.peekitem(last=False) == ('item', 'value')
    assert cache.peek() == (500_000_000_000_000, 'queued')

    assert row(cache, 'item')['expire_time'] == before_item
    assert row(cache, 500_000_000_000_000)['expire_time'] == before_queued

    clock.advance(11)
    assert cache.get('item') is None


def test_contains_does_not_renew(cache, clock):
    cache.set('key', 'value', idle_ttl=10)
    before = row(cache, 'key')['expire_time']

    clock.advance(6)
    assert 'key' in cache
    assert row(cache, 'key')['expire_time'] == before


def test_renewal_does_not_clobber_concurrent_set(cache, clock):
    cache.set('key', 'old', idle_ttl=10)
    stale = row(cache, 'key')

    # Concurrent writer stores a newer version of the key.
    clock.advance(1)
    cache.set('key', 'new', idle_ttl=50)
    fresh = row(cache, 'key')
    assert fresh['expire_time'] == clock.now + 50

    # A late renewal based on the stale row version must not apply.
    cache._renew_idle(
        cache._sql,
        stale['rowid'],
        stale['store_time'],
        stale['expire_time'],
        stale['idle_ttl'],
        stale['hard_expire_time'],
    )
    assert row(cache, 'key')['expire_time'] == fresh['expire_time']
    assert cache.get('key') == 'new'


def test_renewal_throttled_with_bounded_error(cache, clock):
    cache.idle_refresh_factor = 0.1  # Threshold: 10% of the idle TTL.
    cache.set('key', 'value', idle_ttl=10)
    assert row(cache, 'key')['expire_time'] == 1_000_000.0 + 10

    # Read 0.5s in: below the 1s threshold, no database write occurs.
    clock.advance(0.5)
    assert cache.get('key') == 'value'
    assert row(cache, 'key')['expire_time'] == 1_000_000.0 + 10

    # Read 1.5s in: threshold crossed, deadline renewed.
    clock.advance(1.0)
    assert cache.get('key') == 'value'
    assert row(cache, 'key')['expire_time'] == 1_000_000.0 + 1.5 + 10


def test_clock_rollback_does_not_extend_hard_deadline(cache, clock):
    cache.set('key', 'value', idle_ttl=10, hard_ttl=100)
    before = row(cache, 'key')
    assert before['hard_expire_time'] == 1_000_000.0 + 100

    # Clock rolls back an hour (e.g. NTP step or DST correction).
    clock.set(1_000_000.0 - 3600)
    assert cache.get('key') == 'value'

    after = row(cache, 'key')
    assert after['hard_expire_time'] == before['hard_expire_time']
    # The rolled-back clock must not push the idle deadline backwards or
    # forwards; the stored deadline is left untouched.
    assert after['expire_time'] == before['expire_time']

    # Item still expires at the original deadlines.
    clock.set(1_000_000.0 + 11)
    assert cache.get('key') is None


def test_legacy_expire_semantics_preserved(cache, clock):
    cache.set('fixed', 'value', expire=10)
    cache.set('plain', 'value')

    assert row(cache, 'fixed')['idle_ttl'] is None
    assert row(cache, 'fixed')['hard_expire_time'] is None
    assert row(cache, 'plain')['expire_time'] is None

    # Reads of fixed-expiry items never renew the deadline.
    clock.advance(6)
    assert cache.get('fixed') == 'value'
    assert row(cache, 'fixed')['expire_time'] == 1_000_000.0 + 10

    clock.advance(6)
    assert cache.get('fixed') is None
    assert cache.get('plain') == 'value'


def test_get_with_metadata_returns_effective_expire_time(cache, clock):
    cache.set('key', 'value', idle_ttl=10, hard_ttl=100, tag='tag')

    value, expire_time, tag = cache.get('key', expire_time=True, tag=True)
    assert value == 'value'
    assert tag == 'tag'
    assert expire_time == 1_000_000.0 + 10  # Earlier of idle and hard.


def test_add_and_push_support_policy(cache, clock):
    assert cache.add('added', 'value', idle_ttl=10, hard_ttl=100)
    assert row(cache, 'added')['idle_ttl'] == 10
    assert row(cache, 'added')['hard_expire_time'] == clock.now + 100

    key = cache.push('queued', idle_ttl=10)
    assert row(cache, key)['idle_ttl'] == 10

    clock.advance(6)
    assert cache.get('added') == 'value'
    assert row(cache, 'added')['expire_time'] == clock.now + 10


def test_expire_and_cull_remove_idle_expired(cache, clock):
    cache.reset('cull_limit', 0)
    cache.set('key', 'value', idle_ttl=10, hard_ttl=100)

    clock.advance(11)
    assert cache.expire() == 1
    assert len(cache) == 0


def test_migration_from_legacy_database(clock):
    directory = tempfile.mkdtemp(prefix='diskcache-legacy-')
    try:
        # Build a database with the pre-policy schema by hand.
        con = sqlite3.connect(os.path.join(directory, 'cache.db'))
        con.execute(
            'CREATE TABLE Cache ('
            ' rowid INTEGER PRIMARY KEY,'
            ' key BLOB,'
            ' raw INTEGER,'
            ' store_time REAL,'
            ' expire_time REAL,'
            ' access_time REAL,'
            ' access_count INTEGER DEFAULT 0,'
            ' tag BLOB,'
            ' size INTEGER DEFAULT 0,'
            ' mode INTEGER DEFAULT 0,'
            ' filename TEXT,'
            ' value BLOB)'
        )
        con.execute(
            'INSERT INTO Cache'
            ' (key, raw, store_time, expire_time, mode, value)'
            ' VALUES (?, ?, ?, ?, ?, ?)',
            ('plain', 1, clock.now, None, 1, 'value'),
        )
        con.execute(
            'INSERT INTO Cache'
            ' (key, raw, store_time, expire_time, mode, value)'
            ' VALUES (?, ?, ?, ?, ?, ?)',
            ('fixed', 1, clock.now, clock.now + 10, 1, 'value'),
        )
        con.commit()
        con.close()

        with dc.Cache(directory, clock=clock) as cache:
            # Policy columns were added by the migration.
            columns = {
                row[1]
                for row in cache._sql('PRAGMA table_info(Cache)').fetchall()
            }
            assert 'idle_ttl' in columns
            assert 'hard_expire_time' in columns

            # Historical items carry no policy and are never renewed.
            assert row(cache, 'plain')['idle_ttl'] is None
            assert row(cache, 'fixed')['idle_ttl'] is None

            clock.advance(6)
            assert cache.get('plain') == 'value'
            assert cache.get('fixed') == 'value'
            assert row(cache, 'fixed')['expire_time'] == 1_000_000.0 + 10

            clock.advance(6)
            assert cache.get('fixed') is None
            assert cache.get('plain') == 'value'

            # New items may use the policy on a migrated database.
            cache.set('new', 'value', idle_ttl=10)
            assert row(cache, 'new')['idle_ttl'] == 10
            cache.check()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_fanout_supports_policy(clock):
    fanout = dc.FanoutCache(clock=clock)
    try:
        fanout.set('key', 'value', idle_ttl=10, hard_ttl=100)
        clock.advance(6)
        assert fanout.get('key') == 'value'
        clock.advance(6)
        assert fanout.get('key') == 'value'  # Renewed by previous reads.
        clock.advance(11)
        assert fanout.get('key') is None
    finally:
        shutil.rmtree(fanout.directory, ignore_errors=True)
