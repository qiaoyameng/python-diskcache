"""Persistent scheduled task box.

Lightweight delay/priority queue persisted in a :class:`Cache` database. No
Redis, no background threads: callers drive scheduling themselves.
"""

import time

from .core import ENOVAL, Cache

# Payloads are stored in the cache under keys with this prefix to avoid
# collisions with other cache users.

_PAYLOAD_PREFIX = 'diskcache-taskbox'

_SCHEMA = (
    (
        'CREATE TABLE IF NOT EXISTS TaskBox ('
        ' name TEXT NOT NULL,'
        ' id BLOB NOT NULL,'
        ' raw INTEGER NOT NULL,'
        ' eta REAL NOT NULL,'
        ' priority REAL NOT NULL,'
        ' seq INTEGER NOT NULL,'
        ' PRIMARY KEY (name, id)'
        ' )'
    ),
    (
        'CREATE INDEX IF NOT EXISTS TaskBox_due ON'
        ' TaskBox (name, eta, priority, seq)'
    ),
    (
        'CREATE TABLE IF NOT EXISTS TaskBoxMeta ('
        ' name TEXT PRIMARY KEY,'
        ' seq INTEGER NOT NULL'
        ' )'
    ),
)


class TaskBox:
    """Persistent queue of scheduled tasks with stable identifiers.

    Each item has a caller-supplied stable ID, a due time (`eta`, in the
    injected clock's timebase), and a priority. Items become claimable once
    the clock reaches their `eta`. Claims are ordered by descending priority
    and then by ascending enqueue sequence (first-in, first-out among
    equals). Items with a high priority that are not yet due never block
    items that are already due.

    Updates via `reschedule` and `reprioritize` *preserve* the item's
    original enqueue sequence number, so the first-in, first-out tie-break
    among equal priorities is always the original enqueue order, never the
    update time.

    Claims are atomic: the selection and deletion of the next due item
    happen inside a single SQLite transaction, so multiple processes may
    safely claim from the same directory. Payload values are stored in the
    underlying cache; when a payload is large enough to be stored as a file
    on disk, the file is read inside the claim transaction and unlinked only
    after the transaction commits (never earlier).

    Delivery boundary: `claim` is atomic, but once an item is claimed it is
    gone. If the claiming process crashes while executing the task, the item
    is *not* re-queued automatically. This interface provides at-most-once
    delivery after claim; callers needing retries must re-`put` the item
    themselves.

    No background threads are started. Use `next_wakeup` to learn when the
    next item becomes due and schedule your own sleep/timer.

    >>> class Clock:
    ...     def __init__(self):
    ...         self.now = 1000.0
    ...     def __call__(self):
    ...         return self.now
    >>> clock = Clock()
    >>> box = TaskBox(clock=clock)
    >>> box.put('low', 'alpha', priority=1)
    'low'
    >>> box.put('high', 'beta', priority=9, delay=60.0)
    'high'
    >>> box.claim()  # High priority item is not yet due and does not block.
    ('low', 'alpha')
    >>> box.claim() is None
    True
    >>> box.next_wakeup()
    1060.0
    >>> clock.now = 1060.0
    >>> box.claim()
    ('high', 'beta')

    """

    def __init__(self, directory=None, name='default', clock=time.monotonic):
        """Initialize task box instance.

        If directory is None then a temporary directory is created. The
        directory will *not* be automatically removed.

        :param directory: task box directory (default None)
        :param str name: queue name within the cache (default 'default')
        :param clock: callable returning current time (default
            `time.monotonic`)

        """
        cache = Cache(directory, eviction_policy='none')
        self._init(cache, name, clock)

    @classmethod
    def fromcache(cls, cache, name='default', clock=time.monotonic):
        """Initialize task box using `cache`.

        The cache should use ``eviction_policy='none'`` (the default for
        `TaskBox` itself) so that payloads are never culled before their
        metadata is claimed.

        >>> cache = Cache()
        >>> box = TaskBox.fromcache(cache)
        >>> box.cache is cache
        True
        >>> box.put('a', 1)
        'a'
        >>> box.claim()
        ('a', 1)

        :param Cache cache: cache to use
        :param str name: queue name within the cache (default 'default')
        :param clock: callable returning current time (default
            `time.monotonic`)
        :return: initialized TaskBox

        """
        # pylint: disable=protected-access
        self = cls.__new__(cls)
        self._init(cache, name, clock)
        return self

    def _init(self, cache, name, clock):
        self._cache = cache
        self._name = name
        self._clock = clock
        sql_retry = cache._sql_retry  # pylint: disable=protected-access
        for statement in _SCHEMA:
            sql_retry(statement)

    @property
    def cache(self):
        """Cache used by task box."""
        return self._cache

    @property
    def directory(self):
        """Directory path where task box is stored."""
        return self._cache.directory

    @property
    def name(self):
        """Queue name within the cache."""
        return self._name

    def _payload_key(self, item_id):
        return (_PAYLOAD_PREFIX, self._name, item_id)

    def _resolve_eta(self, eta, delay):
        if eta is not None and delay is not None:
            raise ValueError('specify at most one of eta and delay')
        if eta is not None:
            return float(eta)
        now = self._clock()
        if delay is not None:
            return now + delay
        return now

    def put(self, item_id, value, eta=None, delay=None, priority=0):
        """Add item with `item_id`, `value`, due time, and `priority`.

        The due time is given either absolutely as `eta` (in the clock's
        timebase) or relatively as `delay` seconds from now. With neither,
        the item is due immediately. Larger `priority` values are claimed
        first; ties break by enqueue order.

        >>> box = TaskBox()
        >>> box.put('x', 'value', priority=2)
        'x'
        >>> box.put('x', 'other')
        Traceback (most recent call last):
            ...
        ValueError: item already queued: 'x'

        :param item_id: stable identifier, unique within the queue
        :param value: payload returned by `claim`
        :param float eta: absolute due time (default None)
        :param float delay: seconds until due (default None)
        :param float priority: priority, higher claimed first (default 0)
        :return: `item_id`
        :raises ValueError: if `item_id` is already queued or both `eta`
            and `delay` are given

        """
        due = self._resolve_eta(eta, delay)
        cache = self._cache
        db_id, raw = cache._disk.put(item_id)  # pylint: disable=protected-access
        with cache.transact(retry=True):
            sql = cache._sql  # pylint: disable=protected-access
            cursor = sql(
                'UPDATE TaskBoxMeta SET seq = seq + 1 WHERE name = ?',
                (self._name,),
            )
            if cursor.rowcount == 0:
                sql(
                    'INSERT INTO TaskBoxMeta (name, seq) VALUES (?, 1)',
                    (self._name,),
                )
            (seq,) = sql(
                'SELECT seq FROM TaskBoxMeta WHERE name = ?', (self._name,)
            ).fetchone()
            cursor = sql(
                'INSERT OR IGNORE INTO TaskBox'
                ' (name, id, raw, eta, priority, seq)'
                ' VALUES (?, ?, ?, ?, ?, ?)',
                (self._name, db_id, raw, due, float(priority), seq),
            )
            if cursor.rowcount == 0:
                raise ValueError('item already queued: %r' % (item_id,))
            cache.set(self._payload_key(item_id), value)
        return item_id

    def claim(self, now=None, default=None):
        """Atomically claim and remove the next due item.

        Selects the due item with the highest priority (oldest enqueue
        order among ties) and deletes it in a single SQLite transaction, so
        concurrent processes never claim the same item twice.

        The item is removed before this method returns: if the caller
        crashes while executing the task, the item is not re-queued
        automatically (at-most-once delivery after claim).

        >>> box = TaskBox()
        >>> box.put('a', 'alpha')
        'a'
        >>> box.claim()
        ('a', 'alpha')
        >>> print(box.claim())
        None

        :param float now: current time (default None, uses the clock)
        :param default: value returned when no item is due (default None)
        :return: `(item_id, value)` pair or `default`

        """
        if now is None:
            now = self._clock()
        cache = self._cache
        with cache.transact(retry=True):
            sql = cache._sql  # pylint: disable=protected-access
            row = sql(
                'SELECT id, raw FROM TaskBox'
                ' WHERE name = ? AND eta <= ?'
                ' ORDER BY priority DESC, seq ASC'
                ' LIMIT 1',
                (self._name, now),
            ).fetchone()
            if row is None:
                return default
            db_id, raw = row
            sql(
                'DELETE FROM TaskBox WHERE name = ? AND id = ?',
                (self._name, db_id),
            )
            item_id = cache._disk.get(db_id, raw)  # pylint: disable=protected-access
            key = self._payload_key(item_id)
            # Read the payload (possibly from a value file) only after the
            # row is secured, and delete it within the same transaction.
            # Cache defers unlinking any value file until after commit.
            value = cache.get(key, default=ENOVAL)
            cache.delete(key)
        return item_id, None if value is ENOVAL else value

    def reschedule(self, item_id, eta=None, delay=None):
        """Update the due time of a queued item.

        The item keeps its original enqueue sequence number: its
        first-in, first-out tie-break position among equal priorities is
        preserved, not refreshed to the update time.

        >>> box = TaskBox()
        >>> box.put('a', 'alpha', delay=60.0)
        'a'
        >>> box.claim() is None
        True
        >>> box.reschedule('a', delay=0.0)
        True
        >>> box.claim()
        ('a', 'alpha')
        >>> box.reschedule('a', delay=0.0)
        False

        :param item_id: stable identifier of queued item
        :param float eta: absolute due time (default None)
        :param float delay: seconds until due (default None)
        :return: True if the item was found and updated
        :raises ValueError: if both `eta` and `delay` are given

        """
        due = self._resolve_eta(eta, delay)
        return self._update(item_id, 'eta', due)

    def reprioritize(self, item_id, priority):
        """Update the priority of a queued item.

        The item keeps its original enqueue sequence number: its
        first-in, first-out tie-break position among equal priorities is
        preserved, not refreshed to the update time.

        >>> box = TaskBox()
        >>> box.put('a', 'alpha')
        'a'
        >>> box.put('b', 'beta')
        'b'
        >>> box.reprioritize('b', 10)
        True
        >>> box.claim()
        ('b', 'beta')
        >>> box.reprioritize('b', 10)
        False

        :param item_id: stable identifier of queued item
        :param float priority: new priority, higher claimed first
        :return: True if the item was found and updated

        """
        return self._update(item_id, 'priority', float(priority))

    def _update(self, item_id, column, value):
        cache = self._cache
        db_id, raw = cache._disk.put(item_id)  # pylint: disable=protected-access
        statement = 'UPDATE TaskBox SET %s = ? WHERE name = ? AND id = ?'
        with cache.transact(retry=True):
            sql = cache._sql  # pylint: disable=protected-access
            cursor = sql(statement % column, (value, self._name, db_id))
            return cursor.rowcount != 0

    def cancel(self, item_id):
        """Remove a queued item without claiming it.

        The payload is deleted within the same transaction as the metadata;
        any value file on disk is unlinked only after the transaction
        commits.

        >>> box = TaskBox()
        >>> box.put('a', 'alpha')
        'a'
        >>> box.cancel('a')
        True
        >>> box.cancel('a')
        False
        >>> box.claim() is None
        True

        :param item_id: stable identifier of queued item
        :return: True if the item was found and removed

        """
        cache = self._cache
        db_id, raw = cache._disk.put(item_id)  # pylint: disable=protected-access
        with cache.transact(retry=True):
            sql = cache._sql  # pylint: disable=protected-access
            cursor = sql(
                'DELETE FROM TaskBox WHERE name = ? AND id = ?',
                (self._name, db_id),
            )
            if cursor.rowcount == 0:
                return False
            cache.delete(self._payload_key(item_id))
            return True

    def next_wakeup(self):
        """Return the clock time when the next item becomes due.

        Returns None when the queue is empty. The result may already be in
        the past if an item is due now. Callers use this to schedule their
        own sleep or timer; no background thread is started.

        >>> box = TaskBox()
        >>> box.next_wakeup() is None
        True
        >>> box.put('a', 'alpha', eta=1234.5)
        'a'
        >>> box.next_wakeup()
        1234.5

        :return: float due time or None

        """
        sql = self._cache._sql  # pylint: disable=protected-access
        row = sql(
            'SELECT MIN(eta) FROM TaskBox WHERE name = ?', (self._name,)
        ).fetchone()
        return row[0]

    def __len__(self):
        """Return count of queued items (due and not yet due).

        >>> box = TaskBox()
        >>> box.put('a', 'alpha', delay=60.0)
        'a'
        >>> box.put('b', 'beta')
        'b'
        >>> len(box)
        2

        """
        sql = self._cache._sql  # pylint: disable=protected-access
        (count,) = sql(
            'SELECT COUNT(*) FROM TaskBox WHERE name = ?', (self._name,)
        ).fetchone()
        return count

    def __contains__(self, item_id):
        """Return True if `item_id` is queued.

        >>> box = TaskBox()
        >>> box.put('a', 'alpha')
        'a'
        >>> 'a' in box
        True
        >>> 'b' in box
        False

        """
        cache = self._cache
        db_id, raw = cache._disk.put(item_id)  # pylint: disable=protected-access
        sql = cache._sql  # pylint: disable=protected-access
        row = sql(
            'SELECT 1 FROM TaskBox WHERE name = ? AND id = ?',
            (self._name, db_id),
        ).fetchone()
        return row is not None
