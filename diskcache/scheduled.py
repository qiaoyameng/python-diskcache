"""Persistent scheduled task queue."""

import math
import sqlite3
import time
import uuid

from .core import ENOVAL


class ScheduledQueue:
    """Persistent delayed priority queue.

    Items become available at an absolute time supplied by ``time_func``.
    Due items are claimed in descending priority order. Items with equal
    priority are claimed in their original enqueue order. Updating an item's
    time or priority does not change its tie-breaking sequence; cancel and
    re-enqueue an item to move it to the end of its priority level.

    Claiming an item atomically selects and deletes it in one SQLite
    transaction. If a worker crashes after an item is claimed but before its
    task finishes, the item is not automatically put back on the queue. Use a
    separate acknowledgement or lease design if redelivery is required.

    Value files, if used, are only read after the transaction commits and are
    removed after they have been read.

    >>> from diskcache import Cache, ScheduledQueue
    >>> clock = [9.0]
    >>> cache = Cache(eviction_policy='none')
    >>> tasks = ScheduledQueue(cache, time_func=lambda: clock[0])
    >>> item_id = tasks.put('send report', when=10.0, priority=5)
    >>> tasks.claim() is None
    True
    >>> tasks.next_wakeup()
    10.0
    >>> clock[0] = 10.0
    >>> tasks.claim()[1]
    'send report'

    """

    def __init__(self, cache, name='', time_func=time.time):
        if not isinstance(name, str):
            raise TypeError('name must be a string')
        self._cache = cache
        self._name = name
        self._time = time_func

        sql = cache._sql_retry  # pylint: disable=protected-access

        sql(
            'CREATE TABLE IF NOT EXISTS ScheduledQueue ('
            ' rowid INTEGER PRIMARY KEY,'
            ' queue TEXT NOT NULL,'
            ' item_id TEXT NOT NULL,'
            ' ready_time REAL NOT NULL,'
            ' priority REAL NOT NULL,'
            ' sequence INTEGER NOT NULL,'
            ' size INTEGER DEFAULT 0,'
            ' mode INTEGER DEFAULT 0,'
            ' filename TEXT,'
            ' value BLOB,'
            ' UNIQUE(queue, item_id))'
        )
        sql(
            'CREATE INDEX IF NOT EXISTS ScheduledQueue_ready'
            ' ON ScheduledQueue(queue, ready_time, priority DESC, sequence)'
        )
        sql(
            'CREATE INDEX IF NOT EXISTS ScheduledQueue_order'
            ' ON ScheduledQueue(queue, priority DESC, sequence, ready_time)'
        )

    @property
    def cache(self):
        """Cache used by queue."""
        return self._cache

    @property
    def name(self):
        """Queue namespace."""
        return self._name

    def _validate_item_id(self, item_id):
        if not isinstance(item_id, str) or not item_id:
            raise ValueError('item_id must be a non-empty string')
        return item_id

    def _validate_time(self, ready_time):
        if isinstance(ready_time, bool) or not isinstance(
            ready_time, (int, float)
        ):
            raise TypeError('when must be a number')
        if not math.isfinite(ready_time):
            raise ValueError('when must be finite')
        return float(ready_time)

    def _validate_priority(self, priority):
        if isinstance(priority, bool) or not isinstance(
            priority, (int, float)
        ):
            raise TypeError('priority must be a number')
        if not math.isfinite(priority):
            raise ValueError('priority must be finite')
        return float(priority)

    def put(self, value, when=None, priority=0, item_id=None, read=False):
        """Enqueue an item and return its stable identifier.

        :param value: task value
        :param float when: absolute ready time (default current time)
        :param float priority: larger values are claimed first
        :param str item_id: caller supplied stable identifier
        :param bool read: read value as bytes from a file-like object
        :return: item identifier
        :raises KeyError: if ``item_id`` is already queued
        """
        if item_id is None:
            item_id = uuid.uuid4().hex
        else:
            item_id = self._validate_item_id(item_id)

        if when is None:
            ready_time = self._validate_time(self._time())
        else:
            ready_time = self._validate_time(when)

        priority = self._validate_priority(priority)
        cache = self._cache
        size, mode, filename, db_value = cache._disk.store(  # pylint: disable=protected-access
            value, read, key=(self._name, item_id)
        )
        stored_filename = filename

        try:
            with cache._transact(  # pylint: disable=protected-access
                retry=True, filename=filename
            ) as context:
                sql, _ = context
                rows = sql(
                    'SELECT COALESCE(MAX(sequence), 0) + 1'
                    ' FROM ScheduledQueue WHERE queue = ?',
                    (self._name,),
                ).fetchall()
                (sequence,) = rows[0]

                try:
                    sql(
                        'INSERT INTO ScheduledQueue('
                        ' queue, item_id, ready_time, priority, sequence,'
                        ' size, mode, filename, value'
                        ') VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        (
                            self._name,
                            item_id,
                            ready_time,
                            priority,
                            sequence,
                            size,
                            mode,
                            filename,
                            db_value,
                        ),
                    )
                except sqlite3.IntegrityError:
                    if filename is not None:
                        cache._disk.remove(filename)  # pylint: disable=protected-access
                        stored_filename = None
                    raise KeyError(item_id) from None
        except Exception:
            if stored_filename is not None:
                cache._disk.remove(stored_filename)  # pylint: disable=protected-access
            raise

        return item_id

    push = put
    schedule = put
    enqueue = put

    def claim(self, default=None):
        """Claim and delete the highest-priority due item.

        The item is removed before its value is returned. A crash while
        executing the claimed task will not automatically re-enqueue it.

        :param default: value returned when no item is due
        :return: ``(item_id, value)`` tuple or ``default``
        """
        cache = self._cache
        now = self._validate_time(self._time())
        select = (
            'SELECT rowid, item_id, mode, filename, value'
            ' FROM ScheduledQueue'
            ' WHERE queue = ? AND ready_time <= ?'
            ' ORDER BY priority DESC, sequence ASC LIMIT 1'
        )

        with cache._transact(retry=True) as context:  # pylint: disable=protected-access
            sql, _ = context
            rows = sql(select, (self._name, now)).fetchall()
            if not rows:
                return default
            rowid, item_id, mode, filename, db_value = rows[0]
            sql('DELETE FROM ScheduledQueue WHERE rowid = ?', (rowid,))

        try:
            value = cache._disk.fetch(  # pylint: disable=protected-access
                mode, filename, db_value, False
            )
        finally:
            if filename is not None:
                cache._disk.remove(filename)  # pylint: disable=protected-access

        return item_id, value

    dequeue = claim
    pull = claim
    pop = claim

    def info(self, item_id, default=ENOVAL):
        """Return ``(ready_time, priority, sequence)`` for an item."""
        item_id = self._validate_item_id(item_id)
        cache = self._cache

        with cache._transact(retry=True) as context:  # pylint: disable=protected-access
            sql, _ = context
            rows = sql(
                'SELECT ready_time, priority, sequence'
                ' FROM ScheduledQueue WHERE queue = ? AND item_id = ?',
                (self._name, item_id),
            ).fetchall()

        if not rows:
            if default is ENOVAL:
                raise KeyError(item_id)
            return default
        return rows[0]

    def reschedule(self, item_id, when):
        """Change an item's absolute ready time.

        The item keeps its original tie-breaking sequence.

        :return: True if the item was updated
        """
        item_id = self._validate_item_id(item_id)
        ready_time = self._validate_time(when)

        with self._cache._transact(retry=True) as context:  # pylint: disable=protected-access
            sql, _ = context
            cursor = sql(
                'UPDATE ScheduledQueue SET ready_time = ?'
                ' WHERE queue = ? AND item_id = ?',
                (ready_time, self._name, item_id),
            )
            return cursor.rowcount > 0

    set_time = reschedule
    update_time = reschedule

    def set_priority(self, item_id, priority):
        """Change an item's priority.

        The item keeps its original tie-breaking sequence among items at the
        same priority.

        :return: True if the item was updated
        """
        item_id = self._validate_item_id(item_id)
        priority = self._validate_priority(priority)

        with self._cache._transact(retry=True) as context:  # pylint: disable=protected-access
            sql, _ = context
            cursor = sql(
                'UPDATE ScheduledQueue SET priority = ?'
                ' WHERE queue = ? AND item_id = ?',
                (priority, self._name, item_id),
            )
            return cursor.rowcount > 0

    update_priority = set_priority

    def update(self, item_id, when=ENOVAL, priority=ENOVAL):
        """Change an item's ready time and/or priority.

        Unspecified fields are unchanged. The item's original tie-breaking
        sequence is always preserved.

        :return: True if the item was updated
        """
        item_id = self._validate_item_id(item_id)
        assignments = []
        arguments = []

        if when is not ENOVAL:
            assignments.append('ready_time = ?')
            arguments.append(self._validate_time(when))

        if priority is not ENOVAL:
            assignments.append('priority = ?')
            arguments.append(self._validate_priority(priority))

        if not assignments:
            return self.info(item_id, default=None) is not None

        arguments.extend((self._name, item_id))

        with self._cache._transact(retry=True) as context:  # pylint: disable=protected-access
            sql, _ = context
            cursor = sql(
                'UPDATE ScheduledQueue SET {0}'
                ' WHERE queue = ? AND item_id = ?'.format(
                    ', '.join(assignments)
                ),
                arguments,
            )
            return cursor.rowcount > 0

    def cancel(self, item_id):
        """Cancel and delete an item.

        :return: True if the item was canceled
        """
        item_id = self._validate_item_id(item_id)
        cache = self._cache

        with cache._transact(retry=True) as context:  # pylint: disable=protected-access
            sql, cleanup = context
            rows = sql(
                'SELECT filename FROM ScheduledQueue'
                ' WHERE queue = ? AND item_id = ?',
                (self._name, item_id),
            ).fetchall()
            if not rows:
                return False
            sql(
                'DELETE FROM ScheduledQueue'
                ' WHERE queue = ? AND item_id = ?',
                (self._name, item_id),
            )
            cleanup(rows[0][0])

        return True

    remove = cancel
    delete = cancel

    def next_wakeup(self, now=None):
        """Return the absolute time of the next queued item.

        This method does not sleep or start a thread. A due item can produce a
        wake-up time less than or equal to the current time. ``None`` is
        returned when the queue is empty.
        """
        if now is None:
            now = self._time()
        else:
            now = self._validate_time(now)

        with self._cache._transact(retry=True) as context:  # pylint: disable=protected-access
            sql, _ = context
            rows = sql(
                'SELECT MIN(ready_time) FROM ScheduledQueue WHERE queue = ?',
                (self._name,),
            ).fetchall()

        (wakeup,) = rows[0]
        return wakeup

    wakeup_at = next_wakeup
    wakeup_time = next_wakeup

    def wait_time(self, now=None):
        """Return seconds until the next item, or ``None`` if empty."""
        wakeup = self.next_wakeup(now)
        if wakeup is None:
            return None
        if now is None:
            now = self._time()
        else:
            now = self._validate_time(now)
        return max(0.0, wakeup - now)

    def __len__(self):
        with self._cache._transact(retry=True) as context:  # pylint: disable=protected-access
            sql, _ = context
            rows = sql(
                'SELECT COUNT(*) FROM ScheduledQueue WHERE queue = ?',
                (self._name,),
            ).fetchall()
        return rows[0][0]
