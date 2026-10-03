"""Test diskcache.recipes."""

import shutil
import threading
import time

import pytest

import diskcache as dc


@pytest.fixture
def cache():
    with dc.Cache() as cache:
        yield cache
    shutil.rmtree(cache.directory, ignore_errors=True)


def test_averager(cache):
    nums = dc.Averager(cache, 'nums')
    for i in range(10):
        nums.add(i)
    assert nums.get() == 4.5
    assert nums.pop() == 4.5
    for i in range(20):
        nums.add(i)
    assert nums.get() == 9.5
    assert nums.pop() == 9.5


def test_lock(cache):
    state = {'num': 0}
    lock = dc.Lock(cache, 'demo')

    def worker():
        state['num'] += 1
        with lock:
            assert lock.locked()
            state['num'] += 1
            time.sleep(0.1)

    with lock:
        thread = threading.Thread(target=worker)
        thread.start()
        time.sleep(0.1)
        assert state['num'] == 1
    thread.join()
    assert state['num'] == 2


def test_rlock(cache):
    state = {'num': 0}
    rlock = dc.RLock(cache, 'demo')

    def worker():
        state['num'] += 1
        with rlock:
            with rlock:
                state['num'] += 1
                time.sleep(0.1)

    with rlock:
        thread = threading.Thread(target=worker)
        thread.start()
        time.sleep(0.1)
        assert state['num'] == 1
    thread.join()
    assert state['num'] == 2


def test_semaphore(cache):
    state = {'num': 0}
    semaphore = dc.BoundedSemaphore(cache, 'demo', value=3)

    def worker():
        state['num'] += 1
        with semaphore:
            state['num'] += 1
            time.sleep(0.1)

    semaphore.acquire()
    semaphore.acquire()
    with semaphore:
        thread = threading.Thread(target=worker)
        thread.start()
        time.sleep(0.1)
        assert state['num'] == 1
    thread.join()
    assert state['num'] == 2
    semaphore.release()
    semaphore.release()


def test_memoize_stampede(cache):
    state = {'num': 0}

    @dc.memoize_stampede(cache, 0.1)
    def worker(num):
        time.sleep(0.01)
        state['num'] += 1
        return num

    start = time.time()
    while (time.time() - start) < 1:
        worker(100)
    assert state['num'] > 0


class FakeClock:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


def _claim_worker(directory, results, barrier, clock):
    import diskcache as diskcache

    cache = diskcache.Cache(directory, eviction_policy='none')
    try:
        tasks = diskcache.ScheduledQueue(cache, time_func=clock)
        barrier.wait()
        results.put(tasks.claim())
    finally:
        cache.close()


def test_scheduled_queue_ordering(tmp_path):
    directory = str(tmp_path)
    clock = FakeClock(0)

    with dc.Cache(directory, eviction_policy='none') as first, dc.Cache(
        directory, eviction_policy='none'
    ) as second:
        writer = dc.ScheduledQueue(first, time_func=clock)
        reader = dc.ScheduledQueue(second, time_func=clock)

        writer.put('future-low', when=100, priority=1, item_id='future-low')
        writer.put(
            'future-high', when=101, priority=10, item_id='future-high'
        )

        assert reader.claim() is None
        assert reader.next_wakeup() == 100
        assert reader.wait_time() == 100

        clock.value = 100
        assert reader.claim() == ('future-low', 'future-low')

        writer.put('low-a', when=100, priority=1, item_id='low-a')
        writer.put('high-a', when=100, priority=2, item_id='high-a')
        writer.put('high-b', when=100, priority=2, item_id='high-b')

        assert reader.claim() == ('high-a', 'high-a')
        assert reader.claim() == ('high-b', 'high-b')
        assert reader.claim() == ('low-a', 'low-a')
        clock.value = 101
        assert reader.claim() == ('future-high', 'future-high')
        assert reader.claim() is None
        assert reader.next_wakeup() is None
        assert reader.wait_time() is None


def test_scheduled_queue_updates_preserve_enqueue_sequence():
    with dc.Cache(eviction_policy='none') as cache:
        clock = FakeClock(0)
        tasks = dc.ScheduledQueue(cache, time_func=clock)
        tasks.put('first', when=10, item_id='first')
        tasks.put('second', when=10, item_id='second')
        tasks.put('third', when=10, item_id='third')

        assert tasks.info('second') == (10.0, 0.0, 2)
        assert tasks.set_priority('second', 10)
        assert tasks.set_priority('second', 0)
        assert tasks.reschedule('first', 20)
        assert tasks.update('third', when=10, priority=0)
        assert tasks.info('second') == (10.0, 0.0, 2)

        clock.value = 10
        assert tasks.claim() == ('second', 'second')
        assert tasks.claim() == ('third', 'third')
        assert tasks.claim() is None
        clock.value = 20
        assert tasks.claim() == ('first', 'first')


def test_scheduled_queue_cancel_and_duplicate_id():
    with dc.Cache(eviction_policy='none') as cache:
        tasks = dc.ScheduledQueue(cache)
        tasks.put('one', item_id='one')
        assert tasks.cancel('one') is True
        assert tasks.cancel('one') is False
        tasks.put('one', item_id='one')
        with pytest.raises(KeyError):
            tasks.put('again', item_id='one')


def test_scheduled_queue_persists_with_second_cache_handle(tmp_path):
    directory = str(tmp_path)
    clock = FakeClock(0)

    with dc.Cache(directory, eviction_policy='none') as cache:
        tasks = dc.ScheduledQueue(cache, time_func=clock)
        tasks.put('hello', when=5, item_id='greeting')

    with dc.Cache(directory, eviction_policy='none') as cache:
        tasks = dc.ScheduledQueue(cache, time_func=clock)
        assert tasks.claim() is None
        clock.value = 5
        assert tasks.claim() == ('greeting', 'hello')


def test_scheduled_queue_names_are_isolated(cache):
    emails = dc.ScheduledQueue(cache, name='emails')
    reports = dc.ScheduledQueue(cache, name='reports')
    emails.put('email', item_id='same')
    reports.put('report', item_id='same')
    assert emails.claim() == ('same', 'email')
    assert reports.claim() == ('same', 'report')


def test_scheduled_queue_rejects_non_string_name(cache):
    with pytest.raises(TypeError):
        dc.ScheduledQueue(cache, name=None)


def test_scheduled_queue_check_fix_keeps_value_file(tmp_path):
    with dc.Cache(
        str(tmp_path), eviction_policy='none', disk_min_file_size=1
    ) as cache:
        tasks = dc.ScheduledQueue(cache)
        item_id = tasks.put(b'large value', item_id='large')
        filename = cache._sql(
            'SELECT filename FROM ScheduledQueue WHERE item_id = ?',
            (item_id,),
        ).fetchone()[0]
        assert cache.check(fix=True) == []
        assert (tmp_path / filename).exists()
        assert tasks.claim() == (item_id, b'large value')


def test_scheduled_queue_file_is_read_and_removed_after_commit(tmp_path):
    with dc.Cache(
        str(tmp_path), eviction_policy='none', disk_min_file_size=1
    ) as cache:
        tasks = dc.ScheduledQueue(cache)
        item_id = tasks.put(b'large value', item_id='large')
        rows = cache._sql(
            'SELECT filename FROM ScheduledQueue WHERE item_id = ?',
            (item_id,),
        ).fetchall()
        filename = rows[0][0]
        assert filename is not None
        full_path = tmp_path / filename

        original_fetch = cache._disk.fetch
        original_remove = cache._disk.remove

        def checked_fetch(mode, name, value, read):
            assert full_path.exists()
            assert cache._sql(
                'SELECT COUNT(*) FROM ScheduledQueue WHERE item_id = ?',
                (item_id,),
            ).fetchone()[0] == 0
            return original_fetch(mode, name, value, read)

        def checked_remove(name):
            assert not full_path.exists() or name == filename
            return original_remove(name)

        cache._disk.fetch = checked_fetch
        cache._disk.remove = checked_remove
        assert tasks.claim() == (item_id, b'large value')
        assert not full_path.exists()


def test_scheduled_queue_multiprocess_claim(tmp_path):
    import multiprocessing

    directory = str(tmp_path)
    ctx = multiprocessing.get_context('spawn')

    with dc.Cache(directory, eviction_policy='none') as cache:
        tasks = dc.ScheduledQueue(cache, time_func=FakeClock(100))
        for number in range(4):
            tasks.put(number, when=100, item_id='item-{0}'.format(number))

    results = ctx.Queue()
    barrier = ctx.Barrier(4)
    processes = [
        ctx.Process(
            target=_claim_worker,
            args=(directory, results, barrier, FakeClock(100)),
        )
        for _ in range(4)
    ]

    for process in processes:
        process.start()
    claimed = [results.get(timeout=30) for _ in processes]
    for process in processes:
        process.join(30)
        assert process.exitcode == 0

    assert sorted(item_id for item_id, value in claimed) == [
        'item-0',
        'item-1',
        'item-2',
        'item-3',
    ]
    assert sorted(value for item_id, value in claimed) == [0, 1, 2, 3]

    with dc.Cache(directory, eviction_policy='none') as cache:
        assert len(dc.ScheduledQueue(cache)) == 0
