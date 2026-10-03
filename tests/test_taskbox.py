"""Test diskcache.taskbox.TaskBox."""

import multiprocessing as mp
import os
import shutil
import tempfile

import pytest

import diskcache as dc


class Clock:
    """Injectable clock for deterministic scheduling tests."""

    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def rmdir(directory):
    try:
        shutil.rmtree(directory)
    except OSError:
        pass


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def box(clock):
    box = dc.TaskBox(clock=clock)
    yield box
    rmdir(box.directory)


def test_put_claim_order(box):
    # First-in, first-out among equal priorities.
    box.put('a', 1)
    box.put('b', 2)
    box.put('c', 3)
    assert box.claim() == ('a', 1)
    assert box.claim() == ('b', 2)
    assert box.claim() == ('c', 3)
    assert box.claim() is None


def test_priority_order(box):
    box.put('low', 1, priority=1)
    box.put('high', 2, priority=9)
    box.put('mid', 3, priority=5)
    assert box.claim() == ('high', 2)
    assert box.claim() == ('mid', 3)
    assert box.claim() == ('low', 1)


def test_future_high_priority_does_not_block(box, clock):
    box.put('due', 'alpha', priority=0)
    box.put('future', 'beta', priority=100, delay=60.0)
    assert box.claim() == ('due', 'alpha')
    assert box.claim() is None
    clock.now += 60.0
    assert box.claim() == ('future', 'beta')


def test_eta_and_delay_conflict(box):
    with pytest.raises(ValueError):
        box.put('a', 1, eta=1.0, delay=1.0)
    with pytest.raises(ValueError):
        box.reschedule('a', eta=1.0, delay=1.0)


def test_duplicate_id(box):
    box.put('a', 1)
    with pytest.raises(ValueError):
        box.put('a', 2)
    # Failed put leaves the original item intact.
    assert box.claim() == ('a', 1)
    box.put('a', 2)
    assert box.claim() == ('a', 2)


def test_reschedule(box, clock):
    box.put('a', 'alpha', delay=60.0)
    assert box.claim() is None
    assert box.reschedule('a', delay=0.0)
    assert box.claim() == ('a', 'alpha')
    assert not box.reschedule('a', delay=0.0)


def test_reschedule_preserves_sequence(box):
    # Updates keep the original enqueue sequence: the FIFO tie-break among
    # equal priorities is the original enqueue order, not the update time.
    box.put('a', 'alpha')
    box.put('b', 'beta')
    assert box.reschedule('a', delay=60.0)
    assert box.reschedule('a', delay=0.0)
    assert box.claim() == ('a', 'alpha')
    assert box.claim() == ('b', 'beta')


def test_reprioritize_preserves_sequence(box):
    box.put('a', 'alpha')
    box.put('b', 'beta')
    box.put('c', 'gamma')
    # Bump 'a' and 'b' to the same priority; 'a' was enqueued first and its
    # sequence number is preserved, so it is still claimed before 'b'.
    assert box.reprioritize('b', 5)
    assert box.reprioritize('a', 5)
    assert box.claim() == ('a', 'alpha')
    assert box.claim() == ('b', 'beta')
    assert box.claim() == ('c', 'gamma')
    assert not box.reprioritize('a', 5)


def test_cancel(box):
    box.put('a', 'alpha')
    box.put('b', 'beta')
    assert box.cancel('a')
    assert not box.cancel('a')
    assert len(box) == 1
    assert box.claim() == ('b', 'beta')


def test_next_wakeup(box, clock):
    assert box.next_wakeup() is None
    box.put('a', 1, delay=30.0)
    box.put('b', 2, delay=10.0)
    assert box.next_wakeup() == 1010.0
    clock.now = 1010.0
    assert box.claim() == ('b', 2)
    assert box.next_wakeup() == 1030.0
    box.cancel('a')
    assert box.next_wakeup() is None


def test_len_and_contains(box):
    assert len(box) == 0
    box.put('a', 1, delay=60.0)
    box.put('b', 2)
    assert len(box) == 2
    assert 'a' in box
    assert 'c' not in box
    box.claim()
    assert len(box) == 1
    assert 'b' not in box


def test_two_cache_handles(clock):
    directory = tempfile.mkdtemp()
    try:
        cache1 = dc.Cache(directory, eviction_policy='none')
        cache2 = dc.Cache(directory, eviction_policy='none')
        box1 = dc.TaskBox.fromcache(cache1, clock=clock)
        box2 = dc.TaskBox.fromcache(cache2, clock=clock)

        box1.put('a', 'alpha', priority=1)
        box2.put('b', 'beta', priority=2, delay=60.0)
        box1.put('c', 'gamma', priority=2)

        # Enqueued through two handles; claimed interleaved across handles.
        assert box2.claim() == ('c', 'gamma')
        assert box1.claim() == ('a', 'alpha')
        assert box2.claim() is None
        clock.now += 60.0
        assert box1.claim() == ('b', 'beta')
        assert len(box2) == 0

        cache1.close()
        cache2.close()
    finally:
        rmdir(directory)


def test_large_value_file_cleanup(clock):
    directory = tempfile.mkdtemp()
    try:
        cache = dc.Cache(
            directory, disk_min_file_size=64, eviction_policy='none'
        )
        box = dc.TaskBox.fromcache(cache, clock=clock)
        payload = 'x' * 10000
        box.put('big', payload)

        def value_files():
            return [
                name
                for _, _, names in os.walk(directory)
                for name in names
                if name.endswith('.val')
            ]

        assert len(value_files()) == 1
        assert box.claim() == ('big', payload)
        # The value file is unlinked after the claim transaction commits.
        assert value_files() == []
        cache.close()
    finally:
        rmdir(directory)


def _claim_worker(directory, results):
    cache = dc.Cache(directory, eviction_policy='none')
    box = dc.TaskBox.fromcache(cache)
    claimed = []
    while True:
        item = box.claim()
        if item is None:
            break
        claimed.append(item[0])
    results.put(claimed)
    cache.close()


def test_multiprocess_claim_contention():
    directory = tempfile.mkdtemp()
    try:
        cache = dc.Cache(directory, eviction_policy='none')
        box = dc.TaskBox.fromcache(cache)
        count = 200
        for num in range(count):
            box.put('item-%d' % num, num, priority=num % 3)
        cache.close()

        try:
            context = mp.get_context('fork')
        except ValueError:  # pragma: no cover
            context = mp.get_context()

        results = context.Queue()
        workers = [
            context.Process(target=_claim_worker, args=(directory, results))
            for _ in range(4)
        ]
        for worker in workers:
            worker.start()
        claimed = []
        for _ in workers:
            claimed.extend(results.get(timeout=60))
        for worker in workers:
            worker.join(timeout=60)
            assert worker.exitcode == 0

        # Every item claimed exactly once despite concurrent claimers.
        assert len(claimed) == count
        assert len(set(claimed)) == count
        assert set(claimed) == {'item-%d' % num for num in range(count)}
    finally:
        rmdir(directory)
