"""Test diskcache backup and restore."""

import hashlib
import json
import multiprocessing as mp
import os
import os.path as op
import shutil

import pytest

import diskcache as dc


def make_value(size=2**17):
    """Self-validating large value stored in a value file."""
    body = os.urandom(size)
    return body + hashlib.sha256(body).digest()


def check_value(value):
    """Return True if the value's embedded checksum matches."""
    body, digest = value[:-32], value[-32:]
    return hashlib.sha256(body).digest() == digest


def writer_process(directory, stop, fanout):
    """Continuously replace large values and delete keys."""
    if fanout:
        cache = dc.FanoutCache(directory, shards=4, disk_min_file_size=1024)
    else:
        cache = dc.Cache(directory, disk_min_file_size=1024)

    keys = ['key-%02d' % num for num in range(16)]
    count = 0

    while not stop.is_set():
        key = keys[count % len(keys)]
        cache.set(key, make_value(), retry=True)
        if count % 5 == 0:
            cache.delete(keys[(count + 3) % len(keys)], retry=True)
        count += 1

    cache.close()


@pytest.fixture
def cache(tmp_path):
    directory = str(tmp_path / 'cache')
    with dc.Cache(directory, disk_min_file_size=1024) as cache:
        yield cache


def test_backup_restore_roundtrip(cache, tmp_path):
    for num in range(10):
        cache.set('small-%d' % num, num, retry=True)
        cache.set('large-%d' % num, make_value(), retry=True)

    backup = str(tmp_path / 'backup')
    cache.backup(backup)

    assert op.isfile(op.join(backup, 'cache.db'))
    assert op.isfile(op.join(backup, 'manifest.json'))

    restored = dc.Cache.restore(backup, str(tmp_path / 'restored'))

    with restored:
        assert len(restored) == len(cache)
        for num in range(10):
            assert restored['small-%d' % num] == num
            assert check_value(restored['large-%d' % num])
        restored.check()


def test_backup_manifest_contents(cache, tmp_path):
    cache.set('large', make_value(), retry=True)
    backup = str(tmp_path / 'backup')
    cache.backup(backup)

    with open(op.join(backup, 'manifest.json'), encoding='utf-8') as reader:
        manifest = json.load(reader)

    assert manifest['format'] == 'diskcache-backup'
    assert manifest['version'] == 1
    assert len(manifest['files']) == 1

    for entry in [manifest['db']] + manifest['files']:
        path = op.join(backup, entry['path'])
        assert op.getsize(path) == entry['size']
        digest = hashlib.sha256()
        with open(path, 'rb') as file_reader:
            digest.update(file_reader.read())
        assert digest.hexdigest() == entry['sha256']


def test_backup_existing_directory_raises(cache, tmp_path):
    existing = tmp_path / 'existing'
    existing.mkdir()

    with pytest.raises(FileExistsError):
        cache.backup(str(existing))


def test_backup_in_transaction_raises(cache, tmp_path):
    with cache.transact(retry=True):
        with pytest.raises(ValueError):
            cache.backup(str(tmp_path / 'backup'))


def test_restore_incomplete_backup_raises(cache, tmp_path):
    # A temporary staging directory is not a completed backup.
    staging = tmp_path / 'backup.tmp-1234-abcd'
    staging.mkdir()
    shutil.copy(
        op.join(cache.directory, 'cache.db'), str(staging / 'cache.db')
    )

    with pytest.raises(ValueError):
        dc.Cache.restore(str(staging), str(tmp_path / 'restored'))

    assert not op.lexists(str(tmp_path / 'restored'))


def test_restore_missing_value_file_raises(cache, tmp_path):
    cache.set('large', make_value(), retry=True)
    backup = str(tmp_path / 'backup')
    cache.backup(backup)

    with open(op.join(backup, 'manifest.json'), encoding='utf-8') as reader:
        manifest = json.load(reader)

    os.remove(op.join(backup, manifest['files'][0]['path']))

    with pytest.raises(FileNotFoundError):
        dc.Cache.restore(backup, str(tmp_path / 'restored'))

    assert not op.lexists(str(tmp_path / 'restored'))


def test_restore_checksum_mismatch_raises(cache, tmp_path):
    cache.set('large', make_value(), retry=True)
    backup = str(tmp_path / 'backup')
    cache.backup(backup)

    with open(op.join(backup, 'manifest.json'), encoding='utf-8') as reader:
        manifest = json.load(reader)

    path = op.join(backup, manifest['files'][0]['path'])
    with open(path, 'r+b') as writer:
        writer.write(b'X')

    with pytest.raises(ValueError):
        dc.Cache.restore(backup, str(tmp_path / 'restored'))

    assert not op.lexists(str(tmp_path / 'restored'))


def test_restore_refuses_existing_cache(cache, tmp_path):
    cache.set('large', make_value(), retry=True)
    backup = str(tmp_path / 'backup')
    cache.backup(backup)

    target = str(tmp_path / 'target')
    with dc.Cache(target) as existing:
        existing.set('keep', 'me', retry=True)

    with pytest.raises(ValueError):
        dc.Cache.restore(backup, target)

    with dc.Cache(target) as existing:
        assert existing['keep'] == 'me'


def test_restore_into_empty_directory(cache, tmp_path):
    cache.set('large', make_value(), retry=True)
    backup = str(tmp_path / 'backup')
    cache.backup(backup)

    target = tmp_path / 'empty'
    target.mkdir()
    restored = dc.Cache.restore(backup, str(target))

    with restored:
        assert check_value(restored['large'])


def test_fanout_backup_restore_roundtrip(tmp_path):
    directory = str(tmp_path / 'cache')
    with dc.FanoutCache(directory, shards=4, disk_min_file_size=1024) as cache:
        for num in range(40):
            cache.set('key-%d' % num, make_value(), retry=True)

        backup = str(tmp_path / 'backup')
        cache.backup(backup)

    assert op.isfile(op.join(backup, 'manifest.json'))

    with open(op.join(backup, 'manifest.json'), encoding='utf-8') as reader:
        manifest = json.load(reader)

    assert manifest['format'] == 'diskcache-fanout-backup'
    assert manifest['shards'] == 4

    for num in range(4):
        assert op.isfile(op.join(backup, '%03d' % num, 'manifest.json'))

    restored = dc.FanoutCache.restore(backup, str(tmp_path / 'restored'))

    with restored:
        assert len(restored) == 40
        for num in range(40):
            assert check_value(restored['key-%d' % num])


def test_fanout_restore_refuses_existing_cache(tmp_path):
    directory = str(tmp_path / 'cache')
    with dc.FanoutCache(directory, shards=4, disk_min_file_size=1024) as cache:
        cache.set('key', make_value(), retry=True)
        cache.backup(str(tmp_path / 'backup'))

    with pytest.raises(ValueError):
        dc.FanoutCache.restore(
            str(tmp_path / 'backup'), str(tmp_path / 'cache')
        )


def check_restored(backup, restored, fanout):
    """Restore and verify every key reads back complete content."""
    if fanout:
        cache = dc.FanoutCache.restore(backup, restored)
    else:
        cache = dc.Cache.restore(backup, restored)

    with cache:
        count = len(cache)
        assert count > 0
        for key in cache:
            assert check_value(cache.get(key, retry=True))
    return count


@pytest.mark.parametrize('fanout', [False, True])
def test_concurrent_backup_restore(tmp_path, fanout):
    directory = str(tmp_path / 'cache')

    if fanout:
        cache = dc.FanoutCache(directory, shards=4, disk_min_file_size=1024)
    else:
        cache = dc.Cache(directory, disk_min_file_size=1024)

    for num in range(16):
        cache.set('key-%02d' % num, make_value(), retry=True)

    stop = mp.Event()
    writer = mp.Process(target=writer_process, args=(directory, stop, fanout))
    writer.start()

    try:
        for num in range(3):
            backup = str(tmp_path / ('backup-%d' % num))
            restored = str(tmp_path / ('restored-%d' % num))
            cache.backup(backup)
            check_restored(backup, restored, fanout)
    finally:
        stop.set()
        writer.join()
        cache.close()

    assert writer.exitcode == 0
