"""Test diskcache backup and restore."""

import hashlib
import json
import multiprocessing as mp
import os
import os.path as op
import shutil
import struct
import tempfile
import threading
from unittest import mock

import pytest

import diskcache as dc


@pytest.fixture
def cache():
    with dc.Cache(disk_min_file_size=1024) as cache:
        yield cache
    shutil.rmtree(cache.directory, ignore_errors=True)


@pytest.fixture
def directory():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


def make_value(key, generation, size=2**17):
    """Deterministic large value verifiable from its embedded header."""
    seed = struct.pack('!qq', key, generation)
    chunk = hashlib.sha256(seed).digest()
    body = (chunk * (size // len(chunk) + 1))[: size - len(seed)]
    return seed + body


def check_value(value):
    """Return True if value is complete and matches its embedded header."""
    if not isinstance(value, bytes) or len(value) < 16:
        return False
    key, generation = struct.unpack('!qq', value[:16])
    return value == make_value(key, generation, len(value))


def read_manifest(path):
    with open(op.join(path, 'manifest.json'), encoding='utf-8') as reader:
        return json.load(reader)


def test_backup_restore(cache, directory):
    for num in range(10):
        cache['small-%d' % num] = 'value-%d' % num
        cache['large-%d' % num] = make_value(num, 0)

    path = op.join(directory, 'backup')
    cache.backup(path)

    manifest = read_manifest(path)
    assert manifest['format'] == 'diskcache-backup'
    assert manifest['version'] == 1
    assert manifest['database']['name'] == 'cache.db'
    assert len(manifest['files']) == 10

    for name, meta in manifest['files'].items():
        full_path = op.join(path, name)
        assert op.isfile(full_path)
        assert meta['size'] == op.getsize(full_path)
        assert len(meta['sha256']) == 64

    restored = dc.Cache.restore(path, op.join(directory, 'restored'))

    try:
        assert len(restored) == 20
        assert not restored.check()

        for num in range(10):
            assert restored['small-%d' % num] == 'value-%d' % num
            assert check_value(restored['large-%d' % num])

        # Restored cache remains writable.

        restored['new'] = 'item'
        assert restored['new'] == 'item'
    finally:
        restored.close()


def test_backup_excludes_unreferenced_files(cache, directory):
    cache['kept'] = make_value(0, 0)
    cache['deleted'] = make_value(1, 0)
    cache.delete('deleted')

    path = op.join(directory, 'backup')
    cache.backup(path)

    manifest = read_manifest(path)
    assert len(manifest['files']) == 1

    restored = dc.Cache.restore(path, op.join(directory, 'restored'))

    try:
        assert set(restored) == {'kept'}
        assert not restored.check()
    finally:
        restored.close()


def test_backup_path_exists(cache, directory):
    with pytest.raises(FileExistsError):
        cache.backup(directory)


def test_backup_within_transaction(cache, directory):
    with cache.transact():
        with pytest.raises(dc.BackupError):
            cache.backup(op.join(directory, 'backup'))


def test_restore_missing_manifest(cache, directory):
    cache['key'] = 'value'

    # A bare copy of the database is not a completed backup.

    bare = op.join(directory, 'bare')
    os.mkdir(bare)
    shutil.copyfile(
        op.join(cache.directory, 'cache.db'), op.join(bare, 'cache.db')
    )

    with pytest.raises(dc.BackupError):
        dc.Cache.restore(bare, op.join(directory, 'restored'))

    with pytest.raises(dc.BackupError):
        dc.Cache.restore(
            op.join(directory, 'missing'), op.join(directory, 'restored')
        )


def test_restore_missing_file(cache, directory):
    cache['large'] = make_value(0, 0)

    path = op.join(directory, 'backup')
    cache.backup(path)

    (name,) = read_manifest(path)['files']
    os.remove(op.join(path, name))

    with pytest.raises(dc.BackupError):
        dc.Cache.restore(path, op.join(directory, 'restored'))

    # Failed restore leaves the target untouched.

    assert not op.exists(op.join(directory, 'restored'))


def test_restore_checksum_mismatch(cache, directory):
    cache['large'] = make_value(0, 0)

    path = op.join(directory, 'backup')
    cache.backup(path)

    (name,) = read_manifest(path)['files']

    with open(op.join(path, name), 'r+b') as writer:
        writer.write(b'corrupted')

    with pytest.raises(dc.BackupError):
        dc.Cache.restore(path, op.join(directory, 'restored'))

    with open(op.join(path, 'cache.db'), 'r+b') as writer:
        writer.seek(4096)
        writer.write(b'corrupted')

    with pytest.raises(dc.BackupError):
        dc.Cache.restore(path, op.join(directory, 'restored'))


def test_restore_bad_version(cache, directory):
    cache['key'] = 'value'

    path = op.join(directory, 'backup')
    cache.backup(path)

    manifest = read_manifest(path)
    manifest['version'] = -1

    with open(op.join(path, 'manifest.json'), 'w') as writer:
        json.dump(manifest, writer)

    with pytest.raises(dc.BackupError):
        dc.Cache.restore(path, op.join(directory, 'restored'))


def test_restore_bad_format(cache, directory):
    cache['key'] = 'value'

    path = op.join(directory, 'backup')
    cache.backup(path)

    manifest = read_manifest(path)
    manifest['format'] = 'unknown-format'

    with open(op.join(path, 'manifest.json'), 'w') as writer:
        json.dump(manifest, writer)

    with pytest.raises(dc.BackupError):
        dc.Cache.restore(path, op.join(directory, 'restored'))


def test_restore_into_empty_directory(cache, directory):
    cache['key'] = 'value'

    path = op.join(directory, 'backup')
    cache.backup(path)

    target = op.join(directory, 'restored')
    os.mkdir(target)

    restored = dc.Cache.restore(path, target)

    try:
        assert restored['key'] == 'value'
    finally:
        restored.close()


def test_backup_cleanup_on_error(cache, directory):
    cache['large'] = make_value(0, 0)

    with mock.patch(
        'diskcache.core._copy_file', side_effect=OSError('boom')
    ), pytest.raises(OSError):
        cache.backup(op.join(directory, 'backup'))

    # Temporary directory is cleaned up after failure.

    assert os.listdir(directory) == []


def test_restore_cleanup_on_error(cache, directory):
    cache['key'] = 'value'

    path = op.join(directory, 'backup')
    cache.backup(path)

    with mock.patch(
        'diskcache.core.shutil.copyfile', side_effect=OSError('boom')
    ), pytest.raises(OSError):
        dc.Cache.restore(path, op.join(directory, 'restored'))

    # Temporary directory is cleaned up after failure.

    assert os.listdir(directory) == ['backup']


def test_restore_existing_cache_refused(cache, directory):
    cache['key'] = 'value'

    path = op.join(directory, 'backup')
    cache.backup(path)

    existing = dc.Cache(op.join(directory, 'existing'))

    try:
        existing['other'] = 'item'

        with pytest.raises(FileExistsError):
            dc.Cache.restore(path, existing.directory)

        # Existing cache is not overwritten.

        assert existing['other'] == 'item'
        assert 'key' not in existing
    finally:
        existing.close()


def test_backup_concurrent_writes(cache, directory):
    stop = threading.Event()
    key_count = 8

    def writer():
        generation = 0
        while not stop.is_set():
            for key in range(key_count):
                cache.set(key, make_value(key, generation), retry=True)
                generation += 1
                if key % 3 == 0:
                    cache.delete(('extra', key), retry=True)
                else:
                    cache.set(
                        ('extra', key), make_value(key, generation), retry=True
                    )

    threads = [
        threading.Thread(target=writer, daemon=True) for _ in range(4)
    ]

    for thread in threads:
        thread.start()

    try:
        for num in range(5):
            path = op.join(directory, 'backup-%d' % num)
            cache.backup(path)

            restored = dc.Cache.restore(
                path, op.join(directory, 'restored-%d' % num)
            )

            try:
                # Every index reads complete content after restore.

                assert not restored.check()

                for key in restored.iterkeys():
                    assert check_value(restored[key])
            finally:
                restored.close()
    finally:
        stop.set()

        for thread in threads:
            thread.join()


def replace_worker(directory, key_count, stop):
    """Continuously replace large values from another process."""
    cache = dc.Cache(directory, disk_min_file_size=1024)
    generation = 0

    while not stop.is_set():
        for key in range(key_count):
            cache.set(key, make_value(key, generation, 2**20), retry=True)
            generation += 1

    cache.close()


def test_backup_restore_multiprocess(directory):
    source = op.join(directory, 'source')
    cache = dc.Cache(source, disk_min_file_size=1024)
    key_count = 4

    for key in range(key_count):
        cache.set(key, make_value(key, 0, 2**20), retry=True)

    context = mp.get_context('spawn')
    stop = context.Event()
    process = context.Process(
        target=replace_worker, args=(source, key_count, stop)
    )
    process.start()

    try:
        # Backup while another process replaces large values.

        path = op.join(directory, 'backup')
        cache.backup(path)
    finally:
        stop.set()
        process.join()

    cache.close()

    restored = dc.Cache.restore(path, op.join(directory, 'restored'))

    try:
        assert not restored.check()
        assert len(restored) == key_count

        for key in range(key_count):
            assert check_value(restored[key])
    finally:
        restored.close()


def test_fanout_backup_restore(directory):
    cache = dc.FanoutCache(
        op.join(directory, 'fanout'), shards=4, disk_min_file_size=1024
    )

    try:
        for num in range(40):
            cache['key-%d' % num] = make_value(num, 0)

        path = op.join(directory, 'backup')
        cache.backup(path)

        manifest = read_manifest(path)
        assert manifest['format'] == 'diskcache-fanout-backup'
        assert manifest['version'] == 1
        assert manifest['shards'] == 4

        for num in range(4):
            shard_manifest = read_manifest(op.join(path, '%03d' % num))
            assert shard_manifest['format'] == 'diskcache-backup'
    finally:
        cache.close()

    restored = dc.FanoutCache.restore(path, op.join(directory, 'restored'))

    try:
        assert len(restored) == 40

        for num in range(40):
            assert check_value(restored['key-%d' % num])
    finally:
        restored.close()


def test_fanout_restore_validation(directory):
    cache = dc.FanoutCache(op.join(directory, 'fanout'), shards=4)
    cache['key'] = 'value'

    path = op.join(directory, 'backup')
    cache.backup(path)
    cache.close()

    # Existing cache is never overwritten.

    with pytest.raises(FileExistsError):
        dc.FanoutCache.restore(path, op.join(directory, 'fanout'))

    # Missing shard database is refused.

    os.remove(op.join(path, '000', 'cache.db'))

    with pytest.raises(dc.BackupError):
        dc.FanoutCache.restore(path, op.join(directory, 'restored'))

    assert not op.exists(op.join(directory, 'restored'))


def test_fanout_backup_path_exists(directory):
    cache = dc.FanoutCache(op.join(directory, 'fanout'), shards=4)

    try:
        with pytest.raises(FileExistsError):
            cache.backup(directory)
    finally:
        cache.close()


def test_fanout_backup_cleanup_on_error(directory):
    cache = dc.FanoutCache(op.join(directory, 'fanout'), shards=4)

    try:
        with mock.patch(
            'diskcache.fanout.Cache.backup', side_effect=OSError('boom')
        ), pytest.raises(OSError):
            cache.backup(op.join(directory, 'backup'))
    finally:
        cache.close()

    # Temporary directory is cleaned up after failure.

    assert os.listdir(directory) == ['fanout']


def test_fanout_restore_bad_manifest(directory):
    cache = dc.FanoutCache(op.join(directory, 'fanout'), shards=4)
    cache['key'] = 'value'

    path = op.join(directory, 'backup')
    cache.backup(path)
    cache.close()

    # Missing manifest is refused.

    with pytest.raises(dc.BackupError):
        dc.FanoutCache.restore(
            op.join(directory, 'missing'), op.join(directory, 'restored')
        )

    def restore_with(**updates):
        manifest = read_manifest(path)
        manifest.update(updates)

        with open(op.join(path, 'manifest.json'), 'w') as writer:
            json.dump(manifest, writer)

        with pytest.raises(dc.BackupError):
            dc.FanoutCache.restore(path, op.join(directory, 'restored'))

    restore_with(format='unknown-format')
    restore_with(format='diskcache-fanout-backup', version=-1)
    restore_with(version=1, shards=0)


def test_fanout_restore_into_empty_directory(directory):
    cache = dc.FanoutCache(op.join(directory, 'fanout'), shards=4)
    cache['key'] = 'value'

    path = op.join(directory, 'backup')
    cache.backup(path)
    cache.close()

    target = op.join(directory, 'restored')
    os.mkdir(target)

    restored = dc.FanoutCache.restore(path, target)

    try:
        assert restored['key'] == 'value'
    finally:
        restored.close()
