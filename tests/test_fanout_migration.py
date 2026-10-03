"""Tests for resharding fanout caches."""

import json
import os
import os.path as op
import shutil
import threading
import time
from unittest import mock

import pytest

import diskcache as dc
from diskcache.cli import main as cli_main
from diskcache.migration import cancel_migration, migrate_shards
from diskcache.routes import (
    MIGRATION_FILENAME,
    ROUTE_FILENAME,
    load_migration_state,
    load_route,
    shard_path,
)


@pytest.fixture
def directory():
    path = dc.FanoutCache().directory
    shutil.rmtree(path)
    yield path
    for root, dirs, files in os.walk(path):
        os.chmod(root, 0o700)
        for name in dirs:
            os.chmod(op.join(root, name), 0o700)
        for name in files:
            os.chmod(op.join(root, name), 0o600)
    if op.isdir(path):
        os.chmod(path, 0o700)
        shutil.rmtree(path)


def read_json(root, name):
    with open(op.join(root, name), encoding='utf-8') as reader:
        return json.load(reader)


def make_cache(root, shards=2):
    cache = dc.FanoutCache(
        root,
        shards=shards,
        disk_min_file_size=4096,
        size_limit=2**30,
    )
    cache.set('plain', 'value', expire=3600, tag='plain-tag')
    cache.set(123, 456, expire=None, tag='integer-tag')
    cache.set('large', b'x' * 4096, expire=7200, tag='large-tag')
    cache.set(('pickled', 'key'), {'answer': 42}, tag='object-tag')
    return cache


def test_migrate_preserves_data_and_absolute_expiry(directory):
    with make_cache(directory) as old:
        plain_expire = old.get('plain', expire_time=True)[1]
        large_expire = old.get('large', expire_time=True)[1]

    state = migrate_shards(directory, 4, batch_size=2)

    assert state['status'] == 'complete'
    assert state['source_action'] == 'sealed'
    route = load_route(directory)
    assert route['shards'] == 4
    assert route['generation'] == 2
    assert route['layout_version'] == 2

    with dc.FanoutCache(
        directory, shards=4, disk_min_file_size=4096
    ) as cache:
        assert cache.get('plain', expire_time=True, tag=True) == (
            'value',
            plain_expire,
            'plain-tag',
        )
        assert cache.get('large', expire_time=True, tag=True)[1] == (
            large_expire
        )
        assert cache.get('large', tag=True) == (
            b'x' * 4096,
            'large-tag',
        )
        assert cache[123] == 456
        assert cache.get(('pickled', 'key'), tag=True) == (
            {'answer': 42},
            'object-tag',
        )
        assert len(cache) == 4
        assert cache.check() == []

    assert op.exists(shard_path(directory, state['source'], 0))
    assert not os.access(
        op.join(shard_path(directory, state['source'], 0), 'cache.db'),
        os.W_OK,
    )


def test_opening_wrong_shard_count_requires_migration(directory):
    with make_cache(directory):
        pass
    with pytest.raises(ValueError):
        dc.FanoutCache(directory, shards=4)


def test_old_generation_is_fenced_after_switch(directory):
    with make_cache(directory) as cache:
        migrate_shards(directory, 4, batch_size=2)
        with pytest.raises(dc.GenerationChanged):
            cache.set('late', 'write')
        with pytest.raises(dc.GenerationChanged):
            del cache['plain']


def test_waiting_old_writer_is_fenced_after_switch(directory):
    with make_cache(directory) as cache:
        errors = []
        started = threading.Event()

        def waiting_write():
            started.set()
            try:
                cache.set('waiting', 'write', retry=True)
            except BaseException as error:
                errors.append(error)

        def progress(_state):
            if not started.is_set():
                thread = threading.Thread(target=waiting_write)
                thread.start()
                started.wait()
                time.sleep(0.1)
                _state['_delay'] = True
            if _state.get('_delay') and len(errors) == 0:
                time.sleep(0.1)

        migrate_shards(directory, 4, batch_size=2, progress_callback=progress)
        time.sleep(0.1)
        assert len(errors) == 1
        assert isinstance(
            errors[0],
            (dc.MaintenanceMode, dc.GenerationChanged),
        )


def test_maintenance_route_blocks_old_connection(directory):
    with make_cache(directory) as cache:
        route = load_route(directory)
        route['maintenance'] = True
        with open(op.join(directory, ROUTE_FILENAME), 'w') as writer:
            json.dump(route, writer)
        with pytest.raises(dc.MaintenanceMode):
            cache.set('blocked', 'write')


def test_migration_records_manifest_sources_targets_and_progress(directory):
    with make_cache(directory):
        pass
    state = migrate_shards(directory, 4, batch_size=2)

    assert state['source']['shards'] == 2
    assert state['source']['paths'] == ['000', '001']
    assert state['target']['shards'] == 4
    assert state['target']['paths'][0] == op.join('shards', '002', '000')
    assert [item['copied_rows'] for item in state['progress']]
    assert state['validation']['status'] == 'complete'
    assert state['validation']['source_rows'] == 4
    assert state['validation']['target_rows'] == 4
    assert load_migration_state(directory)['status'] == 'complete'


def test_large_values_are_copied_into_target_directory(directory):
    with make_cache(directory):
        pass
    migrate_shards(directory, 4, batch_size=1)

    for shard in range(4):
        root = shard_path(
            directory,
            {'layout_version': 2, 'generation': 2, 'shards': 4},
            shard,
        )
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith('.val'):
                    full_path = op.join(dirpath, filename)
                    with open(full_path, 'rb') as reader:
                        assert reader.read() == b'x' * 4096
                    assert directory in full_path
                    assert op.isfile(full_path)


def test_migration_resumes_after_injected_interruption(directory):
    with make_cache(directory):
        pass
    calls = {'count': 0}
    real_copy = dc.migration._copy_value_file

    def interrupt_once(*args, **kwargs):
        calls['count'] += 1
        if calls['count'] == 1:
            raise KeyboardInterrupt
        return real_copy(*args, **kwargs)

    with mock.patch(
        'diskcache.migration._copy_value_file', side_effect=interrupt_once
    ):
        with pytest.raises(KeyboardInterrupt):
            migrate_shards(directory, 4, batch_size=1)

    interrupted = read_json(directory, MIGRATION_FILENAME)
    assert interrupted['status'] == 'failed'
    assert interrupted['error']['type'] == 'KeyboardInterrupt'
    assert load_route(directory)['maintenance'] is True

    state = migrate_shards(directory, 4, batch_size=1)
    assert state['status'] == 'complete'
    assert state['error'] is None
    with dc.FanoutCache(directory, shards=4) as cache:
        assert len(cache) == 4
        assert cache.check() == []


def test_cancel_failed_migration_restores_source_writes(directory):
    with make_cache(directory) as cache:
        with mock.patch(
            'diskcache.migration._check_space',
            side_effect=RuntimeError('stop'),
        ):
            with pytest.raises(RuntimeError):
                migrate_shards(directory, 4)
        cancel_migration(directory)
        cache.set('after-cancel', 'ok')
        assert cache['after-cancel'] == 'ok'

    with dc.FanoutCache(directory, shards=2) as reopened:
        assert reopened['after-cancel'] == 'ok'
    assert not op.exists(op.join(directory, MIGRATION_FILENAME))


def test_missing_value_file_is_reported_and_source_remains_readable(directory):
    with make_cache(directory):
        pass
    filename = dc.Cache(op.join(directory, '000')).get(
        'large', read=True, retry=True
    ).name
    os.remove(filename)

    with pytest.raises(dc.MigrationValueFileError):
        migrate_shards(directory, 4, batch_size=1)

    state = load_migration_state(directory)
    assert state['status'] == 'failed'
    assert state['error']['type'] == 'MigrationValueFileError'
    with dc.FanoutCache(directory, shards=2) as cache:
        assert cache.get('large') is None
        assert cache['plain'] == 'value'


def test_disk_space_preflight_reports_required_and_available(directory):
    with make_cache(directory):
        pass
    with mock.patch('diskcache.migration.shutil.disk_usage') as usage:
        usage.return_value.free = 0
        with pytest.raises(dc.MigrationDiskSpaceError, match='0 bytes'):
            migrate_shards(directory, 4)

    state = load_migration_state(directory)
    assert state['status'] == 'failed'
    assert state['error']['type'] == 'MigrationDiskSpaceError'
    assert state['error']['required_bytes'] > 0
    assert state['error']['available_bytes'] == 0
    assert op.exists(shard_path(directory, load_route(directory), 0))


def test_delete_source_removes_old_layout_after_validation(directory):
    with make_cache(directory):
        pass
    state = migrate_shards(directory, 4, delete_source=True)
    assert state['source_action'] == 'deleted'
    assert not op.exists(shard_path(directory, state['source'], 0))
    assert op.exists(
        shard_path(
            directory,
            {'layout_version': 2, 'generation': 2, 'shards': 4},
            0,
        )
    )


def test_command_line_migrates_fanout_cache(directory):
    with make_cache(directory):
        pass
    assert cli_main([
        'migrate-fanout',
        directory,
        '4',
        '--batch-size',
        '2',
    ]) == 0
    with dc.FanoutCache(directory, shards=4) as cache:
        assert len(cache) == 4


def test_migration_preserves_total_size_limit(directory):
    with dc.FanoutCache(
        directory,
        shards=4,
        size_limit=4000,
    ) as cache:
        cache.set('only', 'value')

    migrate_shards(directory, 8, batch_size=1)

    with dc.FanoutCache(directory, shards=8) as cache:
        assert cache.size_limit == 500
        assert sum(shard.size_limit for shard in cache._shards) == 4000
