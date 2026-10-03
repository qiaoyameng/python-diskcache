"""Routing metadata for sharded fanout caches."""

import json
import os
import os.path as op
import tempfile

from .core import DBNAME, GenerationChanged, MaintenanceMode

ROUTE_FILENAME = 'fanout.json'
MIGRATION_FILENAME = 'fanout-migration.json'
MAINTENANCE_LOCK_FILENAME = '.fanout-maintenance.db'

ROUTE_VERSION = 1


def default_route(shards, size_limit=None):
    """Return a route for a new fanout cache."""
    return {
        'version': ROUTE_VERSION,
        'layout_version': 1,
        'generation': 1,
        'shards': int(shards),
        'maintenance': False,
        'size_limit': size_limit,
    }


def _atomic_write(path, document):
    directory = op.dirname(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix='.%s.' % op.basename(path),
        suffix='.tmp',
        dir=directory,
    )
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as writer:
            json.dump(document, writer, indent=2, sort_keys=True)
            writer.write('\n')
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, path)
    except BaseException:
        with _suppress_oserror():
            os.remove(temporary)
        raise


class _suppress_oserror:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return issubclass(exc_type, OSError) if exc_type else False


def write_route(directory, route):
    """Write route metadata atomically."""
    os.makedirs(directory, exist_ok=True)
    _atomic_write(op.join(directory, ROUTE_FILENAME), route)


def write_migration_state(directory, state):
    """Write migration state atomically."""
    os.makedirs(directory, exist_ok=True)
    _atomic_write(op.join(directory, MIGRATION_FILENAME), state)


def read_json(path):
    """Read and return a JSON document."""
    with open(path, 'r', encoding='utf-8') as reader:
        return json.load(reader)


def _validate_route(route):
    if not isinstance(route, dict):
        raise ValueError('fanout route must be a JSON object')
    if route.get('version') != ROUTE_VERSION:
        raise ValueError('unsupported fanout route version')
    for name in ('layout_version', 'generation', 'shards'):
        value = route.get(name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError('invalid fanout route %s' % name)
    if not isinstance(route.get('maintenance'), bool):
        raise ValueError('invalid fanout route maintenance flag')
    size_limit = route.get('size_limit')
    if size_limit is not None and (
        not isinstance(size_limit, (int, float)) or size_limit < 0
    ):
        raise ValueError('invalid fanout route size_limit')
    return route


def load_route(directory):
    """Return the route, or None when no route exists."""
    path = op.join(directory, ROUTE_FILENAME)
    if not op.exists(path):
        return None
    return _validate_route(read_json(path))


def load_migration_state(directory):
    """Return migration state, or None when no state exists."""
    path = op.join(directory, MIGRATION_FILENAME)
    if not op.exists(path):
        return None
    state = read_json(path)
    if not isinstance(state, dict):
        raise ValueError('migration state must be a JSON object')
    return state


def remove_migration_state(directory):
    """Remove migration state after cancellation."""
    with _suppress_oserror():
        os.remove(op.join(directory, MIGRATION_FILENAME))


def existing_legacy_shards(directory):
    """Count first-generation shard directories in an existing cache."""
    count = 0
    if not op.isdir(directory):
        return 0
    for name in os.listdir(directory):
        if len(name) != 3 or not name.isdigit():
            continue
        if op.exists(op.join(directory, name, DBNAME)):
            count = max(count, int(name) + 1)
    return count


def open_route(directory, shards, size_limit=None):
    """Load or initialize the route for a FanoutCache instance."""
    if shards <= 0:
        raise ValueError('shards must be a positive integer')

    route = load_route(directory)
    if route is not None:
        if route['shards'] != shards:
            raise ValueError(
                'cache uses %d shards; migrate before opening with %d shards'
                % (route['shards'], shards)
            )
        return route, True

    legacy_shards = existing_legacy_shards(directory)
    if legacy_shards:
        if legacy_shards != shards:
            raise ValueError(
                'cache uses %d shards; migrate before opening with %d shards'
                % (legacy_shards, shards)
            )
        return default_route(shards, size_limit), False

    os.makedirs(directory, exist_ok=True)
    route = default_route(shards, size_limit)
    write_route(directory, route)
    return route, True


def shard_path(directory, layout, index):
    """Return the directory for a shard described by `layout`."""
    version = layout['layout_version']
    if version == 1:
        return op.join(directory, '%03d' % index)
    if version == 2:
        return op.join(
            directory,
            'shards',
            '%03d' % layout['generation'],
            '%03d' % index,
        )
    raise ValueError('unsupported fanout layout version: %d' % version)


def layout_matches(left, right):
    """Return whether two route layouts identify the same generation."""
    return (
        left['layout_version'] == right['layout_version']
        and left['generation'] == right['generation']
        and left['shards'] == right['shards']
    )


def layout_from_route(route):
    """Return the layout portion of a route."""
    return {
        'layout_version': route['layout_version'],
        'generation': route['generation'],
        'shards': route['shards'],
    }


def route_signature(directory, route_exists):
    """Return a cached file-system signature for the active route."""
    path = op.join(directory, ROUTE_FILENAME)
    if not route_exists or not op.exists(path):
        return None
    stat = os.stat(path)
    return (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size)


def check_generation(directory, expected, route_exists, signature):
    """Read the route when it changes and validate the open generation."""
    path = op.join(directory, ROUTE_FILENAME)
    if not op.exists(path):
        if route_exists or expected['layout_version'] != 1:
            raise GenerationChanged('fanout cache generation changed')
        route = default_route(expected['shards'])
        route['maintenance'] = False
        return route, None

    route = load_route(directory)
    if route['maintenance']:
        raise MaintenanceMode('fanout cache is in maintenance mode')
    if (
        route['layout_version'] != expected['layout_version']
        or route['generation'] != expected['generation']
    ):
        raise GenerationChanged('fanout cache generation changed')
    return route, signature
