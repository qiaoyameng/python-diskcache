"""Command line interface for diskcache maintenance tools.

Usage::

    python -m diskcache.cli migrate DIRECTORY --shards 8
    python -m diskcache.cli info DIRECTORY

"""

import argparse
import json
import sys

from .core import Disk
from .migration import (
    abort_migrate,
    MigrationError,
    layout_info,
    migrate as run_migration,
)


def _cmd_abort(args):
    try:
        state = abort_migrate(
            args.directory, lock_timeout=args.lock_timeout
        )
    except MigrationError as error:
        print('abort failed: %s' % error, file=sys.stderr)
        return 1

    print('aborted migration in %s (was phase=%s)' % (
        args.directory, state.get('phase'),
    ))
    return 0


def _cmd_migrate(args):
    try:
        result = run_migration(
            args.directory,
            shards=args.shards,
            disk=Disk,
            timeout=args.timeout,
            keep_source=args.keep_source,
            lock_timeout=args.lock_timeout,
            free_space_margin=args.free_space_margin,
            resume=not args.no_resume,
        )
    except MigrationError as error:
        print('migration failed: %s' % error, file=sys.stderr)
        return 1

    payload = result.as_dict()

    if args.quiet:
        print(
            '%s: %d -> %d shards (generation %s, %d rows)'
            % (
                args.directory,
                result.source_shards,
                result.target_shards,
                result.generation,
                result.rows,
            )
        )
    else:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write('\n')

    return 0


def _cmd_info(args):
    json.dump(layout_info(args.directory), sys.stdout, indent=2,
              sort_keys=True, default=str)
    sys.stdout.write('\n')
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog='diskcache')
    subparsers = parser.add_subparsers(dest='command', required=True)

    migrate = subparsers.add_parser(
        'migrate', help='redistribute a fanout cache over N shards'
    )
    migrate.add_argument('directory', help='fanout cache directory')
    migrate.add_argument(
        '--shards', type=int, required=True, help='target shard count'
    )
    migrate.add_argument('--timeout', type=float, default=60)
    migrate.add_argument('--lock-timeout', type=float, default=60)
    migrate.add_argument(
        '--free-space-margin', type=float, default=1.10,
        help='required free-space multiplier (default 1.10)',
    )
    migrate.add_argument(
        '--keep-source', action='store_true',
        help='keep old shard directories after cutover',
    )
    migrate.add_argument(
        '--no-resume', action='store_true',
        help='fail instead of resuming an interrupted migration',
    )
    migrate.add_argument('--quiet', action='store_true')
    migrate.set_defaults(func=_cmd_migrate)

    info = subparsers.add_parser(
        'info', help='show fanout layout and migration state'
    )
    info.add_argument('directory', help='fanout cache directory')
    info.set_defaults(func=_cmd_info)

    abort = subparsers.add_parser(
        'abort', help='cancel an unfinished fanout migration'
    )
    abort.add_argument('directory', help='fanout cache directory')
    abort.add_argument('--lock-timeout', type=float, default=60)
    abort.set_defaults(func=_cmd_abort)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
