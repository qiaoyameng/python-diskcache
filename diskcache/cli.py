"""Command line interface to disk cache."""

import argparse
import sys

from .migration import migrate_shards


def build_parser():
    """Build the command line argument parser."""
    parser = argparse.ArgumentParser(prog='diskcache')
    commands = parser.add_subparsers(dest='command', required=True)

    migrate = commands.add_parser(
        'migrate-fanout',
        help='redistribute a fanout cache into a new shard layout',
    )
    migrate.add_argument('directory', help='fanout cache directory')
    migrate.add_argument(
        'shards', type=int, help='target number of shards'
    )
    migrate.add_argument(
        '--batch-size', type=int, default=100, help='rows per transaction'
    )
    migrate.add_argument(
        '--timeout', type=float, default=60, help='lock timeout in seconds'
    )
    migrate.add_argument(
        '--delete-source',
        action='store_true',
        help='delete source shards after successful switch',
    )
    return parser


def main(argv=None):
    """Run the diskcache command line interface."""
    args = build_parser().parse_args(argv)
    if args.command == 'migrate-fanout':
        state = migrate_shards(
            args.directory,
            args.shards,
            timeout=args.timeout,
            batch_size=args.batch_size,
            delete_source=args.delete_source,
        )
        print(
            'migrated %d -> %d shards: %s'
            % (
                state['source']['shards'],
                state['target']['shards'],
                state['status'],
            )
        )
        return 0
    return 2


if __name__ == '__main__':
    sys.exit(main())
