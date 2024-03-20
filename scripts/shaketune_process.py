#!/usr/bin/env python3

import argparse
import sys
from importlib import import_module
from pathlib import Path

DPI = 200


class NumberRange:

    def __init__(self, start: float, end: float) -> None:
        self._start = start
        self._end = end

    def __eq__(self, other) -> bool:
        return self._start <= other <= self._end


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='action', required=True)

    map_axes_parser = subparsers.add_parser('map_axes')
    map_axes_parser.add_argument('csv_path', type=Path)

    belts_parser = subparsers.add_parser('belts')
    belts_parser.add_argument('-f', '--max-freq', type=float, default=200.0)
    belts_parser.add_argument('-o',
                              '--output',
                              type=Path,
                              required=True,
                              help='Output graph image path')
    belts_parser.add_argument('csv_paths', type=Path, nargs='+')

    resonances_parser = subparsers.add_parser('resonances')
    resonances_parser.add_argument('-f',
                                   '--max-freq',
                                   type=float,
                                   default=200.0)
    resonances_parser.add_argument('-s',
                                   '--max-smoothing',
                                   type=float,
                                   default=None,
                                   choices=[NumberRange(0.05, 1.00)])
    resonances_parser.add_argument('--scv',
                                   '--square-corner-velocity',
                                   type=float,
                                   default=5.0)
    resonances_parser.add_argument('-o',
                                   '--output',
                                   type=Path,
                                   required=True,
                                   help='Output graph image path')
    resonances_parser.add_argument('csv_path', type=Path)

    vibrations_parser = subparsers.add_parser('vibrations')
    vibrations_parser.add_argument('-a', '--axis', default=None)
    vibrations_parser.add_argument('-c', '--accel', type=int, default=None)
    vibrations_parser.add_argument('-f',
                                   '--max-freq',
                                   type=float,
                                   default=1000.0)
    vibrations_parser.add_argument(
        '-r',
        '--remove',
        type=int,
        default=None,
        choices=[NumberRange(0, 50)],
        help="Percentage of data removed at start/end of each CSV files")
    vibrations_parser.add_argument('-o',
                                   '--output',
                                   type=Path,
                                   required=True,
                                   help='Output graph image path')
    vibrations_parser.add_argument('csv_paths', type=Path, nargs='+')

    args = parser.parse_args()

    sys.path.append(str((Path(__file__).parent / '..' / 'klippy').absolute()))

    match args.action:
        case 'map_axes':
            map_axes = import_module('.map_axes', 'extras.shaketune')
            results = map_axes.map_axes(args.csv_path)
            print(results)

        case 'belts':
            belts = import_module('.belts', 'extras.shaketune')
            fig = belts.graph_belts_test_results(args.csv_paths, args.max_freq)
            fig.savefig(args.output, dpi=DPI)

        case 'resonances':
            resonances = import_module('.resonances', 'extras.shaketune')
            fig = resonances.graph_resonances_test_results(
                args.csv_path, args.max_smoothing, args.square_corner_velocity,
                args.max_freq)
            fig.savefig(args.output, dpi=DPI)

        case 'vibrations':
            vibrations = import_module('.vibrations', 'extras.shaketune')
            fig = vibrations.graph_vibrations_test_results(
                args.csv_paths, args.axis, args.accel, args.max_freq,
                args.remove)
            fig.savefig(args.output, dpi=DPI)


if __name__ == '__main__':
    main()
