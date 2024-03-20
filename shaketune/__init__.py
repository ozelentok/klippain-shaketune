import shutil
import tarfile
import traceback
from datetime import datetime
from pathlib import Path

from . import belts, map_axes, resonances, vibrations

RESULTS_PATH = Path.home() / 'klipper-shaketune-results'
RAW_DATA_PATH = Path('/tmp')
DPI = 150


def process_axes_map(chip: str) -> tuple[str, Path]:
    t = current_timestamp()
    csv_path = save_latest_csvs('axes_map', t, f'{chip}-*.csv', 1)[0]

    results = map_axes.map_axes(csv_path)
    results_path = csv_path.parent / f'axes_map_{t}.txt'

    with open(results_path, 'w') as f:
        f.write(results)

    return results, results_path


def process_belts(keep_csv: bool) -> Path:
    t = current_timestamp()
    csv_paths = save_latest_csvs('belts', t, f'raw_data_*.csv', 2)

    fig = belts.graph_belts_test_results(csv_paths)
    png_path = csv_paths[0].parent / f'belts_{t}.png'
    fig.savefig(png_path, dpi=DPI)

    if keep_csv:
        for p in csv_paths:
            p.unlink(True)

    return png_path


def process_resonances(axis: str, scv: float, max_smoothing: float | None,
                       keep_csv: bool) -> Path:
    t = current_timestamp()
    csv_path = save_latest_csvs('resonances', t, f'raw_data_*_{axis}*.csv',
                                1)[0]

    fig = resonances.graph_resonances_test_results(csv_path,
                                                   max_smoothing=max_smoothing,
                                                   scv=scv)
    png_path = csv_path.parent / f'resonances_{t}_{axis}.png'
    fig.savefig(png_path, dpi=DPI)

    if keep_csv:
        csv_path.unlink(True)

    return png_path


def process_vibrations(axis: str, accel: float, chip_name: str,
                       keep_csv: bool) -> Path:
    t = current_timestamp()
    csv_paths = save_latest_csvs('vibrations', t, f'{chip_name}-*.csv')

    fig = vibrations.graph_vibrations_test_results(csv_paths, axis, accel)
    png_path = csv_paths[0].parent / f'vibrations_{t}_{axis}.png'
    fig.savefig(png_path, dpi=DPI)

    if keep_csv:
        with tarfile.open(csv_paths[0].parent / f'vibrations_{t}_{axis}.tar.gz',
                          'w:gz') as tar:
            for p in csv_paths:
                tar.add(p, recursive=False)

    for p in csv_paths:
        p.unlink(True)

    return png_path


def save_latest_csvs(calibration_name: str,
                     timestamp: str,
                     input_glob: str,
                     count: int | None = None) -> list[Path]:
    csv_paths = find_latest_csvs(input_glob, count)
    moved_csv_paths = []

    output_dir_path = RESULTS_PATH / calibration_name
    output_dir_path.mkdir(parents=True, exist_ok=True)

    for p in csv_paths:
        delim = '_' if p.stem.startswith('raw_data') else '-'
        sub_part = p.stem.split(delim)[-1].upper()
        new_name = f'{calibration_name}_{timestamp}_{sub_part}.csv'

        output_path = output_dir_path / new_name
        shutil.move(p, output_path)

        moved_csv_paths.append(output_path)

    return moved_csv_paths


def find_latest_csvs(glob_pattern: str, count: int | None = None) -> list[Path]:
    paths = RAW_DATA_PATH.glob(glob_pattern)
    if not paths:
        raise ValueError(f'No CSV files {glob_pattern} found')

    paths = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
    if not count:
        return paths
    if len(paths) >= count:
        return paths[:count]

    raise ValueError(f'Found {len(paths)} files, {count} required')


def current_timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def find_oldest_saved_files(dir_path: Path, extension: str,
                            limit: int) -> list[Path]:
    if not dir_path.exists():
        return []

    paths = [p for p in dir_path.iterdir() if p.suffix == extension]
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[limit:]


def remove_old_files(keep_n_results: int):
    keep1 = keep_n_results + 1
    keep2 = 2 * keep_n_results + 1

    old_belts_files = find_oldest_saved_files(RESULTS_PATH / 'belts', '.png',
                                              keep1)
    old_resonances_files = find_oldest_saved_files(RESULTS_PATH / 'resonances',
                                                   '.png', keep2)
    old_vibrations_files = find_oldest_saved_files(RESULTS_PATH / 'vibrations',
                                                   '.png', keep1)

    for old_file in old_belts_files:
        file_date = '_'.join(old_file.stem.split('_')[1:3])
        for suffix in ['A', 'B']:
            csv_file = RESULTS_PATH / 'belts' / f'belts_{file_date}_{suffix}.csv'
            csv_file.unlink(True)
        old_file.unlink(True)

    for old_file in old_resonances_files:
        csv_file = RESULTS_PATH / 'resonances' / old_file.with_suffix('.csv')
        csv_file.unlink(True)
        old_file.unlink(True)

    for old_file in old_vibrations_files:
        tar_file = RESULTS_PATH / 'vibrations' / old_file.with_suffix('.tar.gz')
        tar_file.unlink(True)
        old_file.unlink(True)


class ShakeTune:

    def __init__(self, config) -> None:
        self._printer = config.get_printer()
        self._pconfig = self._printer.lookup_object('configfile')
        self._gcode = self._printer.lookup_object('gcode')

        gcode = self._printer.lookup_object('gcode')
        gcode.register_command('SHAKETUNE_PROCESS_MAP_AXES',
                               self.cmd_SHAKETUNE_PROCESS_MAP_AXES,
                               desc='Process ShakeTune axes mapping data')
        gcode.register_command('SHAKETUNE_PROCESS_BELTS',
                               self.cmd_SHAKETUNE_PROCESS_BELTS,
                               desc='Process ShakeTune belts test data')
        gcode.register_command('SHAKETUNE_PROCESS_RESONANCES',
                               self.cmd_SHAKETUNE_PROCESS_RESONANCES,
                               desc='Process ShakeTune resonances test data')
        gcode.register_command('SHAKETUNE_PROCESS_VIBRATIONS',
                               self.cmd_SHAKETUNE_PROCESS_VIBRATIONS,
                               desc='Process ShakeTune belts test data')
        gcode.register_command('SHAKETUNE_REMOVE_OLD_RESULTS',
                               self.cmd_SHAKETUNE_REMOVE_OLD_RESULTS,
                               desc='Remove ShakeTune old results')

    def cmd_SHAKETUNE_PROCESS_MAP_AXES(self, gcmd) -> None:
        chip = gcmd.get('CHIP', 'adxl345')
        try:
            results, results_path = process_axes_map(chip)
            self._gcode.respond_info(f'Map axes: {results}')
            self._gcode.respond_info(f'Map axes saved to {results_path}')
        except:
            self._gcode.respond_info(
                f'!! Failed processing axes map\n{traceback.format_exc()}')

    def cmd_SHAKETUNE_PROCESS_BELTS(self, gcmd) -> None:
        keep_csv = bool(gcmd.get_int('KEEP_CSV', 1))
        try:
            graph_path = process_belts(keep_csv)
            self._gcode.respond_info(f'Belts graph generated: {graph_path}')
        except:
            self._gcode.respond_info(
                f'!! Failed processing belts data\n{traceback.format_exc()}')

    def cmd_SHAKETUNE_PROCESS_RESONANCES(self, gcmd) -> None:
        axis = gcmd.get('AXIS')
        scv = gcmd.get_float('SCV')
        max_smoothing = gcmd.get_float('MAX_SMOOTHING', None)
        keep_csv = bool(gcmd.get_int('KEEP_CSV', 1))
        try:
            graph_path = process_resonances(axis, scv, max_smoothing, keep_csv)
            self._gcode.respond_info(
                f'Resonances graph generated: {graph_path}')
        except:
            self._gcode.respond_info(
                f'!! Failed processing resonances data\n{traceback.format_exc()}'
            )

    def cmd_SHAKETUNE_PROCESS_VIBRATIONS(self, gcmd) -> None:
        axis = gcmd.get('AXIS')
        accel = gcmd.get_float('ACCEL')
        chip = gcmd.get('CHIP', 'adxl345')
        keep_csv = bool(gcmd.get_int('KEEP_CSV', 1))
        try:
            graph_path = process_vibrations(axis, accel, chip, keep_csv)
            self._gcode.respond_info(
                f'Vibrations graph generated: {graph_path}')
        except:
            self._gcode.respond_info(
                f'!! Failed processing vibrations data\n{traceback.format_exc()}'
            )

    def cmd_SHAKETUNE_REMOVE_OLD_RESULTS(self, gcmd) -> None:
        keep_n_results = gcmd.get_int('KEEP_N_RESULTS', 3)
        try:
            remove_old_files(keep_n_results)
            self._gcode.respond_info('ShakeTune old files removed')
        except:
            self._gcode.respond_info(
                f'!! Failed removing old results\n{traceback.format_exc()}')


def load_config(config) -> ShakeTune:
    return ShakeTune(config)
