"""
Benchmark ASL and SARA runtime for near-real-time monitoring discussion.

This script reports:
1. Total runtime (load + preprocess + localisation)
2. Localisation-only runtime (preloaded data)
3. Runtime relative to the analysed data duration
4. Scaling with number of stations for both ASL and SARA

Results are saved to CSV files in MakeFigures/RESULTS.
"""

from __future__ import annotations

import csv
import io
from contextlib import redirect_stdout
from datetime import datetime
from math import comb
from pathlib import Path
import statistics
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

import run_ASL as asl_cfg_mod
import run_SARA as sara_cfg_mod
from AnalysisCodes.asl import (
    load_event_data as asl_load_event_data,
    load_topography as asl_load_topography,
    run_localization as asl_run_localization,
)
from AnalysisCodes.sara import (
    amplitude_ratio_localization,
    load_event_data as sara_load_event_data,
    load_topography as sara_load_topography,
)


SUMMARY_CSV = HERE / "RESULTS" / "benchmark_runtime_summary.csv"
SCALING_CSV = HERE / "RESULTS" / "benchmark_station_scaling.csv"

TOTAL_REPEATS = 50
LOCALISATION_REPEATS = 50
STATION_COUNTS = None  # Use None to benchmark all counts from 3 to N.
VERBOSE_PROGRESS = True
SHOW_INTERNAL_FUNCTION_OUTPUT = False
STORE_FULL_FIELDS = False

# Optional benchmark-specific station ordering. Set to None to use the
# station order from run_ASL.py / run_SARA.py.
ASL_BENCHMARK_STATIONS = ["OTVZ","SNVZ","NGZ","ETVZ","NOVZ","TMVZ","WTVZ","NTVZ","KRVZ","COVZ","FWVZ","MAVZ","WHVZ","TRVZ"]
SARA_BENCHMARK_STATIONS = ["OTVZ","SNVZ","NGZ","ETVZ","NOVZ","TMVZ","WTVZ","NTVZ","KRVZ","COVZ","FWVZ","MAVZ","WHVZ","TRVZ"]

# Optional benchmark-specific overrides. Set to None to use the runner defaults.
ASL_SPATIAL_STEP = 100
ASL_TEMPORAL_STEP = 0.5
SARA_SPATIAL_STEP = 100
SARA_TEMPORAL_STEP = 0.5


def _timed_call(func, *args, show_output=False, **kwargs):
    buffer = io.StringIO()
    start = time.perf_counter()
    with redirect_stdout(buffer):
        result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    if show_output:
        text = buffer.getvalue().strip()
        if text:
            print(text)
    return elapsed, result


def _timed_total_run(loader_topo, loader_data, localizer, load_kwargs, localisation_kwargs):
    def _run():
        topo = loader_topo(load_kwargs["TOPOGRAPHY_DIR"])
        data = loader_data(**load_kwargs["event"])
        result = localizer(data=data, topo=topo, **localisation_kwargs)
        return topo, data, result

    return _timed_call(_run, show_output=SHOW_INTERNAL_FUNCTION_OUTPUT)


def _timed_localisation_runs(loader_topo, loader_data, localizer, load_kwargs, localisation_kwargs, repeats):
    _, topo = _timed_call(loader_topo, load_kwargs["TOPOGRAPHY_DIR"], show_output=SHOW_INTERNAL_FUNCTION_OUTPUT)
    _, data = _timed_call(loader_data, **load_kwargs["event"], show_output=SHOW_INTERNAL_FUNCTION_OUTPUT)
    times = []
    result = None
    for _ in range(repeats):
        elapsed, result = _timed_call(
            localizer,
            data=data,
            topo=topo,
            show_output=SHOW_INTERNAL_FUNCTION_OUTPUT,
            **localisation_kwargs,
        )
        times.append(elapsed)
    return topo, data, result, times


def _summary_stats(times):
    times = list(times)
    return {
        "mean_s": float(statistics.mean(times)),
        "min_s": float(min(times)),
        "max_s": float(max(times)),
        "std_s": float(statistics.stdev(times)) if len(times) > 1 else 0.0,
    }


def _data_duration_seconds(start_time, end_time):
    start = datetime.fromisoformat(str(start_time))
    end = datetime.fromisoformat(str(end_time))
    return (end - start).total_seconds()


def _safe_ratio(numerator, denominator):
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)


def _base_station_counts(stations):
    if STATION_COUNTS is not None:
        return STATION_COUNTS
    return list(range(3, len(stations) + 1))


def _series_ratios(numerator, denominators):
    return [_safe_ratio(numerator, value) for value in denominators]


def _series_ratios_with_fixed_denominator(numerators, denominator):
    return [_safe_ratio(value, denominator) for value in numerators]


def _get_method_defaults(method_name):
    if method_name == "ASL":
        stations = list(asl_cfg_mod.STATIONS) if ASL_BENCHMARK_STATIONS is None else list(ASL_BENCHMARK_STATIONS)
        return {
            "stations": stations,
            "start_time": asl_cfg_mod.START_TIME,
            "end_time": asl_cfg_mod.END_TIME,
        }
    if method_name == "SARA":
        stations = list(sara_cfg_mod.STATIONS) if SARA_BENCHMARK_STATIONS is None else list(SARA_BENCHMARK_STATIONS)
        return {
            "stations": stations,
            "start_time": sara_cfg_mod.START_TIME,
            "end_time": sara_cfg_mod.END_TIME,
        }
    raise ValueError(f"Unknown method '{method_name}'.")


def _build_asl_spec(stations):
    spatial_step = asl_cfg_mod.SPATIAL_STEP if ASL_SPATIAL_STEP is None else ASL_SPATIAL_STEP
    temporal_step = asl_cfg_mod.TEMPORAL_STEP if ASL_TEMPORAL_STEP is None else ASL_TEMPORAL_STEP
    cfg = dict(
        TOPOGRAPHY_DIR=asl_cfg_mod.TOPOGRAPHY_DIR,
        MSEED_FILE=asl_cfg_mod.MSEED_FILE,
        STATION_COORDS_FILE=asl_cfg_mod.STATION_COORDS_FILE,
        STATIONS=list(stations),
        START_TIME=asl_cfg_mod.START_TIME,
        END_TIME=asl_cfg_mod.END_TIME,
        CRATER_LAT=asl_cfg_mod.CRATER_LAT,
        CRATER_LON=asl_cfg_mod.CRATER_LON,
        FREQ_MIN=asl_cfg_mod.FREQ_MIN,
        FREQ_MAX=asl_cfg_mod.FREQ_MAX,
        WIN_LENGTH_S=asl_cfg_mod.WIN_LENGTH_S,
        WIN_STEP_S=asl_cfg_mod.WIN_STEP_S,
        SEARCH_AREA=asl_cfg_mod.SEARCH_HALF_WIDTH,
        SEARCH_BOX=asl_cfg_mod.SEARCH_BOX,
        SPATIAL_STEP=spatial_step,
        TEMPORAL_STEP=temporal_step,
        B=asl_cfg_mod.B,
        STORE_FULL_FIELDS=STORE_FULL_FIELDS,
    )
    load_kwargs = {
        "TOPOGRAPHY_DIR": cfg["TOPOGRAPHY_DIR"],
        "event": dict(
            stations=cfg["STATIONS"],
            start_time=cfg["START_TIME"],
            end_time=cfg["END_TIME"],
            mseed_file=cfg["MSEED_FILE"],
            coord_file=cfg["STATION_COORDS_FILE"],
            freq_min=cfg["FREQ_MIN"],
            freq_max=cfg["FREQ_MAX"],
            win_length_s=cfg["WIN_LENGTH_S"],
            win_step_s=cfg["WIN_STEP_S"],
        ),
    }
    localisation_kwargs = dict(
        crater_lat=cfg["CRATER_LAT"],
        crater_lon=cfg["CRATER_LON"],
        search_area=cfg["SEARCH_AREA"],
        search_box=cfg["SEARCH_BOX"],
        spatial_step=cfg["SPATIAL_STEP"],
        temporal_step=cfg["TEMPORAL_STEP"],
        b=cfg["B"],
        store_full_fields=cfg["STORE_FULL_FIELDS"],
    )
    return cfg, load_kwargs, localisation_kwargs, float(cfg["TEMPORAL_STEP"])


def _build_sara_spec(stations):
    spatial_step = sara_cfg_mod.SPATIAL_STEP if SARA_SPATIAL_STEP is None else SARA_SPATIAL_STEP
    temporal_step = sara_cfg_mod.WIN_STEP_S if SARA_TEMPORAL_STEP is None else SARA_TEMPORAL_STEP
    cfg = dict(
        TOPOGRAPHY_DIR=sara_cfg_mod.TOPOGRAPHY_DIR,
        MSEED_FILE=sara_cfg_mod.MSEED_FILE,
        STATION_COORDS_FILE=sara_cfg_mod.STATION_COORDS_FILE,
        STATIONS=list(stations),
        START_TIME=sara_cfg_mod.START_TIME,
        END_TIME=sara_cfg_mod.END_TIME,
        CRATER_LAT=sara_cfg_mod.CRATER_LAT,
        CRATER_LON=sara_cfg_mod.CRATER_LON,
        FREQ_MIN=sara_cfg_mod.FREQ_MIN,
        FREQ_MAX=sara_cfg_mod.FREQ_MAX,
        WIN_LENGTH_S=sara_cfg_mod.WIN_LENGTH_S,
        WIN_STEP_S=temporal_step,
        ENVELOPE_SMOOTH_S=sara_cfg_mod.ENVELOPE_SMOOTH_S,
        SEARCH_AREA=sara_cfg_mod.SEARCH_HALF_WIDTH,
        SEARCH_BOX=sara_cfg_mod.SEARCH_BOX,
        SPATIAL_STEP=spatial_step,
        DECAY_EXPONENT=sara_cfg_mod.DECAY_EXPONENT,
        STORE_FULL_FIELDS=STORE_FULL_FIELDS,
    )
    load_kwargs = {
        "TOPOGRAPHY_DIR": cfg["TOPOGRAPHY_DIR"],
        "event": dict(
            stations=cfg["STATIONS"],
            start_time=cfg["START_TIME"],
            end_time=cfg["END_TIME"],
            mseed_file=cfg["MSEED_FILE"],
            coord_file=cfg["STATION_COORDS_FILE"],
            freq_min=cfg["FREQ_MIN"],
            freq_max=cfg["FREQ_MAX"],
            win_length_s=cfg["WIN_LENGTH_S"],
            win_step_s=cfg["WIN_STEP_S"],
            envelope_smooth_s=cfg["ENVELOPE_SMOOTH_S"],
        ),
    }
    localisation_kwargs = dict(
        crater_lat=cfg["CRATER_LAT"],
        crater_lon=cfg["CRATER_LON"],
        search_area=cfg["SEARCH_AREA"],
        search_box=cfg["SEARCH_BOX"],
        spatial_step=cfg["SPATIAL_STEP"],
        decay_exponent=cfg["DECAY_EXPONENT"],
        store_full_fields=cfg["STORE_FULL_FIELDS"],
    )
    return cfg, load_kwargs, localisation_kwargs, float(cfg["WIN_STEP_S"])


def _extract_asl_shape(result):
    n_time = int(len(result["tplot"]))
    ny, nx = result["x_grid"].shape
    return nx, ny, n_time


def _extract_sara_shape(result):
    search_grid = result["search_grid"]
    ny, nx = search_grid.x_grid.shape
    n_time = int(len(result["window_times"]))
    return nx, ny, n_time


def _benchmark_method(method_name, stations):
    if method_name == "ASL":
        cfg, load_kwargs, localisation_kwargs, update_interval_s = _build_asl_spec(stations)
        loader_topo = asl_load_topography
        loader_data = asl_load_event_data
        localizer = asl_run_localization
        shape_getter = _extract_asl_shape
    elif method_name == "SARA":
        cfg, load_kwargs, localisation_kwargs, update_interval_s = _build_sara_spec(stations)
        loader_topo = sara_load_topography
        loader_data = sara_load_event_data
        localizer = amplitude_ratio_localization
        shape_getter = _extract_sara_shape
    else:
        raise ValueError(f"Unknown method '{method_name}'.")

    total_times = []
    last_result = None
    for _ in range(TOTAL_REPEATS):
        if VERBOSE_PROGRESS:
            print(
                f"  [{method_name}] total run {_ + 1}/{TOTAL_REPEATS} "
                f"with {len(stations)} stations..."
            )
        elapsed, (_, _, result) = _timed_total_run(
            loader_topo,
            loader_data,
            localizer,
            load_kwargs,
            localisation_kwargs,
        )
        total_times.append(elapsed)
        last_result = result
        if VERBOSE_PROGRESS:
            print(f"    completed in {elapsed:.3f} s")

    if VERBOSE_PROGRESS:
        print(f"  [{method_name}] loading data once for localisation-only repeats...")
    _, data, result, localisation_times = _timed_localisation_runs(
        loader_topo,
        loader_data,
        localizer,
        load_kwargs,
        localisation_kwargs,
        repeats=LOCALISATION_REPEATS,
    )
    if VERBOSE_PROGRESS:
        for idx, elapsed in enumerate(localisation_times, start=1):
            print(
                f"    [{method_name}] localisation-only run {idx}/{LOCALISATION_REPEATS} "
                f"completed in {elapsed:.3f} s"
            )
    if last_result is None:
        last_result = result

    nx, ny, n_time = shape_getter(last_result)
    station_count = len(stations)
    pair_count = comb(station_count, 2)
    duration_s = _data_duration_seconds(cfg["START_TIME"], cfg["END_TIME"])

    total_stats = _summary_stats(total_times)
    localisation_stats = _summary_stats(localisation_times)
    total_fraction_data_stats = _summary_stats(_series_ratios_with_fixed_denominator(total_times, duration_s))
    localisation_fraction_data_stats = _summary_stats(_series_ratios_with_fixed_denominator(localisation_times, duration_s))
    total_fraction_cadence_stats = _summary_stats(_series_ratios_with_fixed_denominator(total_times, update_interval_s))
    localisation_fraction_cadence_stats = _summary_stats(
        _series_ratios_with_fixed_denominator(localisation_times, update_interval_s)
    )
    total_speedup_rt_stats = _summary_stats(_series_ratios(duration_s, total_times))
    localisation_speedup_rt_stats = _summary_stats(_series_ratios(duration_s, localisation_times))
    total_speedup_cadence_stats = _summary_stats(_series_ratios(update_interval_s, total_times))
    localisation_speedup_cadence_stats = _summary_stats(_series_ratios(update_interval_s, localisation_times))

    summary = {
        "method": method_name,
        "stations": station_count,
        "pairs": pair_count,
        "grid_nx": nx,
        "grid_ny": ny,
        "time_slices": n_time,
        "data_duration_s": duration_s,
        "update_interval_s": update_interval_s,
        "total_runtime_mean_s": total_stats["mean_s"],
        "total_runtime_min_s": total_stats["min_s"],
        "total_runtime_max_s": total_stats["max_s"],
        "total_runtime_std_s": total_stats["std_s"],
        "localisation_runtime_mean_s": localisation_stats["mean_s"],
        "localisation_runtime_min_s": localisation_stats["min_s"],
        "localisation_runtime_max_s": localisation_stats["max_s"],
        "localisation_runtime_std_s": localisation_stats["std_s"],
        "total_runtime_per_slice_s": _safe_ratio(total_stats["mean_s"], n_time),
        "localisation_runtime_per_slice_s": _safe_ratio(localisation_stats["mean_s"], n_time),
        "total_runtime_fraction_of_data": total_fraction_data_stats["mean_s"],
        "total_runtime_fraction_of_data_std": total_fraction_data_stats["std_s"],
        "localisation_runtime_fraction_of_data": localisation_fraction_data_stats["mean_s"],
        "localisation_runtime_fraction_of_data_std": localisation_fraction_data_stats["std_s"],
        "total_runtime_fraction_of_update_interval": total_fraction_cadence_stats["mean_s"],
        "total_runtime_fraction_of_update_interval_std": total_fraction_cadence_stats["std_s"],
        "localisation_runtime_fraction_of_update_interval": localisation_fraction_cadence_stats["mean_s"],
        "localisation_runtime_fraction_of_update_interval_std": localisation_fraction_cadence_stats["std_s"],
        "total_speedup_vs_real_time": total_speedup_rt_stats["mean_s"],
        "total_speedup_vs_real_time_std": total_speedup_rt_stats["std_s"],
        "localisation_speedup_vs_real_time": localisation_speedup_rt_stats["mean_s"],
        "localisation_speedup_vs_real_time_std": localisation_speedup_rt_stats["std_s"],
        "total_speedup_vs_update_interval": total_speedup_cadence_stats["mean_s"],
        "total_speedup_vs_update_interval_std": total_speedup_cadence_stats["std_s"],
        "localisation_speedup_vs_update_interval": localisation_speedup_cadence_stats["mean_s"],
        "localisation_speedup_vs_update_interval_std": localisation_speedup_cadence_stats["std_s"],
        "theoretical_station_scaling_term": station_count if method_name == "ASL" else pair_count,
    }

    print(
        f"{method_name:4s} | stations={station_count:2d} | pairs={pair_count:2d} | "
        f"grid={nx}x{ny} | slices={n_time:3d} | "
        f"total={summary['total_runtime_mean_s']:.3f}s | "
        f"localisation={summary['localisation_runtime_mean_s']:.3f}s | "
        f"RT speedup={summary['localisation_speedup_vs_real_time']:.2f}x | "
        f"cadence speedup={summary['localisation_speedup_vs_update_interval']:.2f}x"
    )

    return summary


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    asl_defaults = _get_method_defaults("ASL")
    sara_defaults = _get_method_defaults("SARA")
    asl_stations = asl_defaults["stations"]
    sara_stations = sara_defaults["stations"]
    asl_station_counts = _base_station_counts(asl_stations)
    sara_station_counts = _base_station_counts(sara_stations)
    station_counts = sorted(set(asl_station_counts) | set(sara_station_counts))

    print("Benchmarking localisation runtime")
    print(f"ASL data window:  {asl_defaults['start_time']} to {asl_defaults['end_time']}")
    print(f"SARA data window: {sara_defaults['start_time']} to {sara_defaults['end_time']}")
    print(f"ASL station counts tested:  {asl_station_counts}")
    print(f"SARA station counts tested: {sara_station_counts}")
    print(f"Repeats: total={TOTAL_REPEATS}, localisation-only={LOCALISATION_REPEATS}")
    print(f"Verbose progress: {VERBOSE_PROGRESS}")
    print(f"Show internal analysis output: {SHOW_INTERNAL_FUNCTION_OUTPUT}")
    print(f"Store full fields: {STORE_FULL_FIELDS}")
    print(
        "Benchmark overrides: "
        f"ASL spatial={ASL_SPATIAL_STEP}, ASL temporal={ASL_TEMPORAL_STEP}, "
        f"SARA spatial={SARA_SPATIAL_STEP}, SARA temporal={SARA_TEMPORAL_STEP}"
    )
    print()

    summary_rows = []
    scaling_rows = []

    baseline_asl = _benchmark_method("ASL", asl_stations)
    baseline_sara = _benchmark_method("SARA", sara_stations)
    summary_rows.extend([baseline_asl, baseline_sara])

    print("\nStation-count scaling:")
    for count in station_counts:
        print(f"\nTesting subset size {count}:")

        if count in asl_station_counts:
            asl_subset = asl_stations[:count]
            print(f"  ASL stations: {asl_subset}")
            if count == len(asl_stations):
                asl_row = dict(baseline_asl)
            else:
                asl_row = _benchmark_method("ASL", asl_subset)
            scaling_rows.append(asl_row)

        if count in sara_station_counts:
            sara_subset = sara_stations[:count]
            print(f"  SARA stations: {sara_subset}")
            if count == len(sara_stations):
                sara_row = dict(baseline_sara)
            else:
                sara_row = _benchmark_method("SARA", sara_subset)
            scaling_rows.append(sara_row)

    _write_csv(SUMMARY_CSV, summary_rows)
    _write_csv(SCALING_CSV, scaling_rows)

    print("\nSaved benchmark summaries:")
    print(f"  {SUMMARY_CSV}")
    print(f"  {SCALING_CSV}")
    print("\nInterpretation notes:")
    print("  ASL kernel cost scales approximately with the number of stations.")
    print("  SARA kernel cost scales approximately with the number of station pairs, N(N-1)/2.")
    print("  Compare 'localisation_speedup_vs_real_time' against 1.0 for a near-real-time check.")
    print("  Compare 'localisation_speedup_vs_update_interval' against 1.0 to see whether the code can keep up with each new 0.5 s update.")
