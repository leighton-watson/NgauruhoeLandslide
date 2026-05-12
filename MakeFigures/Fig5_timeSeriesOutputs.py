"""
Plot time-series outputs from ASL and SARA from saved output bundles.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from AnalysisCodes.io_utils import load_results_bundle


ASL_OUTPUT = HERE / "RESULTS" / "ASL_output.npz"
SARA_OUTPUT = HERE / "RESULTS" / "SARA_output.npz"
SAVE_FIG = HERE / "RESULTS" / "Fig_timeSeriesOutputs.pdf"

RMS_STATIONS = ["OTVZ", "SNVZ", "NOVZ"]

# Change these station pairs directly in the code when you want a different comparison.
RATIO_PAIRS = [
    ("OTVZ", "SNVZ"),
    ("OTVZ", "NOVZ"),
    ("SNVZ", "NOVZ"),
]
RATIO_PAIR_COLORS = {
    ("OTVZ", "SNVZ"): "#ffb000",
    ("OTVZ", "NOVZ"): "#dc267f",
    ("SNVZ", "NOVZ"): "#785ef0",
}

# Choose which saved RMS series to use for the ratio panel: "ASL" or "SARA".
RATIO_SOURCE = "SARA"
RATIO_LOG_SCALE = False

# Threshold used to define "good" localisation points in the comparison plots.
ASL_GOOD_THRESHOLD = 0.66
SARA_GOOD_THRESHOLD = 0.85

TITLE_FONTSIZE = 14
AXIS_LABEL_FONTSIZE = 13
AXIS_TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 12

ASL_COLORMAP = "YlOrBr"
SARA_COLORMAP = "Greens"
ASL_COLOR_LEVEL = 0.6
SARA_COLOR_LEVEL = 0.6
ASL_LIGHT_COLOR_LEVEL = 0.35
SARA_LIGHT_COLOR_LEVEL = 0.35

ASL_COLOR = plt.get_cmap(ASL_COLORMAP)(ASL_COLOR_LEVEL)
ASL_LIGHT_COLOR = plt.get_cmap(ASL_COLORMAP)(ASL_LIGHT_COLOR_LEVEL)
SARA_COLOR = plt.get_cmap(SARA_COLORMAP)(SARA_COLOR_LEVEL)
SARA_LIGHT_COLOR = plt.get_cmap(SARA_COLORMAP)(SARA_LIGHT_COLOR_LEVEL)


def _plot_window(bundle):
    start = float(bundle.get("plot_window_start_s", 0.0))
    end = float(bundle.get("plot_window_end_s", np.inf))
    return start, end


def _crop_time_series(time, values, bundle):
    start, end = _plot_window(bundle)
    time = np.asarray(time, dtype=float)
    values = np.asarray(values)
    mask = (time >= start) & (time <= end)
    return time[mask] - start, values[mask]


def _time_axis_label(bundle):
    return f"Time (s) since {str(bundle.get('plot_start_time', bundle['start_time'])).replace('T', ' ')} UTC"


def _station_index(bundle, station):
    stations = np.asarray(bundle["station_names"]).tolist()
    if station not in stations:
        raise ValueError(f"Station '{station}' not found in {bundle['analysis_name']} bundle.")
    return stations.index(station)


def _station_ratio_series(bundle, station_a, station_b):
    i = _station_index(bundle, station_a)
    j = _station_index(bundle, station_b)
    rms_a = np.asarray(bundle["rms_values"])[i]
    rms_b = np.asarray(bundle["rms_values"])[j]
    ratio = np.divide(
        rms_a,
        rms_b,
        out=np.full_like(rms_a, np.nan, dtype=float),
        where=np.abs(rms_b) > 0,
    )
    return np.asarray(bundle["rms_time"], dtype=float), ratio


def _grid_axes(bundle):
    x_axis = np.asarray(bundle["map_X"], dtype=float)[0, :]
    y_axis = np.asarray(bundle["map_Y"], dtype=float)[:, 0]
    z_grid = np.asarray(bundle["map_C"], dtype=float)
    return x_axis, y_axis, z_grid


def _sample_grid_nearest(bundle, x, y):
    x_axis, y_axis, z_grid = _grid_axes(bundle)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ix = np.abs(x[:, None] - x_axis[None, :]).argmin(axis=1)
    iy = np.abs(y[:, None] - y_axis[None, :]).argmin(axis=1)
    return z_grid[iy, ix]


def _align_for_correlation(time_a, values_a, time_b, values_b):
    time_a = np.asarray(time_a, dtype=float)
    values_a = np.asarray(values_a, dtype=float)
    time_b = np.asarray(time_b, dtype=float)
    values_b = np.asarray(values_b, dtype=float)

    valid_a = np.isfinite(time_a) & np.isfinite(values_a)
    valid_b = np.isfinite(time_b) & np.isfinite(values_b)
    if not np.any(valid_a) or not np.any(valid_b):
        return np.array([], dtype=float), np.array([], dtype=float)

    time_a = time_a[valid_a]
    values_a = values_a[valid_a]
    time_b = time_b[valid_b]
    values_b = values_b[valid_b]

    if time_a.size == time_b.size and np.allclose(time_a, time_b):
        return time_a, values_a, values_b

    overlap_start = max(time_a.min(), time_b.min())
    overlap_end = min(time_a.max(), time_b.max())
    overlap_mask = (time_a >= overlap_start) & (time_a <= overlap_end)
    if not np.any(overlap_mask):
        empty = np.array([], dtype=float)
        return empty, empty, empty

    ref_time = time_a[overlap_mask]
    aligned_a = values_a[overlap_mask]
    aligned_b = np.interp(ref_time, time_b, values_b)
    return ref_time, aligned_a, aligned_b


def _pearson_from_aligned(aligned_a, aligned_b):
    aligned_a = np.asarray(aligned_a, dtype=float)
    aligned_b = np.asarray(aligned_b, dtype=float)
    if aligned_a.size < 2:
        return np.nan, 0
    if np.allclose(aligned_a, aligned_a[0]) or np.allclose(aligned_b, aligned_b[0]):
        return np.nan, aligned_a.size
    return float(np.corrcoef(aligned_a, aligned_b)[0, 1]), aligned_a.size


def _pearson_correlation(time_a, values_a, time_b, values_b):
    _, aligned_a, aligned_b = _align_for_correlation(time_a, values_a, time_b, values_b)
    return _pearson_from_aligned(aligned_a, aligned_b)


def _pearson_correlation_good_only(
    time_a,
    values_a,
    quality_a,
    threshold_a,
    time_b,
    values_b,
    quality_b,
    threshold_b,
):
    ref_time, aligned_a, aligned_b = _align_for_correlation(time_a, values_a, time_b, values_b)
    if ref_time.size == 0:
        return np.nan, 0

    _, aligned_quality_a, aligned_quality_b = _align_for_correlation(time_a, quality_a, time_b, quality_b)
    common_good = (aligned_quality_a >= threshold_a) & (aligned_quality_b >= threshold_b)
    return _pearson_from_aligned(aligned_a[common_good], aligned_b[common_good])


if __name__ == "__main__":
    asl = load_results_bundle(ASL_OUTPUT)
    sara = load_results_bundle(SARA_OUTPUT)
    ratio_bundle = asl if RATIO_SOURCE.upper() == "ASL" else sara
    plot_start_s, plot_end_s = _plot_window(asl)
    plot_duration = plot_end_s - plot_start_s

    rms_time = None
    rms_series = {}
    rms_colors = {
        "OTVZ": "#00a6fb",
        "SNVZ": "#7b2cbf",
        "NOVZ": "#00cc99",
    }
    for station in RMS_STATIONS:
        station_idx = _station_index(asl, station)
        station_time, station_rms = _crop_time_series(
            np.asarray(asl["rms_time"], dtype=float),
            np.asarray(asl["rms_values"])[station_idx],
            asl,
        )
        rms_time = station_time
        rms_series[station] = station_rms

    fig, axes = plt.subplots(4, 2, figsize=(15, 14))

    ax_wave = axes[0, 0]
    ax_ratio = axes[0, 1]
    ax_asl_q = axes[1, 0]
    ax_sara_q = axes[1, 1]
    ax_east = axes[2, 0]
    ax_north = axes[2, 1]
    ax_elev = axes[3, 0]
    ax_empty = axes[3, 1]

    for station in RMS_STATIONS:
        ax_wave.plot(
            rms_time,
            rms_series[station],
            linewidth=1.8,
            color=rms_colors.get(station),
            label=station,
        )
    ax_wave.set_ylabel("RMS (m/s)", fontsize=AXIS_LABEL_FONTSIZE)
    ax_wave.set_title("(a) RMS amplitudes", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax_wave.grid(True, alpha=0.3)
    ax_wave.tick_params(labelbottom=False)
    ax_wave.set_xlim(0.0, plot_duration)
    ax_wave.set_ylim(0.0, 4e-6)
    ax_wave.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)
    ax_wave.legend(loc="best", fontsize=LEGEND_FONTSIZE)

    for station_a, station_b in RATIO_PAIRS:
        ratio_time, ratio = _station_ratio_series(ratio_bundle, station_a, station_b)
        ratio_time, ratio = _crop_time_series(ratio_time, ratio, ratio_bundle)
        ax_ratio.plot(
            ratio_time,
            ratio,
            linewidth=1.5,
            color=RATIO_PAIR_COLORS.get((station_a, station_b)),
            label=f"{station_a}/{station_b}",
        )
    ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    if RATIO_LOG_SCALE:
        ax_ratio.set_yscale("log")
    ax_ratio.set_ylabel("RMS ratio", fontsize=AXIS_LABEL_FONTSIZE)
    ax_ratio.set_title(f"(b) Station ratios from {ratio_bundle['analysis_name']}", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax_ratio.grid(True, alpha=0.3)
    ax_ratio.legend(loc="best", fontsize=LEGEND_FONTSIZE)
    ax_ratio.tick_params(labelbottom=False)
    ax_ratio.set_xlim(0.0, plot_duration)
    ax_ratio.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)

    asl_time, asl_quality = _crop_time_series(np.asarray(asl["result_time"], dtype=float), np.asarray(asl["quality"], dtype=float), asl)
    ax_asl_q.scatter(
        asl_time,
        asl_quality,
        s=30,
        color=ASL_COLOR,
        edgecolor="none",
        alpha=0.95,
    )
    ax_asl_q.axhline(
        ASL_GOOD_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        label=f"Threshold = {ASL_GOOD_THRESHOLD:.2f}",
    )
    ax_asl_q.set_ylabel(r"Maximum $R^2$", fontsize=AXIS_LABEL_FONTSIZE)
    ax_asl_q.set_title("(c) ASL localisation quality", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax_asl_q.grid(True, alpha=0.3)
    ax_asl_q.tick_params(labelbottom=False)
    ax_asl_q.set_xlim(0.0, plot_duration)
    ax_asl_q.legend(loc="best", fontsize=LEGEND_FONTSIZE)
    ax_asl_q.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)

    sara_time, sara_quality = _crop_time_series(np.asarray(sara["result_time"], dtype=float), np.asarray(sara["quality"], dtype=float), sara)
    ax_sara_q.scatter(
        sara_time,
        sara_quality,
        s=30,
        color=SARA_COLOR,
        edgecolor="none",
        alpha=0.95,
    )
    ax_sara_q.axhline(
        SARA_GOOD_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        label=f"Threshold = {SARA_GOOD_THRESHOLD:.2f}",
    )
    ax_sara_q.set_ylabel("Best score, S", fontsize=AXIS_LABEL_FONTSIZE)
    ax_sara_q.set_title("(d) SARA quality score", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax_sara_q.grid(True, alpha=0.3)
    ax_sara_q.tick_params(labelbottom=False)
    ax_sara_q.set_xlim(0.0, plot_duration)
    ax_sara_q.legend(loc="best", fontsize=LEGEND_FONTSIZE)
    ax_sara_q.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)

    asl_time, asl_east = _crop_time_series(np.asarray(asl["result_time"], dtype=float), np.asarray(asl["x_offset"], dtype=float), asl)
    sara_time, sara_east = _crop_time_series(np.asarray(sara["result_time"], dtype=float), np.asarray(sara["x_offset"], dtype=float), sara)
    asl_good = asl_quality >= ASL_GOOD_THRESHOLD
    sara_good = sara_quality >= SARA_GOOD_THRESHOLD
    ax_east.scatter(
        asl_time,
        asl_east,
        s=45,
        color=ASL_LIGHT_COLOR,
        edgecolor="none",
        alpha=0.9,
        label="ASL all points",
    )
    ax_east.scatter(
        sara_time,
        sara_east,
        s=45,
        color=SARA_LIGHT_COLOR,
        edgecolor="none",
        alpha=0.9,
        label="SARA all points",
    )
    ax_east.scatter(
        asl_time[asl_good],
        asl_east[asl_good],
        s=48,
        color=ASL_COLOR,
        edgecolor="black",
        linewidth=0.4,
        zorder=5,
        label=f"ASL >= {ASL_GOOD_THRESHOLD:.2f}",
    )
    ax_east.scatter(
        sara_time[sara_good],
        sara_east[sara_good],
        s=48,
        color=SARA_COLOR,
        edgecolor="black",
        linewidth=0.4,
        zorder=5,
        label=f"SARA >= {SARA_GOOD_THRESHOLD:.2f}",
    )
    ax_east.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax_east.set_xlabel(_time_axis_label(asl), fontsize=AXIS_LABEL_FONTSIZE)
    ax_east.set_ylabel("ΔEasting (m)", fontsize=AXIS_LABEL_FONTSIZE)
    ax_east.set_title("(e) Easting", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax_east.grid(True, alpha=0.3)
    ax_east.set_xlim(0.0, plot_duration)
    ax_east.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)

    asl_time, asl_north = _crop_time_series(np.asarray(asl["result_time"], dtype=float), np.asarray(asl["y_offset"], dtype=float), asl)
    sara_time, sara_north = _crop_time_series(np.asarray(sara["result_time"], dtype=float), np.asarray(sara["y_offset"], dtype=float), sara)
    ax_north.scatter(
        asl_time,
        asl_north,
        s=45,
        color=ASL_LIGHT_COLOR,
        edgecolor="none",
        alpha=0.9,
        label="ASL all points",
    )
    ax_north.scatter(
        sara_time,
        sara_north,
        s=45,
        color=SARA_LIGHT_COLOR,
        edgecolor="none",
        alpha=0.9,
        label="SARA all points",
    )
    ax_north.scatter(
        asl_time[asl_good],
        asl_north[asl_good],
        s=48,
        color=ASL_COLOR,
        edgecolor="black",
        linewidth=0.4,
        zorder=5,
        label=f"ASL >= {ASL_GOOD_THRESHOLD:.2f}",
    )
    ax_north.scatter(
        sara_time[sara_good],
        sara_north[sara_good],
        s=48,
        color=SARA_COLOR,
        edgecolor="black",
        linewidth=0.4,
        zorder=5,
        label=f"SARA >= {SARA_GOOD_THRESHOLD:.2f}",
    )
    ax_north.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax_north.set_xlabel(_time_axis_label(asl), fontsize=AXIS_LABEL_FONTSIZE)
    ax_north.set_ylabel("ΔNorthing (m)", fontsize=AXIS_LABEL_FONTSIZE)
    ax_north.set_title("(f) Northing", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax_north.grid(True, alpha=0.3)
    ax_north.set_xlim(0.0, plot_duration)
    ax_north.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)

    _, asl_x = _crop_time_series(
        np.asarray(asl["result_time"], dtype=float),
        np.asarray(asl["x_absolute"], dtype=float),
        asl,
    )
    _, asl_y = _crop_time_series(
        np.asarray(asl["result_time"], dtype=float),
        np.asarray(asl["y_absolute"], dtype=float),
        asl,
    )
    _, sara_x = _crop_time_series(
        np.asarray(sara["result_time"], dtype=float),
        np.asarray(sara["x_absolute"], dtype=float),
        sara,
    )
    _, sara_y = _crop_time_series(
        np.asarray(sara["result_time"], dtype=float),
        np.asarray(sara["y_absolute"], dtype=float),
        sara,
    )
    asl_elev = _sample_grid_nearest(asl, asl_x, asl_y)
    sara_elev = _sample_grid_nearest(sara, sara_x, sara_y)

    east_corr, east_n = _pearson_correlation(asl_time, asl_east, sara_time, sara_east)
    north_corr, north_n = _pearson_correlation(asl_time, asl_north, sara_time, sara_north)
    elev_corr, elev_n = _pearson_correlation(asl_time, asl_elev, sara_time, sara_elev)
    east_corr_good, east_n_good = _pearson_correlation_good_only(
        asl_time,
        asl_east,
        asl_quality,
        ASL_GOOD_THRESHOLD,
        sara_time,
        sara_east,
        sara_quality,
        SARA_GOOD_THRESHOLD,
    )
    north_corr_good, north_n_good = _pearson_correlation_good_only(
        asl_time,
        asl_north,
        asl_quality,
        ASL_GOOD_THRESHOLD,
        sara_time,
        sara_north,
        sara_quality,
        SARA_GOOD_THRESHOLD,
    )
    elev_corr_good, elev_n_good = _pearson_correlation_good_only(
        asl_time,
        asl_elev,
        asl_quality,
        ASL_GOOD_THRESHOLD,
        sara_time,
        sara_elev,
        sara_quality,
        SARA_GOOD_THRESHOLD,
    )

    print("ASL vs SARA Pearson correlations over the plotting window:")
    print(f"  Easting   : r = {east_corr:.3f} (n = {east_n})")
    print(f"  Northing  : r = {north_corr:.3f} (n = {north_n})")
    print(f"  Elevation : r = {elev_corr:.3f} (n = {elev_n})")
    print("ASL vs SARA Pearson correlations for common good points only:")
    print(f"  Easting   : r = {east_corr_good:.3f} (n = {east_n_good})")
    print(f"  Northing  : r = {north_corr_good:.3f} (n = {north_n_good})")
    print(f"  Elevation : r = {elev_corr_good:.3f} (n = {elev_n_good})")

    ax_elev.scatter(
        asl_time,
        asl_elev,
        s=45,
        color=ASL_LIGHT_COLOR,
        edgecolor="none",
        alpha=0.9,
        label="ASL all points",
    )
    ax_elev.scatter(
        sara_time,
        sara_elev,
        s=45,
        color=SARA_LIGHT_COLOR,
        edgecolor="none",
        alpha=0.9,
        label="SARA all points",
    )
    ax_elev.scatter(
        asl_time[asl_good],
        asl_elev[asl_good],
        s=48,
        color=ASL_COLOR,
        edgecolor="black",
        linewidth=0.4,
        zorder=5,
        label=f"ASL >= {ASL_GOOD_THRESHOLD:.2f}",
    )
    ax_elev.scatter(
        sara_time[sara_good],
        sara_elev[sara_good],
        s=48,
        color=SARA_COLOR,
        edgecolor="black",
        linewidth=0.4,
        zorder=5,
        label=f"SARA >= {SARA_GOOD_THRESHOLD:.2f}",
    )
    ax_elev.set_xlabel(_time_axis_label(asl), fontsize=AXIS_LABEL_FONTSIZE)
    ax_elev.set_ylabel("Elevation (m)", fontsize=AXIS_LABEL_FONTSIZE)
    ax_elev.set_title("(g) Elevation", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax_elev.grid(True, alpha=0.3)
    ax_elev.set_xlim(0.0, plot_duration)
    ax_elev.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)

    ax_empty.axis("off")
    legend_handles, legend_labels = ax_east.get_legend_handles_labels()
    ax_empty.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if SAVE_FIG is not None:
        SAVE_FIG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(SAVE_FIG, bbox_inches="tight")
        print(f"Saved figure → {SAVE_FIG}")

    plt.show()
