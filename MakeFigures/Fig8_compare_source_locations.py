"""
Comparison map of source locations from ASL, SARA, and
the differential-arrival localisation workflow.
"""

from pathlib import Path
import sys

import matplotlib
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Rectangle

try:
    import contextily as ctx
except ImportError:
    ctx = None


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from AnalysisCodes.io_utils import load_results_bundle


ASL_OUTPUT = HERE / "RESULTS" / "ASL_output.npz"
SARA_OUTPUT = HERE / "RESULTS" / "SARA_output.npz"
DIFF_SUMMARY_CSV = HERE / "RESULTS" / "seismoacoustic_all_pick_combinations_summary.csv"
SAVE_FIG = HERE / "RESULTS" / "Fig_compare_source_locations.pdf"

ASL_GOOD_THRESHOLD = 0.66
SARA_GOOD_THRESHOLD = 0.85
TRY_SPYDER_INTERACTIVE_BACKEND = True
USE_SATELLITE_BASEMAP = True
BASEMAP_SOURCE = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
STATION_MARKER_SIZE = 95
CRATER_MARKER_SIZE = 220
SOURCE_MARKER_SIZE_MIN = 55
SOURCE_MARKER_SIZE_MAX = 320
DIFF_DENSITY_ALPHA = 0.78
DIFF_DENSITY_BINS = 50
AXIS_LABEL_FONTSIZE = 15
AXIS_TICK_FONTSIZE = 13
STATION_LABEL_FONTSIZE = 12
VOLCANO_LABEL_FONTSIZE = 14
TITLE_FONTSIZE = 14
LEGEND_FONTSIZE = 11
SCALEBAR_FONTSIZE = 12
COLORBAR_LABEL_FONTSIZE = 12
COLORBAR_TICK_FONTSIZE = 11
COLORBAR_FRACTION = 0.04
COLORBAR_SHRINK = 0.8
MAP_PADDING_M = 180.0
SIZE_SCALING_STATION = "OTVZ"
DOMAIN_EAST_MIN = 1.826e6
DOMAIN_EAST_MAX = 1.832e6
DOMAIN_NORTH_MIN = 5.659e6
DOMAIN_NORTH_MAX = 5.665e6
ASL_COLORMAP = "YlOrBr"
SARA_COLORMAP = "Greens"
DIFF_COLORMAP = "Greys"

def _enable_interactive_backend():
    if not TRY_SPYDER_INTERACTIVE_BACKEND:
        return
    if "spyder_kernels" not in sys.modules:
        return
    try:
        matplotlib.use("QtAgg")
    except Exception:
        pass


_enable_interactive_backend()

import matplotlib.pyplot as plt


def _plot_window(bundle):
    start = float(bundle.get("plot_window_start_s", 0.0))
    end = float(bundle.get("plot_window_end_s", np.inf))
    return start, end


def _bundle_locations(bundle, min_quality):
    times = np.asarray(bundle["result_time"], dtype=float)
    quality = np.asarray(bundle["quality"], dtype=float)
    plot_start_s, plot_end_s = _plot_window(bundle)
    mask = (times >= plot_start_s) & (times <= plot_end_s) & (quality >= min_quality)
    return {
        "x": np.asarray(bundle["x_absolute"], dtype=float)[mask],
        "y": np.asarray(bundle["y_absolute"], dtype=float)[mask],
        "mask": mask,
        "times_absolute": times,
        "time_since_start": times[mask] - plot_start_s,
        "duration": plot_end_s - plot_start_s,
    }


def _station_index(bundle, station):
    stations = np.asarray(bundle["station_names"]).tolist()
    if station not in stations:
        raise ValueError(f"Station '{station}' not found in {bundle['analysis_name']} bundle.")
    return stations.index(station)


def _station_rms_at_times(bundle, station, target_time):
    station_idx = _station_index(bundle, station)
    rms_time = np.asarray(bundle["rms_time"], dtype=float)
    rms_values = np.asarray(bundle["rms_values"], dtype=float)[station_idx]
    target_time = np.asarray(target_time, dtype=float)
    return np.interp(target_time, rms_time, rms_values, left=rms_values[0], right=rms_values[-1])


def _scale_marker_sizes(values, min_size, max_size, value_min, value_max):
    values = np.asarray(values, dtype=float)
    if value_max <= value_min:
        return np.full(values.shape, 0.5 * (min_size + max_size), dtype=float)
    scale = np.clip((values - value_min) / (value_max - value_min), 0.0, 1.0)
    return min_size + scale * (max_size - min_size)


def _add_station_labels(ax, station_names, station_easting, station_northing):
    outline = [pe.withStroke(linewidth=2.2, foreground="black")]
    for label, east, north in zip(station_names, station_easting, station_northing):
        label = str(label)
        if label == "WTVZ":
            xytext = (0, -10)
            va = "top"
        else:
            xytext = (0, 8)
            va = "bottom"
        ax.annotate(
            label,
            xy=(east, north),
            xytext=xytext,
            textcoords="offset points",
            fontsize=STATION_LABEL_FONTSIZE,
            fontweight="bold",
            color="white",
            ha="center",
            va=va,
            path_effects=outline,
            zorder=13,
        )


def _add_reference_markers(ax, bundle):
    station_easting = np.asarray(bundle["station_easting"], dtype=float)
    station_northing = np.asarray(bundle["station_northing"], dtype=float)
    station_names = np.asarray(bundle["station_names"])

    ax.scatter(
        station_easting,
        station_northing,
        c="yellow",
        marker="s",
        s=STATION_MARKER_SIZE,
        edgecolor="black",
        linewidths=0.8,
        zorder=10,
        #label="Stations",
    )
    _add_station_labels(ax, station_names, station_easting, station_northing)

    crater_x = float(np.asarray(bundle["crater_easting"], dtype=float)[0])
    crater_y = float(np.asarray(bundle["crater_northing"], dtype=float)[0])
    ax.scatter(
        crater_x,
        crater_y,
        marker="*",
        s=CRATER_MARKER_SIZE,
        color="red",
        edgecolor="black",
        linewidth=0.9,
        zorder=14,
        #label="Ngauruhoe",
    )
    ax.annotate(
        "Ngauruhoe",
        xy=(crater_x, crater_y),
        xytext=(0, 10),
        textcoords="offset points",
        fontsize=VOLCANO_LABEL_FONTSIZE,
        fontweight="bold",
        color="red",
        ha="center",
        va="bottom",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
        zorder=15,
    )


def _axis_limits(*coordinate_sets):
    x_values = np.concatenate([np.asarray(x, dtype=float).ravel() for x, _ in coordinate_sets])
    y_values = np.concatenate([np.asarray(y, dtype=float).ravel() for _, y in coordinate_sets])
    return (
        float(np.nanmin(x_values) - MAP_PADDING_M),
        float(np.nanmax(x_values) + MAP_PADDING_M),
        float(np.nanmin(y_values) - MAP_PADDING_M),
        float(np.nanmax(y_values) + MAP_PADDING_M),
    )


def _add_size_legend(ax, rms_min, rms_max):
    levels = np.array([1e-6, 2e-6, 3e-6], dtype=float)
    sizes = _scale_marker_sizes(
        levels,
        SOURCE_MARKER_SIZE_MIN,
        SOURCE_MARKER_SIZE_MAX,
        rms_min,
        rms_max,
    )
    handles = []
    labels = []
    for level, size in zip(levels, sizes):
        circle = Line2D(
            [],
            [],
            linestyle="None",
            marker="o",
            markersize=np.sqrt(size),
            markerfacecolor="0.7",
            markeredgecolor="black",
            markeredgewidth=0.7,
        )
        triangle = Line2D(
            [],
            [],
            linestyle="None",
            marker="^",
            markersize=np.sqrt(size),
            markerfacecolor="0.7",
            markeredgecolor="black",
            markeredgewidth=0.7,
        )
        handles.append((circle, triangle))
        labels.append(f"{level * 1e6:.0f} μm/s")
    ax.legend(
        handles=handles,
        labels=labels,
        title="OTVZ RMS amplitude",
        loc="lower left",
        handler_map={tuple: HandlerTuple(ndivide=None, pad=1.0)},
        handletextpad=0.8,
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_FONTSIZE,
        frameon=True,
    )


def _colorbar(fig, mappable, ax, label, pad):
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        shrink=COLORBAR_SHRINK,
        fraction=COLORBAR_FRACTION,
        pad=pad,
    )
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)
    cbar.set_label(label, fontsize=COLORBAR_LABEL_FONTSIZE)
    return cbar


def _set_density_colorbar_ticks(cbar, mappable):
    counts = np.asarray(mappable.get_array(), dtype=float)
    counts = counts[np.isfinite(counts)]
    if counts.size == 0:
        cbar.set_ticks([1.0])
        cbar.set_ticklabels(["1"])
        return

    count_max = float(np.nanmax(counts))
    if count_max <= 1.0:
        cbar.set_ticks([1.0])
        cbar.set_ticklabels(["1"])
        return

    ticks = np.unique(np.concatenate(([1.0], cbar.get_ticks())))
    ticks = ticks[(ticks >= 1.0) & (ticks <= count_max)]
    cbar.set_ticks(ticks)

    tick_labels = []
    for tick in ticks:
        if np.isclose(tick, 1.0):
            tick_labels.append("1")
        elif np.isclose(tick, round(tick)):
            tick_labels.append(f"{int(round(tick))}")
        else:
            tick_labels.append(f"{tick:g}")
    cbar.set_ticklabels(tick_labels)


def _resolve_colormap(cmap):
    if isinstance(cmap, str):
        return plt.get_cmap(cmap)
    return cmap


def _add_scale_bar(ax, total_length_m=2000.0, segment_length_m=500.0, tick_step_m=500.0):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_range = x_max - x_min
    y_range = y_max - y_min

    bar_height = 0.018 * y_range
    margin_x = 0.05 * x_range
    margin_y = 0.055 * y_range

    x0 = x_max - margin_x - total_length_m
    y0 = y_min + margin_y

    outline = [pe.withStroke(linewidth=2.2, foreground="black")]
    segment_colors = ["black", "white"]

    for i in range(int(total_length_m / segment_length_m)):
        ax.add_patch(
            Rectangle(
                (x0 + i * segment_length_m, y0),
                segment_length_m,
                bar_height,
                facecolor=segment_colors[i % len(segment_colors)],
                edgecolor="black",
                linewidth=0.9,
                zorder=16,
            )
        )

    tick_positions_m = np.arange(0.0, total_length_m + 0.5 * tick_step_m, tick_step_m)
    tick_labels = []
    for distance_m in tick_positions_m:
        distance_km = distance_m / 1000.0
        if np.isclose(distance_km, 0.0):
            tick_labels.append("0")
        elif np.isclose(distance_km, total_length_m / 1000.0):
            tick_labels.append(f"{distance_km:g} km")
        else:
            tick_labels.append(f"{distance_km:g}")

    for distance_m, label in zip(tick_positions_m, tick_labels):
        ax.text(
            x0 + distance_m,
            y0 + bar_height + 0.012 * y_range,
            label,
            ha="center",
            va="bottom",
            fontsize=SCALEBAR_FONTSIZE,
            color="white",
            path_effects=outline,
            zorder=17,
        )


if __name__ == "__main__":
    asl = load_results_bundle(ASL_OUTPUT)
    sara = load_results_bundle(SARA_OUTPUT)
    diff_summary = pd.read_csv(DIFF_SUMMARY_CSV)

    asl_loc = _bundle_locations(asl, ASL_GOOD_THRESHOLD)
    sara_loc = _bundle_locations(sara, SARA_GOOD_THRESHOLD)
    diff_x = diff_summary["best_x_m"].to_numpy(dtype=float)
    diff_y = diff_summary["best_y_m"].to_numpy(dtype=float)

    time_max = max(float(asl_loc["duration"]), float(sara_loc["duration"]))
    time_norm = Normalize(vmin=0.0, vmax=time_max if time_max > 0 else 1.0)
    asl_cmap = _resolve_colormap(ASL_COLORMAP)
    sara_cmap = _resolve_colormap(SARA_COLORMAP)
    diff_cmap = _resolve_colormap(DIFF_COLORMAP)

    asl_otvz_rms = _station_rms_at_times(asl, SIZE_SCALING_STATION, asl_loc["times_absolute"])
    sara_otvz_rms = _station_rms_at_times(sara, SIZE_SCALING_STATION, sara_loc["times_absolute"])
    combined_rms = np.concatenate([asl_otvz_rms[asl_loc["mask"]], sara_otvz_rms[sara_loc["mask"]]])
    if combined_rms.size == 0:
        rms_min = 0.0
        rms_max = 1.0
    else:
        rms_min = float(np.nanmin(combined_rms))
        rms_max = float(np.nanmax(combined_rms))
    asl_sizes = _scale_marker_sizes(
        asl_otvz_rms[asl_loc["mask"]],
        SOURCE_MARKER_SIZE_MIN,
        SOURCE_MARKER_SIZE_MAX,
        rms_min,
        rms_max,
    )
    sara_sizes = _scale_marker_sizes(
        sara_otvz_rms[sara_loc["mask"]],
        SOURCE_MARKER_SIZE_MIN,
        SOURCE_MARKER_SIZE_MAX,
        rms_min,
        rms_max,
    )

    fig, ax = plt.subplots(figsize=(12.8, 9.6), constrained_layout=True)

    x_min = DOMAIN_EAST_MIN
    x_max = DOMAIN_EAST_MAX
    y_min = DOMAIN_NORTH_MIN
    y_max = DOMAIN_NORTH_MAX
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    if USE_SATELLITE_BASEMAP and ctx is not None:
        ctx.add_basemap(
            ax,
            source=BASEMAP_SOURCE,
            zoom="auto",
            crs="EPSG:2193",
        )
    else:
        ax.set_facecolor("0.96")

    diff_density = ax.hexbin(
        diff_x,
        diff_y,
        gridsize=DIFF_DENSITY_BINS,
        cmap=diff_cmap,
        mincnt=1,
        linewidths=0.0,
        alpha=DIFF_DENSITY_ALPHA,
        zorder=5,
        extent=(x_min, x_max, y_min, y_max),
    )

    ax.scatter(
        sara_loc["x"],
        sara_loc["y"],
        c=sara_loc["time_since_start"],
        cmap=sara_cmap,
        norm=time_norm,
        s=sara_sizes,
        marker="^",
        edgecolor="black",
        linewidth=0.45,
        alpha=0.68,
        zorder=8,
        label="SARA",
    )
    ax.scatter(
        asl_loc["x"],
        asl_loc["y"],
        c=asl_loc["time_since_start"],
        cmap=asl_cmap,
        norm=time_norm,
        s=asl_sizes,
        marker="o",
        edgecolor="black",
        linewidth=0.45,
        alpha=0.72,
        zorder=9,
        label="ASL",
    )

    _add_reference_markers(ax, asl)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting (m)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Northing (m)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)
    ax.xaxis.offsetText.set_fontsize(AXIS_TICK_FONTSIZE)
    ax.yaxis.offsetText.set_fontsize(AXIS_TICK_FONTSIZE)

    legend_handles = [
        Line2D(
            [],
            [],
            linestyle="None",
            marker="o",
            markersize=np.sqrt(0.5 * (SOURCE_MARKER_SIZE_MIN + SOURCE_MARKER_SIZE_MAX)),
            markerfacecolor="0.55",
            markeredgecolor="black",
            markeredgewidth=0.7,
            label="ASL",
        ),
        Line2D(
            [],
            [],
            linestyle="None",
            marker="^",
            markersize=np.sqrt(0.5 * (SOURCE_MARKER_SIZE_MIN + SOURCE_MARKER_SIZE_MAX)),
            markerfacecolor="0.55",
            markeredgecolor="black",
            markeredgewidth=0.7,
            label="SARA",
        ),
        Line2D(
            [],
            [],
            linestyle="None",
            marker="h",
            markersize=np.sqrt(0.5 * (SOURCE_MARKER_SIZE_MIN + SOURCE_MARKER_SIZE_MAX)),
            markerfacecolor="0.55",
            markeredgecolor="black",
            label="Differential-arrival",
        ),
        Line2D(
            [],
            [],
            linestyle="None",
            marker="s",
            markersize=np.sqrt(STATION_MARKER_SIZE),
            markerfacecolor="yellow",
            markeredgecolor="black",
            markeredgewidth=0.8,
            label="Stations",
        ),
        Line2D(
            [],
            [],
            linestyle="None",
            marker="*",
            markersize=np.sqrt(CRATER_MARKER_SIZE),
            markerfacecolor="red",
            markeredgecolor="black",
            markeredgewidth=0.8,
            label="Ngauruhoe",
        ),
    ]
    method_legend = ax.legend(handles=legend_handles, loc="upper right", fontsize=LEGEND_FONTSIZE, frameon=True)
    ax.add_artist(method_legend)
    _add_size_legend(ax, rms_min, rms_max)
    _add_scale_bar(ax)

    asl_sm = ScalarMappable(norm=time_norm, cmap=asl_cmap)
    sara_sm = ScalarMappable(norm=time_norm, cmap=sara_cmap)
    diff_cbar = _colorbar(fig, diff_density, ax, "Differential-arrival density (counts per hexbin)", pad=0.07)
    _colorbar(fig, sara_sm, ax, "SARA: Time (s) since 2026-03-21 13:36:05 (UTC)", pad=0.04)
    _colorbar(fig, asl_sm, ax, "ASL: Time (s) since 2026-03-21 13:36:05 (UTC)", pad=0.01)
    _set_density_colorbar_ticks(diff_cbar, diff_density)

    SAVE_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_FIG, bbox_inches="tight")
    print(f"Saved figure -> {SAVE_FIG}")
    plt.show()
