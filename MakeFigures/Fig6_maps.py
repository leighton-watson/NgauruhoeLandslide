"""
Plot ASL and SARA source-location maps from saved output bundles.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

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
SAVE_FIG = HERE / "RESULTS" / "Fig_maps.pdf"

# Threshold used to define "good" localisation points that are plotted on the maps.
ASL_GOOD_THRESHOLD = 0.66
SARA_GOOD_THRESHOLD = 0.85
SIZE_SCALING_STATION = "OTVZ"
USE_SATELLITE_BASEMAP = True
BASEMAP_SOURCE = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
SHOW_CONTOURS = False
STATION_MARKER_SIZE = 100
SOURCE_MARKER_SIZE_MIN = 45
SOURCE_MARKER_SIZE_MAX = 300
CRATER_MARKER_SIZE = 220
AXIS_LABEL_FONTSIZE = 15
AXIS_TICK_FONTSIZE = 13
STATION_LABEL_FONTSIZE = 13
VOLCANO_LABEL_FONTSIZE = 14
TITLE_FONTSIZE = 14
LEGEND_FONTSIZE = 11
COLORBAR_LABEL_FONTSIZE = 14
COLORBAR_TICK_FONTSIZE = 12
ASL_COLORMAP = "YlOrBr"
SARA_COLORMAP = "Greens"
COLORBAR_SHRINK = 0.72

def _plot_window(bundle):
    start = float(bundle.get("plot_window_start_s", 0.0))
    end = float(bundle.get("plot_window_end_s", np.inf))
    return start, end


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


def _resolve_colormap(cmap):
    if isinstance(cmap, str):
        return plt.get_cmap(cmap)
    return cmap


def _add_size_legend(ax, rms_min, rms_max, marker):
    levels = np.array([1e-6, 2e-6, 3e-6], dtype=float)
    sizes = _scale_marker_sizes(
        levels,
        SOURCE_MARKER_SIZE_MIN,
        SOURCE_MARKER_SIZE_MAX,
        rms_min,
        rms_max,
    )
    handles = [
        Line2D(
            [],
            [],
            linestyle="None",
            marker=marker,
            markersize=np.sqrt(size),
            markerfacecolor="0.6",
            markeredgecolor="black",
            markeredgewidth=0.6,
            label=f"{level * 1e6:.0f} μm/s",
        )
        for level, size in zip(levels, sizes)
    ]
    ax.legend(
        handles=handles,
        title="OTVZ RMS Amplitude",
        loc="lower right",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_FONTSIZE,
        frameon=True,
    )


def _add_reference_legend(ax):
    handles = [
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
    ax.legend(handles=handles, loc="upper right", fontsize=LEGEND_FONTSIZE, frameon=True)


def _add_station_labels(ax, bundle):
    outline = [pe.withStroke(linewidth=2.5, foreground="black")]
    for label, east, north in zip(
        np.asarray(bundle["station_names"]),
        np.asarray(bundle["station_easting"], dtype=float),
        np.asarray(bundle["station_northing"], dtype=float),
    ):
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


def _add_crater_label(ax, bundle):
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
        label="Ngauruhoe",
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


def _plot_locations(ax, bundle, title, min_quality, norm, cmap, source_sizes):
    X = np.asarray(bundle["map_X"], dtype=float)
    Y = np.asarray(bundle["map_Y"], dtype=float)
    C = np.asarray(bundle["map_C"], dtype=float)
    times = np.asarray(bundle["result_time"], dtype=float)
    quality = np.asarray(bundle["quality"], dtype=float)
    x = np.asarray(bundle["x_absolute"], dtype=float)
    y = np.asarray(bundle["y_absolute"], dtype=float)
    plot_start_s, plot_end_s = _plot_window(bundle)
    mask = (times >= plot_start_s) & (times <= plot_end_s)
    if min_quality is not None:
        mask &= quality >= min_quality
    display_time = times - plot_start_s

    ax.set_xlim(float(bundle["search_box_east"][0]), float(bundle["search_box_east"][1]))
    ax.set_ylim(float(bundle["search_box_north"][0]), float(bundle["search_box_north"][1]))

    if USE_SATELLITE_BASEMAP and ctx is not None:
        ctx.add_basemap(
            ax,
            source=BASEMAP_SOURCE,
            zoom="auto",
            crs="EPSG:2193",
        )

    if SHOW_CONTOURS or (USE_SATELLITE_BASEMAP and ctx is None):
        contour_color = "white" if USE_SATELLITE_BASEMAP else "0.5"
        contour_alpha = 0.45 if USE_SATELLITE_BASEMAP else 0.5
        ax.contour(X, Y, C, colors=contour_color, levels=18, linewidths=0.5, alpha=contour_alpha)

    ax.scatter(
        np.asarray(bundle["station_easting"], dtype=float),
        np.asarray(bundle["station_northing"], dtype=float),
        c="yellow",
        marker="s",
        s=STATION_MARKER_SIZE,
        edgecolor="black",
        linewidths=0.8,
        zorder=10,
        label="Stations",
    )
    _add_station_labels(ax, bundle)
    _add_crater_label(ax, bundle)

    if np.any(mask):
        marker = "o" if str(bundle.get("analysis_name", "")).upper() == "ASL" else "^"
        ax.scatter(
            x[mask],
            y[mask],
            c=display_time[mask],
            cmap=cmap,
            norm=norm,
            s=np.asarray(source_sizes, dtype=float)[mask],
            marker=marker,
            edgecolor="black",
            linewidth=0.4,
            zorder=9,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting (m)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Northing (m)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)
    ax.xaxis.offsetText.set_fontsize(AXIS_TICK_FONTSIZE)
    ax.yaxis.offsetText.set_fontsize(AXIS_TICK_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")


if __name__ == "__main__":
    asl = load_results_bundle(ASL_OUTPUT)
    sara = load_results_bundle(SARA_OUTPUT)

    asl_plot_start, asl_plot_end = _plot_window(asl)
    sara_plot_start, sara_plot_end = _plot_window(sara)
    time_min = 0.0
    time_max = max(asl_plot_end - asl_plot_start, sara_plot_end - sara_plot_start)
    norm = Normalize(vmin=time_min, vmax=time_max)
    asl_cmap = _resolve_colormap(ASL_COLORMAP)
    sara_cmap = _resolve_colormap(SARA_COLORMAP)

    asl_times = np.asarray(asl["result_time"], dtype=float)
    sara_times = np.asarray(sara["result_time"], dtype=float)
    asl_quality = np.asarray(asl["quality"], dtype=float)
    sara_quality = np.asarray(sara["quality"], dtype=float)
    asl_plot_mask = (asl_times >= asl_plot_start) & (asl_times <= asl_plot_end) & (asl_quality >= ASL_GOOD_THRESHOLD)
    sara_plot_mask = (sara_times >= sara_plot_start) & (sara_times <= sara_plot_end) & (sara_quality >= SARA_GOOD_THRESHOLD)
    asl_otvz_rms = _station_rms_at_times(asl, SIZE_SCALING_STATION, asl_times)
    sara_otvz_rms = _station_rms_at_times(sara, SIZE_SCALING_STATION, sara_times)
    combined_rms = np.concatenate([asl_otvz_rms[asl_plot_mask], sara_otvz_rms[sara_plot_mask]])
    if combined_rms.size == 0:
        rms_min = 0.0
        rms_max = 1.0
    else:
        rms_min = float(np.nanmin(combined_rms))
        rms_max = float(np.nanmax(combined_rms))
    asl_source_sizes = _scale_marker_sizes(
        asl_otvz_rms,
        SOURCE_MARKER_SIZE_MIN,
        SOURCE_MARKER_SIZE_MAX,
        rms_min,
        rms_max,
    )
    sara_source_sizes = _scale_marker_sizes(
        sara_otvz_rms,
        SOURCE_MARKER_SIZE_MIN,
        SOURCE_MARKER_SIZE_MAX,
        rms_min,
        rms_max,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5), constrained_layout=True)

    _plot_locations(axes[0], asl, "(a) ASL source locations", ASL_GOOD_THRESHOLD, norm, asl_cmap, asl_source_sizes)
    _plot_locations(axes[1], sara, "(b) SARA source locations", SARA_GOOD_THRESHOLD, norm, sara_cmap, sara_source_sizes)

    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    axes[1].yaxis.offsetText.set_visible(False)

    _add_reference_legend(axes[0])
    _add_size_legend(axes[0], rms_min, rms_max, marker="o")
    _add_reference_legend(axes[1])
    _add_size_legend(axes[1], rms_min, rms_max, marker="^")

    asl_sm = ScalarMappable(norm=norm, cmap=asl_cmap)
    sara_sm = ScalarMappable(norm=norm, cmap=sara_cmap)
    asl_cbar = fig.colorbar(
        asl_sm,
        ax=axes[0],
        shrink=COLORBAR_SHRINK,
        pad=0.03,
    )
    asl_cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)
    asl_cbar.set_label(
        f"ASL: Time (s) since {str(asl.get('plot_start_time', asl['start_time'])).replace('T', ' ')} UTC",
        fontsize=COLORBAR_LABEL_FONTSIZE,
    )
    sara_cbar = fig.colorbar(
        sara_sm,
        ax=axes[1],
        shrink=COLORBAR_SHRINK,
        pad=0.03,
    )
    sara_cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)
    sara_cbar.set_label(
        f"SARA: Time (s) since {str(sara.get('plot_start_time', sara['start_time'])).replace('T', ' ')} UTC",
        fontsize=COLORBAR_LABEL_FONTSIZE,
    )

    if SAVE_FIG is not None:
        SAVE_FIG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(SAVE_FIG, bbox_inches="tight")
        print(f"Saved figure → {SAVE_FIG}")

    plt.show()
