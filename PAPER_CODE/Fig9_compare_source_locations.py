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
from matplotlib.patches import FancyBboxPatch, Rectangle

try:
    import contextily as ctx
except ImportError:
    ctx = None


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from AnalysisCodes.io_utils import load_results_bundle
from AnalysisCodes.source_localisation import (
    _extract_station_metadata,
    load_topography,
    seismoacoustic_differential_localization,
)


ASL_OUTPUT = HERE / "Outputs" /  "ASL_output.npz"
SARA_OUTPUT = HERE / "Outputs" / "SARA_output.npz"
TOPOGRAPHY_DIR = HERE / "DATA"
DATA_DIR = HERE / "DATA"
PICK_DIR = HERE 
STATION_COORDS_FILE = DATA_DIR / "seismic_20260321_133600.xml"
ARRIVAL_PICKS_FILE = PICK_DIR / "OnsetTimes" / "offline_change_point_arrival_picks.csv"
SAVE_FIG = HERE / "FIGURES" / "Figure9.pdf"

CRATER_LAT = -39.1566302543244
CRATER_LON = 175.63253480007924
DIFF_SEARCH_AREA = 1000
DIFF_SEARCH_BOX = None
DIFF_SPATIAL_STEP = 10
DIFF_SEISMIC_VELOCITY_MPS = 3400
DIFF_INFRASOUND_VELOCITY_MPS = 334
ASL_GOOD_THRESHOLD = 0.742
SARA_GOOD_THRESHOLD = 0.912
TRY_SPYDER_INTERACTIVE_BACKEND = True
USE_SATELLITE_BASEMAP = True
BASEMAP_SOURCE = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
STATION_MARKER_SIZE = 200
CRATER_MARKER_SIZE = 600
SOURCE_MARKER_SIZE_MIN = 140
SOURCE_MARKER_SIZE_MAX = 760
ASL_MARKER_ALPHA = 0.7
SARA_MARKER_ALPHA = 0.7
DIFF_SOURCE_MARKER_SIZE = 650
AXIS_LABEL_FONTSIZE = 18
AXIS_TICK_FONTSIZE = 16
STATION_LABEL_FONTSIZE = 18
VOLCANO_LABEL_FONTSIZE = 18
TITLE_FONTSIZE = 18
LEGEND_FONTSIZE = 16
SCALEBAR_FONTSIZE = 16
COLORBAR_LABEL_FONTSIZE = 16
COLORBAR_TICK_FONTSIZE = 16
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
        label="Stations",
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
        loc="lower right",
        handler_map={tuple: HandlerTuple(ndivide=None, pad=1.0)},
        handletextpad=0.8,
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
        Line2D(
            [],
            [],
            linestyle="None",
            marker="*",
            markersize=np.sqrt(DIFF_SOURCE_MARKER_SIZE),
            markerfacecolor="yellow",
            markeredgecolor="black",
            markeredgewidth=0.8,
            label="Differential arrivals",
        ),
    ]
    legend = ax.legend(handles=handles, loc="upper right", fontsize=LEGEND_FONTSIZE, frameon=True)
    ax.add_artist(legend)


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


def _resolve_colormap(cmap):
    if isinstance(cmap, str):
        return plt.get_cmap(cmap)
    return cmap


def _load_arrival_picks(path: Path) -> pd.DataFrame:
    picks = pd.read_csv(path)
    required = {"station", "kind", "pick_s"}
    missing = required - set(picks.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    picks = picks.copy()
    picks["kind"] = picks["kind"].astype(str).str.strip().str.lower()
    picks["pick_s"] = pd.to_numeric(picks["pick_s"], errors="coerce")

    if "status" in picks.columns:
        picks = picks.loc[picks["status"].astype(str).str.strip().str.lower() == "ok"].copy()

    picks = picks.dropna(subset=["pick_s"]).copy()

    rows = []
    for station, station_rows in picks.groupby("station", sort=True):
        seismic_rows = station_rows.loc[station_rows["kind"] == "seismic"]
        infrasound_rows = station_rows.loc[station_rows["kind"] == "infrasound"]
        if seismic_rows.empty or infrasound_rows.empty:
            continue

        rows.append({
            "station": station,
            "seismic_arrival_s": float(seismic_rows.iloc[0]["pick_s"]),
            "infrasound_arrival_s": float(infrasound_rows.iloc[0]["pick_s"]),
            "uncertainty_s": 1.0,
        })

    arrival_picks = pd.DataFrame(rows)
    if arrival_picks.empty:
        raise ValueError("No stations have both seismic and infrasound picks.")

    return arrival_picks.sort_values("station").reset_index(drop=True)


def _best_differential_location():
    topo = load_topography(TOPOGRAPHY_DIR)
    arrival_picks = _load_arrival_picks(ARRIVAL_PICKS_FILE)
    station_coords = _extract_station_metadata(arrival_picks["station"].tolist(), STATION_COORDS_FILE)
    result = seismoacoustic_differential_localization(
        arrival_picks=arrival_picks,
        station_coords=station_coords,
        topo=topo,
        crater_lat=CRATER_LAT,
        crater_lon=CRATER_LON,
        search_area=DIFF_SEARCH_AREA,
        search_box=DIFF_SEARCH_BOX,
        spatial_step=DIFF_SPATIAL_STEP,
        seismic_velocity=DIFF_SEISMIC_VELOCITY_MPS,
        infrasound_velocity=DIFF_INFRASOUND_VELOCITY_MPS,
    )
    return {
        "x": result["best_x"],
        "y": result["best_y"],
        "rms": result["best_rms_s"],
    }


def _add_scale_bar(ax, total_length_m=2000.0, segment_length_m=500.0, tick_step_m=500.0):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_range = x_max - x_min
    y_range = y_max - y_min

    bar_height = 0.018 * y_range
    margin_x = 0.05 * x_range
    margin_y = 0.055 * y_range

    x0 = x_min + margin_x
    y0 = y_max - margin_y - bar_height

    segment_colors = ["black", "white"]
    label_y = y0 - 0.012 * y_range
    box_pad_x = 0.028 * x_range
    box_pad_y = 0.030 * y_range
    ax.add_patch(
        FancyBboxPatch(
            (x0 - box_pad_x, label_y - box_pad_y),
            total_length_m + 2.0 * box_pad_x,
            y0 + bar_height - label_y + 2.0 * box_pad_y,
            boxstyle="round,pad=0.0,rounding_size=0.01",
            facecolor="white",
            edgecolor="0.8",
            linewidth=0.8,
            alpha=0.9,
            zorder=15,
        )
    )

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
            label_y,
            label,
            ha="center",
            va="top",
            fontsize=SCALEBAR_FONTSIZE,
            color="black",
            zorder=17,
        )


if __name__ == "__main__":
    asl = load_results_bundle(ASL_OUTPUT)
    sara = load_results_bundle(SARA_OUTPUT)

    asl_loc = _bundle_locations(asl, ASL_GOOD_THRESHOLD)
    sara_loc = _bundle_locations(sara, SARA_GOOD_THRESHOLD)
    diff_best = _best_differential_location()

    time_max = max(float(asl_loc["duration"]), float(sara_loc["duration"]))
    time_norm = Normalize(vmin=0.0, vmax=time_max if time_max > 0 else 1.0)
    asl_cmap = _resolve_colormap(ASL_COLORMAP)
    sara_cmap = _resolve_colormap(SARA_COLORMAP)

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

    fig, ax = plt.subplots(figsize=(15.2, 11.4), constrained_layout=True)

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
        alpha=ASL_MARKER_ALPHA,
        zorder=8,
        label="ASL",
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
        alpha=SARA_MARKER_ALPHA,
        zorder=9,
        label="SARA",
    )
    ax.scatter(
        diff_best["x"],
        diff_best["y"],
        marker="*",
        s=DIFF_SOURCE_MARKER_SIZE,
        c="yellow",
        edgecolor="black",
        linewidth=0.9,
        alpha=0.95,
        zorder=11,
        label="Differential arrivals",
    )

    _add_reference_markers(ax, asl)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting (m)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Northing (m)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)
    ax.xaxis.offsetText.set_fontsize(AXIS_TICK_FONTSIZE)
    ax.yaxis.offsetText.set_fontsize(AXIS_TICK_FONTSIZE)

    _add_reference_legend(ax)
    _add_size_legend(ax, rms_min, rms_max)
    _add_scale_bar(ax)

    asl_sm = ScalarMappable(norm=time_norm, cmap=asl_cmap)
    sara_sm = ScalarMappable(norm=time_norm, cmap=sara_cmap)
    _colorbar(fig, sara_sm, ax, "SARA: Time (s) since 2026-03-21 13:36:05 (UTC)", pad=0.04)
    _colorbar(fig, asl_sm, ax, "ASL: Time (s) since 2026-03-21 13:36:05 (UTC)", pad=0.01)

    SAVE_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_FIG, bbox_inches="tight")
    print(f"Saved figure -> {SAVE_FIG}")
    plt.show()
