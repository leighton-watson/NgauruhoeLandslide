from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

try:
    import contextily as ctx
except ImportError:
    ctx = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT / "SourceLocalisation") not in sys.path:
    sys.path.append(str(ROOT / "SourceLocalisation"))

from source_localisation import (  # noqa: E402
    _extract_station_metadata,
    load_topography,
    seismoacoustic_differential_localization,
)


TOPOGRAPHY_DIR = ROOT / "Topography"
DATA_DIR = ROOT / "DATA"
PICK_DIR = ROOT / "AnalysisCodes"
STATION_COORDS_FILE = DATA_DIR / "seismic_20260321_133600.xml"
ARRIVAL_PICKS_FILE = PICK_DIR / "picked_arrivals.csv"
SUMMARY_CSV = HERE / "RESULTS" / "seismoacoustic_all_pick_combinations_summary.csv"
SAVE_FIG = HERE / "RESULTS" / "Fig_differential_arrivals_location.pdf"

CRATER_LAT = -39.1566302543244
CRATER_LON = 175.63253480007924

SEARCH_AREA = 5000
SEARCH_BOX = None
SPATIAL_STEP = 100

BASE_SEISMIC_VELOCITY = 3000.0
INFRASOUND_VELOCITY = 334.0
USE_SATELLITE_BASEMAP = True
BASEMAP_SOURCE = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
BASEMAP_ZOOM = "auto"
STATION_MARKER_SIZE = 100
CRATER_MARKER_SIZE = 220
AXIS_LABEL_FONTSIZE = 15
AXIS_TICK_FONTSIZE = 13
STATION_LABEL_FONTSIZE = 13
VOLCANO_LABEL_FONTSIZE = 14
TITLE_FONTSIZE = 14
LEGEND_FONTSIZE = 11
COLORBAR_LABEL_FONTSIZE = 14
COLORBAR_TICK_FONTSIZE = 12

SEISMIC_PICK_OPTIONS = {
    "pick_1": "seismic_pick_1_s",
    "pick_2": "seismic_pick_2_s",
    "early": "seismic_pick_1_s",
    "late": "seismic_pick_2_s",
}

INFRASOUND_PICK_OPTIONS = {
    "pick_1": "infrasound_pick_1_s",
    "pick_2": "infrasound_pick_2_s",
    "early": "infrasound_pick_1_s",
    "late": "infrasound_pick_2_s",
}


def load_arrival_picks(path: Path) -> pd.DataFrame:
    picks = pd.read_csv(path)
    required = {
        "station",
        "seismic_arrival_s",
        "seismic_pick_1_s",
        "seismic_pick_2_s",
        "infrasound_arrival_s",
        "infrasound_pick_1_s",
        "infrasound_pick_2_s",
    }
    missing = required - set(picks.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    if "use" in picks.columns:
        keep = picks["use"].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})
        picks = picks.loc[keep].copy()

    numeric_columns = [
        "seismic_arrival_s",
        "seismic_pick_1_s",
        "seismic_pick_2_s",
        "infrasound_arrival_s",
        "infrasound_pick_1_s",
        "infrasound_pick_2_s",
        "uncertainty_s",
    ]
    for column in numeric_columns:
        if column in picks.columns:
            picks[column] = pd.to_numeric(picks[column], errors="coerce")

    picks = picks.dropna(subset=["seismic_arrival_s"]).copy()
    if picks.empty:
        raise ValueError("No usable seismic picks found.")
    return picks.sort_values("station").reset_index(drop=True)


def build_pick_table(
    base_picks: pd.DataFrame,
    stations: list[str],
    seismic_option_name: str,
    infrasound_option_name: str,
) -> pd.DataFrame:
    scenario = base_picks.copy()
    seismic_column = SEISMIC_PICK_OPTIONS[seismic_option_name]
    infrasound_column = INFRASOUND_PICK_OPTIONS[infrasound_option_name]
    for station in stations:
        mask = scenario["station"] == station
        scenario.loc[mask, "seismic_arrival_s"] = scenario.loc[mask, seismic_column]
        scenario.loc[mask, "infrasound_arrival_s"] = scenario.loc[mask, infrasound_column]

    scenario = scenario.dropna(subset=["seismic_arrival_s", "infrasound_arrival_s"]).copy()
    if scenario.empty:
        raise ValueError("Base scenario produced no usable picks.")
    return scenario


def run_location(arrival_picks: pd.DataFrame, topo, seismic_velocity: float):
    station_coords = _extract_station_metadata(arrival_picks["station"].tolist(), STATION_COORDS_FILE)
    return seismoacoustic_differential_localization(
        arrival_picks=arrival_picks,
        station_coords=station_coords,
        topo=topo,
        crater_lat=CRATER_LAT,
        crater_lon=CRATER_LON,
        search_area=SEARCH_AREA,
        search_box=SEARCH_BOX,
        spatial_step=SPATIAL_STEP,
        seismic_velocity=seismic_velocity,
        infrasound_velocity=INFRASOUND_VELOCITY,
    )


def build_best_scenario_pick_table(base_picks: pd.DataFrame, best_row: pd.Series) -> tuple[pd.DataFrame, float]:
    stations = base_picks["station"].tolist()
    station_options: dict[str, tuple[str, str]] = {}
    for part in str(best_row["combination_name"]).split("|"):
        part = part.strip()
        if not part:
            continue
        station, option_text = part.split(":", maxsplit=1)
        seismic_text, infrasound_text = option_text.split("/", maxsplit=1)
        station_options[station.strip()] = (
            seismic_text.removeprefix("seis_"),
            infrasound_text.removeprefix("inf_"),
        )

    scenario_picks = base_picks.copy()
    for station in stations:
        seismic_option, infrasound_option = station_options[station]
        mask = scenario_picks["station"] == station
        scenario_picks.loc[mask, "seismic_arrival_s"] = scenario_picks.loc[mask, SEISMIC_PICK_OPTIONS[seismic_option]]
        scenario_picks.loc[mask, "infrasound_arrival_s"] = scenario_picks.loc[mask, INFRASOUND_PICK_OPTIONS[infrasound_option]]

    scenario_picks = scenario_picks.dropna(subset=["seismic_arrival_s", "infrasound_arrival_s"]).copy()
    return scenario_picks, float(best_row["seismic_velocity_mps"])


def _grid_edges(values: np.ndarray) -> np.ndarray:
    if values.size < 2:
        raise ValueError("At least two grid values are required to build bin edges.")
    ordered = np.asarray(values, dtype=float)
    if ordered[0] > ordered[-1]:
        ordered = ordered[::-1]
    delta = np.diff(ordered)
    first = ordered[0] - delta[0] / 2.0
    mids = ordered[:-1] + delta / 2.0
    last = ordered[-1] + delta[-1] / 2.0
    return np.concatenate(([first], mids, [last]))


def build_density_grid(summary_df: pd.DataFrame, search_grid):
    x_centres = search_grid.x_grid[0, :]
    y_centres = search_grid.y_grid[:, 0]
    x_edges = _grid_edges(x_centres)
    y_edges = _grid_edges(y_centres)

    counts, _, _ = np.histogram2d(
        summary_df["best_y_m"].to_numpy(float),
        summary_df["best_x_m"].to_numpy(float),
        bins=[y_edges, x_edges],
    )
    return counts, x_edges, y_edges


def _minimum_enclosing_circle(points: np.ndarray) -> tuple[float, float, float]:
    if points.size == 0:
        raise ValueError("No source locations available for coverage-circle calculation.")

    points_list = [tuple(row) for row in np.asarray(points, dtype=float)]
    rng = np.random.default_rng(0)
    shuffled = [points_list[idx] for idx in rng.permutation(len(points_list))]

    def _distance(a, b) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _inside(point, circle, eps: float = 1e-9) -> bool:
        return _distance(point, circle[:2]) <= circle[2] + eps

    def _circle_from_two(a, b):
        cx = (a[0] + b[0]) / 2.0
        cy = (a[1] + b[1]) / 2.0
        return (cx, cy, _distance(a, b) / 2.0)

    def _circle_from_three(a, b, c):
        ax, ay = a
        bx, by = b
        cx, cy = c
        d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-12:
            return None

        ux = (
            (ax * ax + ay * ay) * (by - cy)
            + (bx * bx + by * by) * (cy - ay)
            + (cx * cx + cy * cy) * (ay - by)
        ) / d
        uy = (
            (ax * ax + ay * ay) * (cx - bx)
            + (bx * bx + by * by) * (ax - cx)
            + (cx * cx + cy * cy) * (bx - ax)
        ) / d
        return (ux, uy, _distance((ux, uy), a))

    circle = None
    for i, p in enumerate(shuffled):
        if circle is not None and _inside(p, circle):
            continue
        circle = (p[0], p[1], 0.0)
        for j in range(i):
            q = shuffled[j]
            if _inside(q, circle):
                continue
            circle = _circle_from_two(p, q)
            for k in range(j):
                r = shuffled[k]
                if _inside(r, circle):
                    continue
                candidate = _circle_from_three(p, q, r)
                if candidate is None:
                    pairs = (_circle_from_two(p, q), _circle_from_two(p, r), _circle_from_two(q, r))
                    circle = max(
                        (
                            pair
                            for pair in pairs
                            if _inside(p, pair) and _inside(q, pair) and _inside(r, pair)
                        ),
                        key=lambda item: item[2],
                    )
                else:
                    circle = candidate

    if circle is None:
        raise ValueError("Could not determine the source-location coverage circle.")
    return circle


def build_coverage_circle(summary_df: pd.DataFrame, coverage: float = 0.95) -> tuple[float, float, float]:
    if not 0.0 < coverage <= 1.0:
        raise ValueError("Coverage must be between 0 and 1.")

    points = summary_df[["best_x_m", "best_y_m"]].dropna().to_numpy(float)
    center_x, center_y, _ = _minimum_enclosing_circle(points)
    distances = np.hypot(points[:, 0] - center_x, points[:, 1] - center_y)
    radius = float(np.quantile(distances, coverage, method="higher"))
    return center_x, center_y, radius


def _add_station_labels(ax, station_results: pd.DataFrame):
    outline = [pe.withStroke(linewidth=2.5, foreground="black")]
    for _, row in station_results.iterrows():
        label = str(row["station"])
        if label == "WTVZ":
            xytext = (0, -10)
            va = "top"
        else:
            xytext = (0, 8)
            va = "bottom"
        ax.annotate(
            label,
            xy=(row["easting"], row["northing"]),
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


def _add_crater_marker_and_label(ax, xcrater: float, ycrater: float):
    ax.scatter(
        xcrater,
        ycrater,
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
        xy=(xcrater, ycrater),
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


def _add_station_markers(ax, station_results: pd.DataFrame):
    ax.scatter(
        station_results["easting"],
        station_results["northing"],
        c="yellow",
        marker="s",
        s=STATION_MARKER_SIZE,
        edgecolor="black",
        linewidths=0.8,
        zorder=10,
        label="Stations",
    )
    _add_station_labels(ax, station_results)


def plot_density_summary(best_result, summary_df: pd.DataFrame, topo, save_path: Path):
    print("Preparing density grid...")
    search_grid = best_result["search_grid"]
    xcrater, ycrater = best_result["crater_xy"]
    counts, x_edges, y_edges = build_density_grid(summary_df, search_grid)
    coverage_center_x, coverage_center_y, coverage_radius = build_coverage_circle(summary_df, coverage=0.95)
    x_min = float(search_grid.x_grid.min())
    x_max = float(search_grid.x_grid.max())
    y_min = float(search_grid.y_grid.min())
    y_max = float(search_grid.y_grid.max())

    print("Creating matplotlib figure...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True, sharex=True, sharey=True)
    ax_left, ax_right = axes

    print("Drawing left-panel RMS map...")
    ax_left.contour(topo["X"], topo["Y"], topo["C"], levels=36, colors="k", linewidths=0.45, alpha=0.3)
    left_im = ax_left.pcolormesh(
        search_grid.x_grid,
        search_grid.y_grid,
        best_result["rms_grid"],
        shading="auto",
        cmap="viridis_r",
    )
    left_cbar = fig.colorbar(left_im, ax=ax_left, pad=0.02)
    left_cbar.set_label("RMS residual (s)", fontsize=COLORBAR_LABEL_FONTSIZE)
    left_cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    station_results = best_result["station_results"].sort_values("station").reset_index(drop=True)
    _add_station_markers(ax_left, station_results)
    _add_crater_marker_and_label(ax_left, xcrater, ycrater)
    ax_left.scatter(
        best_result["best_x"],
        best_result["best_y"],
        marker="*",
        s=180,
        c="gold",
        edgecolor="black",
        linewidths=0.8,
        zorder=11,
        label="Best source location",
    )
    ax_left.set_title("(a) Best fitting source localization", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax_left.set_xlabel("Easting (m)", fontsize=AXIS_LABEL_FONTSIZE)
    ax_left.set_ylabel("Northing (m)", fontsize=AXIS_LABEL_FONTSIZE)
    ax_left.set_aspect("equal", adjustable="box")
    ax_left.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)
    ax_left.xaxis.offsetText.set_fontsize(AXIS_TICK_FONTSIZE)
    ax_left.yaxis.offsetText.set_fontsize(AXIS_TICK_FONTSIZE)
    ax_left.legend(loc="upper right", fontsize=LEGEND_FONTSIZE)
    ax_left.set_xlim(x_min, x_max)
    ax_left.set_ylim(y_min, y_max)

    ax_right.set_xlim(x_min, x_max)
    ax_right.set_ylim(y_min, y_max)

    if USE_SATELLITE_BASEMAP and ctx is not None:
        print(f"Requesting satellite basemap tiles at zoom {BASEMAP_ZOOM}...")
        ctx.add_basemap(
            ax_right,
            source=BASEMAP_SOURCE,
            zoom=BASEMAP_ZOOM,
            crs="EPSG:2193",
        )
    else:
        print("Skipping satellite basemap (contextily unavailable or disabled).")

    station_results = best_result["station_results"].sort_values("station").reset_index(drop=True)
    print("Drawing right-panel station markers and density layer...")
    _add_station_markers(ax_right, station_results)

    masked_counts = np.ma.masked_less_equal(counts, 0.0)
    density_cmap = plt.get_cmap("Greys").copy()
    density_cmap.set_bad(alpha=0.0)
    right_im = ax_right.pcolormesh(
        x_edges,
        y_edges,
        masked_counts,
        shading="flat",
        cmap=density_cmap,
        alpha=0.85,
        zorder=9,
    )
    right_cbar = fig.colorbar(right_im, ax=ax_right, pad=0.02)
    right_cbar.set_label("Number of best source locations", fontsize=COLORBAR_LABEL_FONTSIZE)
    right_cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)
    print(
        "95% source-location circle radius "
        f"= {coverage_radius:.3f} m ({coverage_radius / 1000.0:.6f} km)"
    )
    _add_crater_marker_and_label(ax_right, xcrater, ycrater)
    ax_right.add_patch(
        Circle(
            (coverage_center_x, coverage_center_y),
            coverage_radius,
            fill=False,
            edgecolor="cyan",
            linewidth=2.0,
            linestyle="--",
            zorder=12,
            label="95% of source-locations",
        )
    )
    ax_right.set_title("(b) All pick and velocity combinations", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax_right.set_xlabel("Easting (m)", fontsize=AXIS_LABEL_FONTSIZE)
    ax_right.set_aspect("equal", adjustable="box")
    ax_right.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)
    ax_right.xaxis.offsetText.set_fontsize(AXIS_TICK_FONTSIZE)
    ax_right.yaxis.offsetText.set_fontsize(AXIS_TICK_FONTSIZE)
    ax_right.legend(loc="upper right", fontsize=LEGEND_FONTSIZE)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving figure to {save_path}...")
    fig.savefig(save_path, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    print("Loading topography...")
    topo = load_topography(TOPOGRAPHY_DIR)
    print("Loading picked arrivals...")
    base_picks = load_arrival_picks(ARRIVAL_PICKS_FILE)
    print("Loading combination summary table...")
    summary_df = pd.read_csv(SUMMARY_CSV)
    print("Selecting the best-fitting pick and velocity combination...")
    best_row = summary_df.sort_values("best_rms_s", kind="stable").iloc[0]
    best_picks, best_seismic_velocity = build_best_scenario_pick_table(base_picks, best_row)
    print("Running localisation for the best-fitting combination...")
    best_result = run_location(best_picks, topo, best_seismic_velocity)

    print("Building density figure...")
    fig = plot_density_summary(best_result, summary_df, topo, SAVE_FIG)
    print("Figure rendered and saved.")
    print(f"Saved figure -> {SAVE_FIG}")
    print("Opening interactive figure window...")
    plt.show()
