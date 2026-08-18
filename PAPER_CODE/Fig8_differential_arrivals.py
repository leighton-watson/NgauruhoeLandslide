from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT / "AnalysisCodes") not in sys.path:
    sys.path.append(str(ROOT / "AnalysisCodes"))

from source_localisation import (  # noqa: E402
    _extract_station_metadata,
    load_topography,
    seismoacoustic_differential_localization,
)


TOPOGRAPHY_DIR = HERE / "DATA"
DATA_DIR = ROOT / "DATA"
PICK_DIR = HERE
STATION_COORDS_FILE = DATA_DIR / "seismic_20260321_133600.xml"
ARRIVAL_PICKS_FILE = PICK_DIR / "OnsetTimes" / "offline_change_point_arrival_picks.csv"
SAVE_FIG = HERE / "FIGURES" / "Figure8.pdf"

CRATER_LAT = -39.1566302543244
CRATER_LON = 175.63253480007924

SEARCH_AREA = 5000
SEARCH_BOX = None
SPATIAL_STEP = 100

SEISMIC_VELOCITIES_MPS = [3400]
INFRASOUND_VELOCITY_MPS = 334.0

MAP_VERTICAL_FRACTION = 0.5
STATION_MARKER_SIZE = 100
CRATER_MARKER_SIZE = 220
SOURCE_MARKER_SIZE = 190
AXIS_LABEL_FONTSIZE = 15
AXIS_TICK_FONTSIZE = 13
STATION_LABEL_FONTSIZE = 13
VOLCANO_LABEL_FONTSIZE = 14
TITLE_FONTSIZE = 15
LEGEND_FONTSIZE = 11
COLORBAR_LABEL_FONTSIZE = 14
COLORBAR_TICK_FONTSIZE = 12


def load_arrival_picks(path: Path) -> pd.DataFrame:
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


def run_location(arrival_picks: pd.DataFrame, topo, seismic_velocity_mps: float):
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
        seismic_velocity=seismic_velocity_mps,
        infrasound_velocity=INFRASOUND_VELOCITY_MPS,
    )


def _add_station_labels(ax, station_results: pd.DataFrame):
    outline = [pe.withStroke(linewidth=2.5, foreground="black")]
    for _, row in station_results.iterrows():
        label = str(row["station"])
        xytext = (0, -10) if label == "WTVZ" else (0, 8)
        va = "top" if label == "WTVZ" else "bottom"
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


def plot_result_panel(
    ax,
    result,
    topo,
    seismic_velocity_mps: float,
    panel_label: str,
    residual_vmin: float,
    residual_vmax: float,
):
    search_grid = result["search_grid"]
    xcrater, ycrater = result["crater_xy"]

    x_min = float(search_grid.x_grid.min())
    x_max = float(search_grid.x_grid.max())
    y_min = float(search_grid.y_grid.min())
    y_max = float(search_grid.y_grid.max())
    y_mid = (y_min + y_max) / 2.0
    y_half_span = (y_max - y_min) * MAP_VERTICAL_FRACTION / 2.0
    plot_y_min = y_mid - y_half_span
    plot_y_max = y_mid + y_half_span

    ax.contour(topo["X"], topo["Y"], topo["C"], levels=36, colors="k", linewidths=0.45, alpha=0.3)

    im = ax.pcolormesh(
        search_grid.x_grid,
        search_grid.y_grid,
        result["rms_grid"],
        shading="auto",
        cmap="viridis_r",
        vmin=residual_vmin,
        vmax=residual_vmax,
    )

    station_results = result["station_results"].sort_values("station").reset_index(drop=True)
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

    ax.scatter(
        result["best_x"],
        result["best_y"],
        marker="*",
        s=SOURCE_MARKER_SIZE,
        c="yellow",
        edgecolor="black",
        linewidths=0.8,
        zorder=16,
        label="Best source location",
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(plot_y_min, plot_y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting (m)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Northing (m)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)
    ax.xaxis.offsetText.set_fontsize(AXIS_TICK_FONTSIZE)
    ax.yaxis.offsetText.set_fontsize(AXIS_TICK_FONTSIZE)
    # ax.set_title(
    #     f"{panel_label} Vs = {seismic_velocity_mps / 1000.0:.1f} km/s",
    #     fontsize=TITLE_FONTSIZE,
    #     fontweight="bold",
    # )
    ax.legend(loc="upper right", fontsize=LEGEND_FONTSIZE)
    return im


def plot_results(results_by_velocity: dict[float, dict], topo, save_path: Path):
    residual_vmin = min(
        float(np.nanmin(result["rms_grid"]))
        for result in results_by_velocity.values()
    )
    residual_vmax = max(
        float(np.nanmax(result["rms_grid"]))
        for result in results_by_velocity.values()
    )

    fig = plt.figure(figsize=(8.2 * len(results_by_velocity) + 0.8, 7.5))
    grid_spec = fig.add_gridspec(
        1,
        len(results_by_velocity) + 1,
        width_ratios=[1.0] * len(results_by_velocity) + [0.04],
        wspace=0.22,
    )
    axes = np.array([
        fig.add_subplot(grid_spec[0, idx])
        for idx in range(len(results_by_velocity))
    ])
    cax = fig.add_subplot(grid_spec[0, -1])

    panel_labels = ["(a)", "(b)", "(c)"]
    im = None
    for ax, (velocity_mps, result), panel_label in zip(
        axes,
        results_by_velocity.items(),
        panel_labels,
    ):
        im = plot_result_panel(
            ax,
            result,
            topo,
            velocity_mps,
            panel_label,
            residual_vmin,
            residual_vmax,
        )

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("RMS residual (s)", fontsize=COLORBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    return fig


def print_best_location_distances(results_by_velocity: dict[float, dict]) -> None:
    print("Best-fitting source locations:")
    for velocity_mps, result in results_by_velocity.items():
        print(
            f"  Vs={velocity_mps / 1000.0:.1f} km/s: "
            f"x={result['best_x']:.1f} m, y={result['best_y']:.1f} m, "
            f"RMS={result['best_rms_s']:.2f} s"
        )

    print("Distances between best-fitting source locations:")
    velocities = list(results_by_velocity)
    for idx, velocity_a in enumerate(velocities):
        result_a = results_by_velocity[velocity_a]
        for velocity_b in velocities[idx + 1:]:
            result_b = results_by_velocity[velocity_b]
            distance_m = float(np.hypot(
                result_a["best_x"] - result_b["best_x"],
                result_a["best_y"] - result_b["best_y"],
            ))
            print(
                f"  {velocity_a / 1000.0:.1f} vs {velocity_b / 1000.0:.1f} km/s: "
                f"{distance_m / 1000.0:.3f} km"
            )


if __name__ == "__main__":
    print("Loading topography...")
    topo = load_topography(TOPOGRAPHY_DIR)
    print("Loading preferred change-point picks...")
    arrival_picks = load_arrival_picks(ARRIVAL_PICKS_FILE)
    results_by_velocity = {}
    for seismic_velocity_mps in SEISMIC_VELOCITIES_MPS:
        print(
            "Running differential-arrival localization "
            f"(Vs={seismic_velocity_mps / 1000.0:.1f} km/s, "
            f"Vi={INFRASOUND_VELOCITY_MPS:.0f} m/s)..."
        )
        results_by_velocity[seismic_velocity_mps] = run_location(
            arrival_picks,
            topo,
            seismic_velocity_mps,
        )

    print("Building figure...")
    fig = plot_results(results_by_velocity, topo, SAVE_FIG)
    print(f"Saved figure -> {SAVE_FIG}")
    print_best_location_distances(results_by_velocity)
    plt.show()
