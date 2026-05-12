from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT / "SourceLocalisation") not in sys.path:
    sys.path.append(str(ROOT / "SourceLocalisation"))

from source_localisation import (  # noqa: E402
    _extract_station_metadata,
    load_topography,
    seismoacoustic_differential_localization,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

DATA_DIR = ROOT / "DATA"
PICK_DIR = ROOT / "AnalysisCodes"
TOPOGRAPHY_DIR = ROOT / "Topography"
STATION_COORDS_FILE = DATA_DIR / "seismic_20260321_133600.xml"
ARRIVAL_PICKS_FILE = PICK_DIR / "picked_arrivals.csv"

CRATER_LAT = -39.1566302543244
CRATER_LON = 175.63253480007924

SEARCH_AREA = 5000
SEARCH_BOX = None
SPATIAL_STEP = 100

SEISMIC_VELOCITIES = (2500.0, 3000.0, 3300.0)
BASE_SEISMIC_VELOCITY = 3000.0
INFRASOUND_VELOCITY = 334.0

OUTPUT_DIR = HERE / "RESULTS"
OUTPUT_DIR.mkdir(exist_ok=True)

SEISMIC_PICK_OPTIONS = {
    "pick_1": "seismic_pick_1_s",
    "pick_2": "seismic_pick_2_s",
}

INFRASOUND_PICK_OPTIONS = {
    "pick_1": "infrasound_pick_1_s",
    "pick_2": "infrasound_pick_2_s",
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

    picks = picks.dropna(subset=["seismic_arrival_s", "infrasound_arrival_s"]).copy()
    if picks.empty:
        raise ValueError("No usable midpoint seismic/infrasound picks found.")
    return picks.sort_values("station").reset_index(drop=True)


def build_combination_pick_table(
    base_picks: pd.DataFrame,
    stations: list[str],
    option_names: tuple[tuple[str, str], ...],
) -> pd.DataFrame:
    scenario = base_picks.copy()
    for station, (seismic_option, infrasound_option) in zip(stations, option_names):
        seismic_column = SEISMIC_PICK_OPTIONS[seismic_option]
        infrasound_column = INFRASOUND_PICK_OPTIONS[infrasound_option]
        mask = scenario["station"] == station
        scenario.loc[mask, "seismic_arrival_s"] = scenario.loc[mask, seismic_column]
        scenario.loc[mask, "infrasound_arrival_s"] = scenario.loc[mask, infrasound_column]

    scenario = scenario.dropna(subset=["seismic_arrival_s", "infrasound_arrival_s"]).copy()
    if scenario.empty:
        raise ValueError("Combination produced no usable picks.")
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


def _progress_iter(total: int):
    if tqdm is not None:
        return tqdm(total=total, desc="Running pick combinations", unit="combo")
    return None


def _combo_name(stations: list[str], option_names: tuple[tuple[str, str], ...]) -> str:
    parts = [
        f"{station}:seis_{seismic_option}/inf_{infrasound_option}"
        for station, (seismic_option, infrasound_option) in zip(stations, option_names)
    ]
    return " | ".join(parts)


def plot_combination_summary(base_result, combination_results: pd.DataFrame, topo, save_path: Path):
    search_grid = base_result["search_grid"]
    xcrater, ycrater = base_result["crater_xy"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True, sharex=True, sharey=True)
    ax_map = axes[0]
    ax_combo = axes[1]

    ax_map.contour(topo["X"], topo["Y"], topo["C"], levels=18, colors="k", linewidths=0.45, alpha=0.3)
    im = ax_map.pcolormesh(
        search_grid.x_grid,
        search_grid.y_grid,
        base_result["rms_grid"],
        shading="auto",
        cmap="viridis_r",
    )
    cbar = fig.colorbar(im, ax=ax_map, pad=0.02)
    cbar.set_label("RMS residual (s)")

    station_results = base_result["station_results"].sort_values("station").reset_index(drop=True)
    ax_map.scatter(
        station_results["easting"],
        station_results["northing"],
        marker="v",
        s=55,
        c="tab:red",
        edgecolor="black",
        linewidths=0.5,
        zorder=5,
        label="Stations",
    )
    for _, row in station_results.iterrows():
        ax_map.text(row["easting"] + 35, row["northing"] + 35, row["station"], fontsize=8)

    ax_map.scatter(
        xcrater,
        ycrater,
        marker="X",
        s=110,
        c="tab:blue",
        edgecolor="black",
        linewidths=0.7,
        zorder=6,
        label="Crater",
    )
    ax_map.scatter(
        base_result["best_x"],
        base_result["best_y"],
        marker="*",
        s=180,
        c="gold",
        edgecolor="black",
        linewidths=0.8,
        zorder=7,
        label="Base scenario source",
    )
    ax_map.set_title(
        f"Midpoint picks, Vp=3.0 km/s\nRMS={base_result['best_rms_s']:.2f} s, lag={base_result['best_source_lag_s']:.2f} s"
    )
    ax_map.set_xlabel("Easting (m)")
    ax_map.set_ylabel("Northing (m)")
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.legend(loc="upper right", fontsize=8)

    ax_combo.contour(topo["X"], topo["Y"], topo["C"], levels=18, colors="k", linewidths=0.45, alpha=0.3)
    scatter = ax_combo.scatter(
        combination_results["best_x_m"],
        combination_results["best_y_m"],
        c=combination_results["best_rms_s"],
        cmap="plasma_r",
        s=48,
        edgecolor="black",
        linewidths=0.35,
        alpha=0.9,
        zorder=5,
    )
    ax_combo.scatter(
        xcrater,
        ycrater,
        marker="X",
        s=110,
        c="tab:blue",
        edgecolor="black",
        linewidths=0.7,
        zorder=6,
        label="Crater",
    )
    ax_combo.scatter(
        base_result["best_x"],
        base_result["best_y"],
        marker="*",
        s=180,
        c="gold",
        edgecolor="black",
        linewidths=0.8,
        zorder=7,
        label="Base scenario source",
    )
    combo_cbar = fig.colorbar(scatter, ax=ax_combo, pad=0.02)
    combo_cbar.set_label("Combination RMS residual (s)")
    ax_combo.set_title(f"All seismic/infrasound pick combinations\n{len(combination_results)} source locations")
    ax_combo.set_xlabel("Easting (m)")
    ax_combo.set_aspect("equal", adjustable="box")
    ax_combo.legend(loc="upper right", fontsize=8)

    for ax in axes:
        ax.set_xlim(search_grid.x_grid.min(), search_grid.x_grid.max())
        ax.set_ylim(search_grid.y_grid.min(), search_grid.y_grid.max())

    fig.suptitle("Seismoacoustic localisation sensitivity to seismic and infrasound picks")
    fig.savefig(save_path, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    topo = load_topography(TOPOGRAPHY_DIR)
    base_picks = load_arrival_picks(ARRIVAL_PICKS_FILE)
    stations = base_picks["station"].tolist()
    station_pick_options = list(itertools.product(SEISMIC_PICK_OPTIONS.keys(), INFRASOUND_PICK_OPTIONS.keys()))

    total_pick_combinations = int(math.pow(len(station_pick_options), len(stations)))
    total_combinations = total_pick_combinations * len(SEISMIC_VELOCITIES)
    print(f"Stations included: {stations}")
    print(f"Running {total_pick_combinations} pick combinations across {len(SEISMIC_VELOCITIES)} seismic velocities.")

    base_result = run_location(base_picks.copy(), topo, BASE_SEISMIC_VELOCITY)

    summary_rows = []
    progress = _progress_iter(total_combinations)

    try:
        for seismic_velocity in SEISMIC_VELOCITIES:
            for option_names in itertools.product(station_pick_options, repeat=len(stations)):
                scenario_picks = build_combination_pick_table(base_picks, stations, option_names)
                result = run_location(scenario_picks, topo, seismic_velocity)
                summary_rows.append({
                    "combination_name": _combo_name(stations, option_names),
                    "combination_code": "".join(
                        f"S{seismic_option[-1]}I{infrasound_option[-1]}"
                        for seismic_option, infrasound_option in option_names
                    ),
                    "seismic_velocity_mps": seismic_velocity,
                    "seismic_velocity_kmps": seismic_velocity / 1000.0,
                    "best_x_m": result["best_x"],
                    "best_y_m": result["best_y"],
                    "best_z_m": result["best_z"],
                    "best_rms_s": result["best_rms_s"],
                    "best_source_lag_s": result["best_source_lag_s"],
                    "stations_used": len(result["station_results"]),
                })
                if progress is not None:
                    progress.update(1)
                else:
                    count = len(summary_rows)
                    print(f"Completed {count}/{total_combinations} combinations", end="\r", flush=True)
    finally:
        if progress is not None:
            progress.close()
        else:
            print()

    summary_df = pd.DataFrame(summary_rows).sort_values("best_rms_s").reset_index(drop=True)
    summary_csv_path = OUTPUT_DIR / "seismoacoustic_all_pick_combinations_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    best_row = summary_df.iloc[0]
    summary_lines = [
        "Seismoacoustic all-pick-combination comparison",
        f"Stations included: {', '.join(stations)}",
        f"Pick combinations per velocity: {total_pick_combinations}",
        f"Per-station pick options: {', '.join(f'seis_{s}/inf_{i}' for s, i in station_pick_options)}",
        f"Seismic velocities (km/s): {', '.join(f'{velocity / 1000.0:.1f}' for velocity in SEISMIC_VELOCITIES)}",
        f"Total combinations: {total_combinations}",
        "",
        "Base scenario (midpoint picks, Vp = 3.0 km/s):",
        f"  Best easting: {base_result['best_x']:.1f} m",
        f"  Best northing: {base_result['best_y']:.1f} m",
        f"  Best elevation: {base_result['best_z']:.1f} m",
        f"  RMS residual: {base_result['best_rms_s']:.2f} s",
        f"  Fitted source lag: {base_result['best_source_lag_s']:.2f} s",
        "",
        "Best combination by RMS:",
        f"  Combination: {best_row['combination_name']}",
        f"  Code: {best_row['combination_code']}",
        f"  Seismic velocity: {best_row['seismic_velocity_kmps']:.1f} km/s",
        f"  Best easting: {best_row['best_x_m']:.1f} m",
        f"  Best northing: {best_row['best_y_m']:.1f} m",
        f"  Best elevation: {best_row['best_z_m']:.1f} m",
        f"  RMS residual: {best_row['best_rms_s']:.2f} s",
        f"  Fitted source lag: {best_row['best_source_lag_s']:.2f} s",
    ]

    summary_txt_path = OUTPUT_DIR / "seismoacoustic_all_pick_combinations_summary.txt"
    summary_txt_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    fig = plot_combination_summary(
        base_result,
        summary_df,
        topo,
        OUTPUT_DIR / "seismoacoustic_all_pick_combinations.pdf",
    )
    plt.close(fig)

    for line in summary_lines:
        print(line)
