"""
Plot benchmark station-scaling runtimes for ASL and SARA.

This script reads PAPER_CODE/Outputs/benchmark_station_scaling.csv and
generates a four-panel figure showing:
1. Total runtime (load + preprocess + localisation)
2. Localization-only runtime
3. Speedup relative to the analysed data duration
4. Localization cost per time slice
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent

SCALING_CSV = HERE / "Outputs" / "benchmark_station_scaling.csv"
SAVE_FIG = HERE / "FIGURES" / "Figure10.pdf"

TITLE_FONTSIZE = 14
AXIS_LABEL_FONTSIZE = 12
AXIS_TICK_FONTSIZE = 11
LEGEND_FONTSIZE = 11

ASL_COLORMAP = "YlOrBr"
SARA_COLORMAP = "Greens"
ASL_COLOR_LEVEL = 0.6
SARA_COLOR_LEVEL = 0.6

ASL_COLOR = plt.get_cmap(ASL_COLORMAP)(ASL_COLOR_LEVEL)
SARA_COLOR = plt.get_cmap(SARA_COLORMAP)(SARA_COLOR_LEVEL)

TEXT_COLUMNS = {"method", "site_amplification_csv"}
BOOLEAN_COLUMNS = {
    "site_amplification_applied",
    "intrinsic_attenuation_applied",
}


def _read_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark results not found: {path}. "
            "Run run_computational_cost.py before generating Figure 10."
        )

    rows = []
    with open(path, "r", encoding="utf-8", newline="") as fobj:
        reader = csv.DictReader(fobj)
        for row in reader:
            clean = {}
            for key, value in row.items():
                if key in TEXT_COLUMNS:
                    clean[key] = value
                elif key in BOOLEAN_COLUMNS:
                    normalized = value.strip().lower()
                    if normalized not in {"true", "false"}:
                        raise ValueError(
                            f"Invalid boolean {value!r} in column {key!r} of {path}."
                        )
                    clean[key] = normalized == "true"
                else:
                    try:
                        clean[key] = float(value)
                    except ValueError as exc:
                        raise ValueError(
                            f"Expected a numeric value in column {key!r}, "
                            f"but found {value!r} in {path}."
                        ) from exc
            rows.append(clean)
    return rows


def _rows_by_method(rows, method):
    return sorted(
        (row for row in rows if row["method"] == method),
        key=lambda row: row["stations"],
    )


def _plot_mean_std(ax, rows, mean_key, std_key, ylabel, title):
    stations = np.array([row["stations"] for row in rows], dtype=float)
    mean = np.array([row[mean_key] for row in rows], dtype=float)
    std = np.array([row[std_key] for row in rows], dtype=float)
    method = rows[0]["method"]
    color = ASL_COLOR if method == "ASL" else SARA_COLOR

    ax.plot(stations, mean, "o-", color=color, linewidth=2, label=f"{method} mean")
    ax.fill_between(stations, mean - std, mean + std, color=color, alpha=0.2, label=f"{method} ±1 std")
    ax.set_xlabel("Number of stations", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)


def _plot_metric(ax, asl_rows, sara_rows, key, ylabel, title, threshold_label=None):
    asl_stations = np.array([row["stations"] for row in asl_rows], dtype=float)
    sara_stations = np.array([row["stations"] for row in sara_rows], dtype=float)

    ax.plot(
        asl_stations,
        [row[key] for row in asl_rows],
        "o-",
        color=ASL_COLOR,
        linewidth=2,
        label="ASL",
    )
    ax.plot(
        sara_stations,
        [row[key] for row in sara_rows],
        "o-",
        color=SARA_COLOR,
        linewidth=2,
        label="SARA",
    )
    if threshold_label is not None:
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label=threshold_label)
    ax.set_xlabel("Number of stations", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)


def _plot_metric_mean_std(ax, asl_rows, sara_rows, mean_key, std_key, ylabel, title, threshold_label=None):
    for method, rows, color in (("ASL", asl_rows, ASL_COLOR), ("SARA", sara_rows, SARA_COLOR)):
        stations = np.array([row["stations"] for row in rows], dtype=float)
        mean = np.array([row[mean_key] for row in rows], dtype=float)
        std = np.array([row[std_key] for row in rows], dtype=float)
        ax.plot(stations, mean, "o-", color=color, linewidth=2, label=f"{method} mean")
        ax.fill_between(stations, mean - std, mean + std, color=color, alpha=0.2, label=f"{method} ±1 std")
    if threshold_label is not None:
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label=threshold_label)
    ax.set_xlabel("Number of stations", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)


def _plot_per_slice_runtime(ax, asl_rows, sara_rows):
    for method, rows, color in (("ASL", asl_rows, ASL_COLOR), ("SARA", sara_rows, SARA_COLOR)):
        stations = np.array([row["stations"] for row in rows], dtype=float)
        mean = np.array([row["localisation_runtime_per_slice_s"] for row in rows], dtype=float)
        std = np.array(
            [
                row["localisation_runtime_std_s"] / row["time_slices"]
                if row["time_slices"] > 0
                else 0.0
                for row in rows
            ],
            dtype=float,
        )
        ax.plot(stations, mean, "o-", color=color, linewidth=2, label=f"{method} mean")
        ax.fill_between(stations, mean - std, mean + std, color=color, alpha=0.2, label=f"{method} ±1 std")

    ax.set_xlabel("Number of stations", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Localization runtime per time step (s)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("(d) Localization runtime per time step", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)


if __name__ == "__main__":
    scaling_rows = _read_csv(SCALING_CSV)
    asl_rows = _rows_by_method(scaling_rows, "ASL")
    sara_rows = _rows_by_method(scaling_rows, "SARA")

    print("Computational cost figure")
    print(f"Input CSV: {SCALING_CSV}")
    print("Localization runtime in this file is the localization-only benchmark.")
    print("It excludes topography loading and event-data loading/preprocessing.")
    print("Panel (d) shows average localization cost per temporal step.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=False)
    ax_total = axes[0, 0]
    ax_localisation = axes[0, 1]
    ax_real_time = axes[1, 0]
    ax_cadence = axes[1, 1]

    _plot_mean_std(
        ax_total,
        asl_rows,
        "total_runtime_mean_s",
        "total_runtime_std_s",
        "Total runtime (s)",
        "(a) Total runtime",
    )
    _plot_mean_std(
        ax_total,
        sara_rows,
        "total_runtime_mean_s",
        "total_runtime_std_s",
        "Total runtime (s)",
        "(a) Total runtime",
    )
    ax_total.legend(fontsize=LEGEND_FONTSIZE)

    _plot_mean_std(
        ax_localisation,
        asl_rows,
        "localisation_runtime_mean_s",
        "localisation_runtime_std_s",
        "Localization runtime (s)",
        "(b) Localization runtime",
    )
    _plot_mean_std(
        ax_localisation,
        sara_rows,
        "localisation_runtime_mean_s",
        "localisation_runtime_std_s",
        "Localization runtime (s)",
        "(b) Localization runtime",
    )
    ax_localisation.legend(fontsize=LEGEND_FONTSIZE)

    total_upper = max(
        max(row["total_runtime_mean_s"] + row["total_runtime_std_s"] for row in asl_rows),
        max(row["total_runtime_mean_s"] + row["total_runtime_std_s"] for row in sara_rows),
    )
    localisation_upper = max(
        max(row["localisation_runtime_mean_s"] + row["localisation_runtime_std_s"] for row in asl_rows),
        max(row["localisation_runtime_mean_s"] + row["localisation_runtime_std_s"] for row in sara_rows),
    )
    runtime_upper = max(total_upper, localisation_upper)
    ax_total.set_ylim(0.0, runtime_upper * 1.05)
    ax_localisation.set_ylim(0.0, runtime_upper * 1.05)

    _plot_metric_mean_std(
        ax_real_time,
        asl_rows,
        sara_rows,
        "total_runtime_fraction_of_data",
        "total_runtime_fraction_of_data_std",
        "Total runtime / Data duration",
        "(c) Faster than real time?",
    )
    ax_real_time.legend(fontsize=LEGEND_FONTSIZE)

    _plot_per_slice_runtime(ax_cadence, asl_rows, sara_rows)
    ax_cadence.legend(fontsize=LEGEND_FONTSIZE)

    fig.tight_layout()
    SAVE_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_FIG, bbox_inches="tight")
    print(f"Saved figure -> {SAVE_FIG}")
