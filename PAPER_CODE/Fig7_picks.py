from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# User-editable parameters
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

INPUT_RESULTS_NPZ = HERE / "OnsetTimes" / "offline_change_point_arrival_picks_results.npz"
OUTPUT_DIR = HERE / "FIGURES"
OUTPUT_FIG = OUTPUT_DIR / "Figure7.pdf"

SHOW_PLOTS = True
SAVE_FIGURE = True

STATIONS = ["OTVZ", "SNVZ", "NGZ", "NOVZ", "ETVZ"]
KINDS = ("seismic", "infrasound")

PLOT_START_S = -30.0
PLOT_END_S = 50.0

TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 14
ANNOTATION_FONTSIZE = 14
STATION_LABEL_FONTSIZE = 15

REFERENCE_TIME_LABEL = "2026-03-21 13:36:05"

TAB20 = plt.get_cmap("tab20")
STATION_COLORS = {
    "OTVZ": TAB20(0),
    "SNVZ": TAB20(1),
    "NGZ": TAB20(2),
    "NOVZ": TAB20(3),
    "ETVZ": TAB20(6),
}


# ---------------------------------------------------------------------------
# Loading and plotting
# ---------------------------------------------------------------------------


def normalise_for_plot(data: np.ndarray) -> np.ndarray:
    scale = np.nanmax(np.abs(data))
    if not np.isfinite(scale) or scale <= 0.0:
        return data.astype(float)
    return data.astype(float) / scale


def load_pick_results(path: Path) -> list[dict]:
    results = []
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(data["_metadata_json"].item())
        for idx, item in enumerate(metadata):
            prefix = f"result_{idx}_"
            result = dict(item)
            result["time_s"] = data[prefix + "time_s"]
            result["data"] = data[prefix + "data"]
            result["log_envelope"] = data[prefix + "log_envelope"]
            result["candidate_times"] = data[prefix + "candidate_times"]
            result["posterior"] = data[prefix + "posterior"]
            results.append(result)
    return results


def make_figure(results: list[dict], save_path: Path) -> None:
    nrows = len(STATIONS)
    fig, axes = plt.subplots(nrows, 2, figsize=(14, max(2.6 * nrows, 9)), sharex=True)
    axes = np.atleast_2d(axes)

    by_key = {(result["station"], result["kind"]): result for result in results}

    for row, station in enumerate(STATIONS):
        for col, kind in enumerate(KINDS):
            result = by_key[(station, kind)]
            ax = axes[row, col]
            ax_env = ax.twinx()

            time_s = result["time_s"]
            waveform = result["data"]
            log_envelope = result["log_envelope"]
            pick_s = float(result["pick_s"])
            pre_mean = float(result["pre_mean"])
            post_mean = float(result["post_mean"])

            station_color = STATION_COLORS.get(station, "0.15")
            ax.plot(time_s, normalise_for_plot(waveform), color=station_color, linewidth=0.9)
            ax_env.plot(time_s, log_envelope, color="black", linewidth=1.0, alpha=0.9)

            if np.isfinite(pick_s):
                ax.axvline(pick_s, color="black", linewidth=1.2)
                pre_mask = (time_s < pick_s) & np.isfinite(log_envelope)
                post_mask = (time_s >= pick_s) & np.isfinite(log_envelope)
                ax_env.plot(
                    time_s[pre_mask],
                    np.full(np.count_nonzero(pre_mask), pre_mean),
                    color="black",
                    linestyle="--",
                    linewidth=1.0,
                )
                ax_env.plot(
                    time_s[post_mask],
                    np.full(np.count_nonzero(post_mask), post_mean),
                    color="black",
                    linestyle="--",
                    linewidth=1.0,
                )

            ax.set_ylim(-1.15, 1.15)
            ax.set_xlim(PLOT_START_S, PLOT_END_S)
            ax.set_yticks([])
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
            ax.tick_params(axis="y", left=False)

            ax_env.tick_params(axis="y", labelsize=TICK_FONTSIZE, colors="black")

            if row == 0:
                title = "(a) Seismic" if col == 0 else "(b) Infrasound"
                ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
            if col == 0:
                ax.set_ylabel(station, fontsize=STATION_LABEL_FONTSIZE)

            label = f"Pick = {pick_s:.2f} s" if np.isfinite(pick_s) else "Pick = none"
            ax.text(
                0.03,
                0.12,
                label,
                transform=ax.transAxes,
                va="top",
                fontsize=ANNOTATION_FONTSIZE,
                color="black",
                fontweight="bold",
            )

    time_label = f"Time (s) since {REFERENCE_TIME_LABEL} UTC"
    axes[-1, 0].set_xlabel(time_label, fontsize=LABEL_FONTSIZE)
    axes[-1, 1].set_xlabel(time_label, fontsize=LABEL_FONTSIZE)

    fig.tight_layout()
    if SAVE_FIGURE:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    results = load_pick_results(INPUT_RESULTS_NPZ)
    make_figure(results, OUTPUT_FIG)


if __name__ == "__main__":
    main()
