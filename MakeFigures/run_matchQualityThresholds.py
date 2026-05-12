"""
Helper script to choose comparable ASL and SARA quality thresholds.

The thresholds are matched by equal acceptance fraction within the plotting
window, rather than by equal raw value.
"""

from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from AnalysisCodes.io_utils import load_results_bundle


ASL_OUTPUT = HERE / "RESULTS" / "ASL_output.npz"
SARA_OUTPUT = HERE / "RESULTS" / "SARA_output.npz"

# Current figure thresholds to compare.
ASL_THRESHOLD = 0.50
SARA_THRESHOLD = 0.55

# Which threshold should define the target acceptance fraction?
# Options: "ASL", "SARA"
REFERENCE_METHOD = "ASL"

# Extra target fractions to report as a quick lookup table.
TARGET_FRACTIONS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.75]


def _plot_window(bundle):
    start = float(bundle.get("plot_window_start_s", 0.0))
    end = float(bundle.get("plot_window_end_s", np.inf))
    return start, end


def _cropped_quality(bundle):
    time = np.asarray(bundle["result_time"], dtype=float)
    quality = np.asarray(bundle["quality"], dtype=float)
    start, end = _plot_window(bundle)
    mask = (time >= start) & (time <= end)
    return quality[mask]


def _acceptance_fraction(values, threshold):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan
    return float(np.mean(values >= threshold))


def _threshold_for_fraction(values, fraction):
    values = np.sort(np.asarray(values, dtype=float))[::-1]
    if values.size == 0:
        return np.nan
    fraction = float(np.clip(fraction, 0.0, 1.0))
    if fraction <= 0.0:
        return float(np.nextafter(values.max(), np.inf))
    n_keep = max(1, int(np.ceil(fraction * values.size)))
    return float(values[n_keep - 1])


def _format_pct(value):
    return f"{100.0 * value:.1f}%"


if __name__ == "__main__":
    asl = load_results_bundle(ASL_OUTPUT)
    sara = load_results_bundle(SARA_OUTPUT)

    asl_quality = _cropped_quality(asl)
    sara_quality = _cropped_quality(sara)

    asl_fraction = _acceptance_fraction(asl_quality, ASL_THRESHOLD)
    sara_fraction = _acceptance_fraction(sara_quality, SARA_THRESHOLD)

    print("Matched quality-threshold helper")
    print("=" * 72)
    print(f"Plot window: {asl.get('plot_start_time', asl['start_time'])} to {asl.get('plot_end_time', asl['end_time'])}")
    print(f"ASL windows in plot window : {asl_quality.size}")
    print(f"SARA windows in plot window: {sara_quality.size}")
    print()

    print("Current thresholds")
    print(f"  ASL  threshold = {ASL_THRESHOLD:.3f}  -> accepts {_format_pct(asl_fraction)} of windows")
    print(f"  SARA threshold = {SARA_THRESHOLD:.3f} -> accepts {_format_pct(sara_fraction)} of windows")
    print()

    sara_match_for_asl = _threshold_for_fraction(sara_quality, asl_fraction)
    asl_match_for_sara = _threshold_for_fraction(asl_quality, sara_fraction)

    print("Direct matched thresholds")
    print(
        f"  To match ASL {ASL_THRESHOLD:.3f} ({_format_pct(asl_fraction)} accepted), "
        f"use SARA {sara_match_for_asl:.3f}"
    )
    print(
        f"  To match SARA {SARA_THRESHOLD:.3f} ({_format_pct(sara_fraction)} accepted), "
        f"use ASL {asl_match_for_sara:.3f}"
    )
    print()

    if REFERENCE_METHOD.upper() == "ASL":
        reference_fraction = asl_fraction
        matched_threshold = sara_match_for_asl
        print(
            f"Recommended if ASL is the reference: keep ASL = {ASL_THRESHOLD:.3f}, "
            f"set SARA = {matched_threshold:.3f}"
        )
    else:
        reference_fraction = sara_fraction
        matched_threshold = asl_match_for_sara
        print(
            f"Recommended if SARA is the reference: keep SARA = {SARA_THRESHOLD:.3f}, "
            f"set ASL = {matched_threshold:.3f}"
        )
    print()

    print("Lookup table by equal acceptance fraction")
    print("  Fraction   ASL threshold   SARA threshold")
    for fraction in TARGET_FRACTIONS:
        asl_thr = _threshold_for_fraction(asl_quality, fraction)
        sara_thr = _threshold_for_fraction(sara_quality, fraction)
        print(f"  {_format_pct(fraction):>7s}   {asl_thr:>12.3f}   {sara_thr:>13.3f}")
