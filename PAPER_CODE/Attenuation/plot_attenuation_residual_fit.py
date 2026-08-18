"""
Plot the residualized log-amplitude attenuation fit from saved Q-estimation outputs.

This script does not download or reprocess waveforms. It loads the CSV outputs
written by estimate_q_from_earthquakes.py and reconstructs the quantity used to
fit the attenuation slope:

    residualized log amplitude =
        log(A_site_corrected) + b log(r) - event_term - station_term

The fitted model is then:

    residualized log amplitude = -B r
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "RESULTS"

AMPLITUDES_CSV = RESULTS_DIR / "earthquake_band_amplitudes.csv"
FITS_CSV = RESULTS_DIR / "attenuation_B_Q_fits.csv"
EVENT_TERMS_CSV = RESULTS_DIR / "attenuation_event_terms.csv"
STATION_TERMS_CSV = RESULTS_DIR / "attenuation_station_terms.csv"
STATION_METADATA_CSV = RESULTS_DIR / "geonet_station_metadata.csv"

FREQ_MIN_HZ = 1.0
FREQ_MAX_HZ = 10.0
GEOMETRIC_SPREADING_EXPONENT = 1.0

SAVE_FIG = RESULTS_DIR / "diagnostic_plots" / "attenuation_residual_log_amplitude_fit.pdf"
SPYDER_SHOW_BLOCK = False
PLOT_BOOTSTRAP_INTERVAL = True


def _read_required_csv(path, description):
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"The {description} is empty: {path}. "
            "Rerun estimate_q_from_earthquakes.py and ensure that enough "
            "events pass the fitting criteria."
        ) from exc


def _station_order_by_crater_distance(data):
    if STATION_METADATA_CSV.exists():
        station_meta = pd.read_csv(STATION_METADATA_CSV)
        required = {"station", "distance_from_crater_km"}
        if required.issubset(station_meta.columns):
            stations = (
                station_meta[["station", "distance_from_crater_km"]]
                .dropna()
                .assign(station=lambda df: df["station"].astype(str))
                .sort_values(["distance_from_crater_km", "station"])
                ["station"]
                .tolist()
            )
            present = set(data["station"].astype(str).unique())
            ordered = [station for station in stations if station in present]
            missing = sorted(present - set(ordered))
            return ordered + missing

    return sorted(data["station"].astype(str).unique())


def _station_color_map(station_order):
    cmap = plt.get_cmap("tab20")
    return {
        station: cmap(index % cmap.N)
        for index, station in enumerate(station_order)
    }


def _fit_value(fit, preferred_key, fallback_key=None):
    if preferred_key in fit and np.isfinite(fit[preferred_key]):
        return float(fit[preferred_key])
    if fallback_key is not None and fallback_key in fit and np.isfinite(fit[fallback_key]):
        return float(fit[fallback_key])
    return np.nan


def _load_band_tables():
    amplitudes = _read_required_csv(AMPLITUDES_CSV, "amplitude table")
    fits = _read_required_csv(FITS_CSV, "attenuation-fit table")
    event_terms = _read_required_csv(EVENT_TERMS_CSV, "event-term table")
    station_terms = _read_required_csv(STATION_TERMS_CSV, "station-term table")

    band_mask = (
        np.isclose(amplitudes["freq_min_hz"], FREQ_MIN_HZ)
        & np.isclose(amplitudes["freq_max_hz"], FREQ_MAX_HZ)
    )
    band_amplitudes = amplitudes[band_mask].copy()

    fit_mask = (
        np.isclose(fits["freq_min_hz"], FREQ_MIN_HZ)
        & np.isclose(fits["freq_max_hz"], FREQ_MAX_HZ)
    )
    band_fit = fits[fit_mask].copy()

    event_mask = (
        np.isclose(event_terms["freq_min_hz"], FREQ_MIN_HZ)
        & np.isclose(event_terms["freq_max_hz"], FREQ_MAX_HZ)
    )
    band_event_terms = event_terms[event_mask][["event_id", "event_term"]].copy()

    station_mask = (
        np.isclose(station_terms["freq_min_hz"], FREQ_MIN_HZ)
        & np.isclose(station_terms["freq_max_hz"], FREQ_MAX_HZ)
    )
    band_station_terms = station_terms[station_mask][["station", "station_term"]].copy()

    if band_amplitudes.empty:
        raise ValueError(f"No amplitude rows found for {FREQ_MIN_HZ:g}-{FREQ_MAX_HZ:g} Hz.")
    if band_fit.empty:
        raise ValueError(f"No fit row found for {FREQ_MIN_HZ:g}-{FREQ_MAX_HZ:g} Hz.")

    return band_amplitudes, band_fit.iloc[0], band_event_terms, band_station_terms


def _build_residualized_table(amplitudes, event_terms, station_terms):
    data = amplitudes.merge(event_terms, on="event_id", how="left")
    data = data.merge(station_terms, on="station", how="left")
    missing_events = sorted(
        data.loc[data["event_term"].isna(), "event_id"].astype(str).unique()
    )
    missing_stations = sorted(
        data.loc[data["station_term"].isna(), "station"].astype(str).unique()
    )
    if missing_events or missing_stations:
        problems = []
        if missing_events:
            problems.append(f"events {missing_events}")
        if missing_stations:
            problems.append(f"stations {missing_stations}")
        raise ValueError(
            "Attenuation terms are missing for " + " and ".join(problems) + "."
        )

    if "log_amplitude_site_corrected" in data.columns:
        log_amplitude_column = "log_amplitude_site_corrected"
    else:
        log_amplitude_column = "log_amplitude"

    data["residualized_log_amplitude"] = (
        data[log_amplitude_column].to_numpy(float)
        + GEOMETRIC_SPREADING_EXPONENT * np.log(np.maximum(data["distance_m"].to_numpy(float), 1.0))
        - data["event_term"].to_numpy(float)
        - data["station_term"].to_numpy(float)
    )
    data["log_amplitude_column"] = log_amplitude_column
    return data


def plot_residual_fit(data, fit):
    n_events = data["event_id"].nunique()
    n_stations = data["station"].nunique()
    n_records = len(data)
    b_per_km = float(fit["attenuation_B_per_km"])
    b_lo_per_km = _fit_value(
        fit,
        "attenuation_B_bootstrap_lower_per_km",
        "attenuation_B_bootstrap_p025_per_km",
    )
    b_hi_per_km = _fit_value(
        fit,
        "attenuation_B_bootstrap_upper_per_km",
        "attenuation_B_bootstrap_p975_per_km",
    )
    bootstrap_interval = _fit_value(fit, "bootstrap_central_interval_percent")
    q_value = float(fit["Q"]) if "Q" in fit and np.isfinite(fit["Q"]) else np.nan

    distance_km = data["distance_km"].to_numpy(float)
    x_line = np.linspace(distance_km.min(), distance_km.max(), 250)
    y_line = -b_per_km * x_line
    has_bootstrap = (
        PLOT_BOOTSTRAP_INTERVAL
        and np.isfinite(b_lo_per_km)
        and np.isfinite(b_hi_per_km)
    )

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    station_order = _station_order_by_crater_distance(data)
    station_to_color = _station_color_map(station_order)

    for station in station_order:
        station_df = data[data["station"].astype(str) == station]
        if station_df.empty:
            continue
        ax.scatter(
            station_df["distance_km"],
            station_df["residualized_log_amplitude"],
            s=24,
            alpha=0.7,
            color=station_to_color[station],
            edgecolor="none",
            label=station,
        )

    ax.plot(
        x_line,
        y_line,
        color="black",
            linewidth=2.0,
            label=f"Fit: B = {b_per_km:.4f} km$^{{-1}}$",
        )
    if has_bootstrap:
        interval_label = (
            f"Bootstrap {bootstrap_interval:.0f}% interval"
            if np.isfinite(bootstrap_interval)
            else "Bootstrap interval"
        )
        y_lo = -b_lo_per_km * x_line
        y_hi = -b_hi_per_km * x_line
        lower_envelope = np.minimum(y_lo, y_hi)
        upper_envelope = np.maximum(y_lo, y_hi)
        ax.fill_between(
            x_line,
            lower_envelope,
            upper_envelope,
            color="black",
            alpha=0.14,
            linewidth=0,
            label=interval_label,
        )
        ax.plot(x_line, y_lo, color="black", linestyle=":", linewidth=1.2)
        ax.plot(x_line, y_hi, color="black", linestyle=":", linewidth=1.2)

    summary = (
        f"Earthquakes: {n_events}\n"
        f"Stations: {n_stations}\n"
        f"Measurements: {n_records}\n"
        f"B: {b_per_km:.4f} km$^{{-1}}$"
    )
    if has_bootstrap:
        interval_text = f"{bootstrap_interval:.0f}%" if np.isfinite(bootstrap_interval) else "interval"
        summary += f"\nB {interval_text}: {b_lo_per_km:.4f}-{b_hi_per_km:.4f} km$^{{-1}}$"
    if np.isfinite(q_value):
        summary += f"\nQ: {q_value:.0f}"

    ax.text(
        0.02,
        0.03,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.4", "alpha": 0.9},
    )

    ax.set_xlabel("Hypocentral distance (km)")
    ax.set_ylabel("Residualized log amplitude")
    ax.set_title(
        f"Residualized attenuation fit ({FREQ_MIN_HZ:g}-{FREQ_MAX_HZ:g} Hz)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncols=2)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    amplitudes, fit, event_terms, station_terms = _load_band_tables()
    residualized = _build_residualized_table(amplitudes, event_terms, station_terms)

    print(
        f"Loaded {residualized['event_id'].nunique()} earthquakes, "
        f"{residualized['station'].nunique()} stations, "
        f"{len(residualized)} measurements for {FREQ_MIN_HZ:g}-{FREQ_MAX_HZ:g} Hz."
    )

    fig = plot_residual_fit(residualized, fit)

    if SAVE_FIG is not None:
        SAVE_FIG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(SAVE_FIG, bbox_inches="tight")
        print(f"Saved figure -> {SAVE_FIG}")

    plt.show(block=SPYDER_SHOW_BLOCK)
