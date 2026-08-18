"""Plot site amplification and the residualized attenuation fit together."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
Q_DIR = HERE / "Attenuation"
RESULTS_DIR = Q_DIR / "RESULTS"

SUMMARY_CSV = HERE / "SiteAmplification" / "site_amplification_summary.csv"
AMPLITUDES_CSV = RESULTS_DIR / "earthquake_band_amplitudes.csv"
FITS_CSV = RESULTS_DIR / "attenuation_B_Q_fits.csv"
EVENT_TERMS_CSV = RESULTS_DIR / "attenuation_event_terms.csv"
STATION_TERMS_CSV = RESULTS_DIR / "attenuation_station_terms.csv"
STATION_METADATA_CSV = RESULTS_DIR / "geonet_station_metadata.csv"

FREQUENCY_BAND = "1-10Hz"
FREQ_MIN_HZ = 1.0
FREQ_MAX_HZ = 10.0
GEOMETRIC_SPREADING_EXPONENT = 1.0

SAVE_FIG = HERE / "FIGURES" / "Figure4.pdf"
SPYDER_SHOW_BLOCK = False
PLOT_BOOTSTRAP_INTERVAL = False

LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10


def _read_required_csv(path, description):
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"The {description} is empty: {path}. "
            "Rerun Attenuation/estimate_q_from_earthquakes.py and ensure "
            "that enough events pass the fitting criteria."
        ) from exc


def _load_station_metadata():
    if not STATION_METADATA_CSV.exists():
        return pd.DataFrame(columns=["station", "distance_from_crater_km"])

    metadata = pd.read_csv(STATION_METADATA_CSV)
    required = {"station", "distance_from_crater_km"}
    if not required.issubset(metadata.columns):
        return pd.DataFrame(columns=sorted(required))

    metadata = metadata[["station", "distance_from_crater_km"]].dropna().copy()
    metadata["station"] = metadata["station"].astype(str)
    return metadata.sort_values(["distance_from_crater_km", "station"])


def _station_order(present_stations, metadata):
    present = {str(station) for station in present_stations}
    ordered = [station for station in metadata["station"] if station in present]
    return ordered + sorted(present.difference(ordered))


def _station_color_map(station_order):
    cmap = plt.get_cmap("tab20")
    return {
        station: cmap(index % cmap.N)
        for index, station in enumerate(station_order)
    }


def _uncertainty_bounds(rows):
    values = rows["site_amplification"].to_numpy(float)
    if {"p16_ratio", "p84_ratio"}.issubset(rows.columns):
        lower = rows["p16_ratio"].to_numpy(float)
        upper = rows["p84_ratio"].to_numpy(float)
    elif "multiplicative_std" in rows.columns:
        spread = rows["multiplicative_std"].to_numpy(float)
        lower = values / spread
        upper = values * spread
    else:
        lower = values
        upper = values

    return np.vstack(
        [np.maximum(values - lower, 0.0), np.maximum(upper - values, 0.0)]
    )


def _load_site_amplification(metadata):
    table = pd.read_csv(SUMMARY_CSV)
    required = {"station", "site_amplification"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{SUMMARY_CSV} is missing columns: {sorted(missing)}")

    if "frequency_band" not in table.columns:
        raise ValueError(f"{SUMMARY_CSV} has no 'frequency_band' column.")

    rows = table.loc[
        table["frequency_band"].astype(str) == FREQUENCY_BAND
    ].copy()
    if rows.empty:
        available = sorted(table["frequency_band"].astype(str).unique())
        raise ValueError(f"No rows for {FREQUENCY_BAND}. Available bands: {available}")

    rows["station"] = rows["station"].astype(str)
    order = _station_order(rows["station"], metadata)
    order_index = {station: index for index, station in enumerate(order)}
    return rows.sort_values(
        "station", key=lambda values: values.map(order_index)
    ), order


def _fit_value(fit, preferred_key, fallback_key=None):
    if preferred_key in fit and np.isfinite(fit[preferred_key]):
        return float(fit[preferred_key])
    if fallback_key is not None and fallback_key in fit and np.isfinite(fit[fallback_key]):
        return float(fit[fallback_key])
    return np.nan


def _load_attenuation_data():
    amplitudes = _read_required_csv(AMPLITUDES_CSV, "amplitude table")
    fits = _read_required_csv(FITS_CSV, "attenuation-fit table")
    event_terms = _read_required_csv(EVENT_TERMS_CSV, "event-term table")
    station_terms = _read_required_csv(STATION_TERMS_CSV, "station-term table")

    def band_mask(table):
        return (
            np.isclose(table["freq_min_hz"], FREQ_MIN_HZ)
            & np.isclose(table["freq_max_hz"], FREQ_MAX_HZ)
        )

    amplitudes = amplitudes.loc[band_mask(amplitudes)].copy()
    fit_rows = fits.loc[band_mask(fits)]
    event_terms = event_terms.loc[band_mask(event_terms), ["event_id", "event_term"]]
    station_terms = station_terms.loc[
        band_mask(station_terms), ["station", "station_term"]
    ]

    if amplitudes.empty:
        raise ValueError(f"No amplitudes for {FREQ_MIN_HZ:g}-{FREQ_MAX_HZ:g} Hz.")
    if fit_rows.empty:
        raise ValueError(f"No fit for {FREQ_MIN_HZ:g}-{FREQ_MAX_HZ:g} Hz.")

    fit = fit_rows.iloc[0]
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

    fitted_amplitude_column = fit.get("log_amplitude_column")
    if isinstance(fitted_amplitude_column, str):
        if fitted_amplitude_column not in data.columns:
            raise ValueError(
                f"Fit used {fitted_amplitude_column!r}, but that column is absent "
                f"from {AMPLITUDES_CSV}."
            )
        amplitude_column = fitted_amplitude_column
    else:
        amplitude_column = (
            "log_amplitude_site_corrected"
            if "log_amplitude_site_corrected" in data.columns
            else "log_amplitude"
        )
    data["residualized_log_amplitude"] = (
        data[amplitude_column].to_numpy(float)
        + GEOMETRIC_SPREADING_EXPONENT
        * np.log(np.maximum(data["distance_m"].to_numpy(float), 1.0))
        - data["event_term"].to_numpy(float)
        - data["station_term"].to_numpy(float)
    )
    data["station"] = data["station"].astype(str)
    return data, fit


def _coefficient_of_determination(observed, predicted):
    """Return the conventional R-squared for the solid-line predictions."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    finite = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[finite]
    predicted = predicted[finite]
    if observed.size < 2:
        return np.nan

    ss_residual = np.sum((observed - predicted) ** 2)
    ss_total = np.sum((observed - observed.mean()) ** 2)
    return 1.0 - ss_residual / ss_total if ss_total > 0.0 else np.nan


def _plot_site_amplification(ax, rows, station_to_color):
    x = np.arange(len(rows))
    values = rows["site_amplification"].to_numpy(float)
    colors = [station_to_color[station] for station in rows["station"]]
    ax.bar(
        x,
        values,
        yerr=_uncertainty_bounds(rows),
        capsize=3,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
        color=colors,
        alpha=0.82,
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(rows["station"], rotation=45, ha="right")
    ax.set_ylabel("Site Amplification Factor", fontsize=LABEL_FONTSIZE)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.text(
        0.015, 0.97, "(a)", transform=ax.transAxes,
        ha="left", va="top", fontsize=13, fontweight="bold",
    )


def _plot_attenuation(ax, data, fit, station_order, station_to_color):
    b_per_km = float(fit["attenuation_B_per_km"])
    b_lo = _fit_value(
        fit, "attenuation_B_bootstrap_lower_per_km",
        "attenuation_B_bootstrap_p025_per_km",
    )
    b_hi = _fit_value(
        fit, "attenuation_B_bootstrap_upper_per_km",
        "attenuation_B_bootstrap_p975_per_km",
    )
    interval = _fit_value(fit, "bootstrap_central_interval_percent")

    for station in station_order:
        station_data = data.loc[data["station"] == station]
        if station_data.empty:
            continue
        ax.scatter(
            station_data["distance_km"],
            station_data["residualized_log_amplitude"],
            s=36,
            alpha=0.7,
            color=station_to_color[station],
            edgecolor="none",
            label=station,
        )

    distance_km = data["distance_km"].to_numpy(float)
    observed = data["residualized_log_amplitude"].to_numpy(float)
    predicted = -b_per_km * distance_km
    r_squared = _coefficient_of_determination(observed, predicted)
    x_line = np.linspace(distance_km.min(), distance_km.max(), 250)
    ax.plot(
        x_line,
        -b_per_km * x_line,
        color="black",
        linewidth=2.0,
        label=f"Fit: B = {b_per_km:.4f} km$^{{-1}}$, $R^2$ = {r_squared:.3f}",
    )

    if PLOT_BOOTSTRAP_INTERVAL and np.isfinite(b_lo) and np.isfinite(b_hi):
        y_lo = -b_lo * x_line
        y_hi = -b_hi * x_line
        interval_label = (
            f"Bootstrap {interval:.0f}% interval"
            if np.isfinite(interval) else "Bootstrap interval"
        )
        ax.fill_between(
            x_line, np.minimum(y_lo, y_hi), np.maximum(y_lo, y_hi),
            color="black", alpha=0.14, linewidth=0, label=interval_label,
        )
        ax.plot(x_line, y_lo, color="black", linestyle=":", linewidth=1.2)
        ax.plot(x_line, y_hi, color="black", linestyle=":", linewidth=1.2)

    ax.set_xlabel("Hypocentral Distance (km)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Residualized Log Amplitude", fontsize=LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.legend(loc="best", fontsize=9, ncols=2)
    ax.text(
        0.015, 0.97, "(b)", transform=ax.transAxes,
        ha="left", va="top", fontsize=13, fontweight="bold",
    )
    return r_squared


def main():
    metadata = _load_station_metadata()
    site_rows, site_order = _load_site_amplification(metadata)
    attenuation_data, fit = _load_attenuation_data()

    all_stations = site_order + [
        station for station in attenuation_data["station"].unique()
        if station not in site_order
    ]
    station_order = _station_order(all_stations, metadata)
    station_to_color = _station_color_map(station_order)

    fig, axes = plt.subplots(2, 1, figsize=(11, 11), constrained_layout=True)
    _plot_site_amplification(axes[0], site_rows, station_to_color)
    r_squared = _plot_attenuation(
        axes[1], attenuation_data, fit, station_order, station_to_color
    )

    print(
        f"Loaded {attenuation_data['event_id'].nunique()} earthquakes, "
        f"{attenuation_data['station'].nunique()} stations, and "
        f"{len(attenuation_data)} attenuation measurements."
    )
    print(f"R^2 for the solid black attenuation fit: {r_squared:.6f}")

    if SAVE_FIG is not None:
        SAVE_FIG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(SAVE_FIG, dpi=300, bbox_inches="tight")
        print(f"Saved combined figure -> {SAVE_FIG}")

    plt.show(block=SPYDER_SHOW_BLOCK)


if __name__ == "__main__":
    main()
