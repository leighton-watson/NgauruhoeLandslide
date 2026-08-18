"""
Estimate station site-amplification factors from regional earthquake coda waves.

This follows the common coda-normalization idea used for volcano ASL station
corrections: for each earthquake/station, measure RMS coda amplitudes beginning
at twice the predicted S-wave travel time, reject low-SNR windows, and compute
station amplitude ratios relative to a reference station or the network median.

The output amplification factors are multiplicative and frequency dependent.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime, read, read_inventory
from obspy.clients.fdsn import Client as FDSNClient
from obspy.geodetics import gps2dist_azimuth

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


HERE = Path(__file__).resolve().parent
BASE = HERE.parent

CATALOGUE_FILE = BASE / "DATA" / "earthquakes.csv"
RESULTS_DIR = HERE 
WAVEFORM_CACHE_DIR = RESULTS_DIR / "Waveforms"

SAVE_EVENT_CSV = RESULTS_DIR / "selected_events.csv"
SAVE_MEASUREMENT_CSV = RESULTS_DIR / "site_amplification_measurements.csv"
SAVE_RATIO_CSV = RESULTS_DIR / "site_amplification_ratios.csv"
SAVE_SUMMARY_CSV = RESULTS_DIR / "site_amplification_summary.csv"
SAVE_FIG = RESULTS_DIR / "site_amplification_summary.pdf"
SAVE_BAR_FIG = RESULTS_DIR / "site_amplification_factors.pdf"

FDSN_BASE_URL = "https://service.geonet.org.nz"

CRATER_LAT = -39.1566302543244
CRATER_LON = 175.63253480007924

STATIONS = [
    ("NZ", "OTVZ", "10", "HHZ"),
    ("NZ", "SNVZ", "10", "EHZ"),
    ("NZ", "ETVZ", "10", "HHZ"),
    ("NZ", "NGZ", "10", "HHZ"),
    ("NZ", "WTVZ", "10", "HHZ"),
    ("NZ", "NOVZ", "10", "EHZ"),
    ("NZ", "KRVZ", "10", "HHZ"),
    ("NZ", "TMVZ", "10", "HHZ"),
    ("NZ", "NTVZ", "10", "HHZ"),
    ("NZ", "COVZ", "11", "HHZ"),
    ("NZ", "FWVZ", "10", "HHZ"),
    ("NZ", "WHVZ", "10", "HHZ"),
    ("NZ", "TWVZ", "10", "HHZ"),
    ("NZ", "MAVZ", "11", "HHZ"),
    ("NZ", "TRVZ", "10", "HHZ"),
    ("NZ", "TUVZ", "10", "EHZ"),
]

MIN_MAGNITUDE = 2.5
MAX_EVENT_DEPTH_KM = 40.0
MIN_EVENT_DISTANCE_FROM_CRATER_KM = 0.0
MAX_EVENT_DISTANCE_FROM_CRATER_KM = 200.0
AZIMUTH_BIN_WIDTH_DEG = 45.0
EVENTS_PER_AZIMUTH_BIN = 2
MAX_SELECTED_EVENTS = 16
EVENT_TYPE = "earthquake"

NORMALIZATION_MODE = "reference"  # "reference" or "network_median"
REFERENCE_STATION = "OTVZ"

GUIDE_VP_KM_S = 6.0
GUIDE_VS_KM_S = 3.5

SECONDS_BEFORE_ORIGIN = 20.0
SECONDS_AFTER_ORIGIN = 180.0
NOISE_WINDOW_LENGTH_S = 10.0
NOISE_WINDOW_END_BEFORE_P_S = 1.0

CODA_START_MULTIPLE_OF_S = 2.0
CODA_WINDOW_LENGTH_S = 10.0
CODA_WINDOW_STEP_S = 5.0
N_CODA_WINDOWS = 3
MIN_CODA_SNR = 2.0

PRIMARY_ASL_BAND_HZ = (1.0, 10.0)
FREQUENCY_BANDS_HZ = [
    PRIMARY_ASL_BAND_HZ,
]

RESPONSE_OUTPUT = "VEL"
RESPONSE_PREFILT_HZ = (0.5, 0.8, 25.0, 30.0)
FILTER_CORNERS = 4

MIN_RATIOS_PER_STATION_BAND = 3

TITLE_FONTSIZE = 12
LABEL_FONTSIZE = 10
TICK_FONTSIZE = 9


def _float_or_nan(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _band_label(freqmin, freqmax):
    return f"{freqmin:g}-{freqmax:g}Hz"


def _load_catalogue(path):
    events = []
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if EVENT_TYPE and row.get("eventtype") != EVENT_TYPE:
                continue

            latitude = _float_or_nan(row.get("latitude"))
            longitude = _float_or_nan(row.get("longitude"))
            depth_km = _float_or_nan(row.get("depth"))
            magnitude = _float_or_nan(row.get("magnitude"))
            origin_time = row.get("origintime")

            if not origin_time:
                continue
            if not np.isfinite(latitude) or not np.isfinite(longitude):
                continue
            if not np.isfinite(depth_km) or not np.isfinite(magnitude):
                continue
            if magnitude < MIN_MAGNITUDE:
                continue
            if depth_km > MAX_EVENT_DEPTH_KM:
                continue

            distance_m, crater_to_event_azimuth, _ = gps2dist_azimuth(
                CRATER_LAT,
                CRATER_LON,
                latitude,
                longitude,
            )
            distance_km = distance_m / 1000.0
            if distance_km < MIN_EVENT_DISTANCE_FROM_CRATER_KM:
                continue
            if distance_km > MAX_EVENT_DISTANCE_FROM_CRATER_KM:
                continue

            azimuth_bin = int(crater_to_event_azimuth // AZIMUTH_BIN_WIDTH_DEG)
            events.append(
                {
                    "publicid": row.get("publicid", ""),
                    "origintime": origin_time,
                    "time": UTCDateTime(origin_time),
                    "latitude": latitude,
                    "longitude": longitude,
                    "depth_km": depth_km,
                    "magnitude": magnitude,
                    "magnitudetype": row.get("magnitudetype", ""),
                    "distance_from_crater_km": distance_km,
                    "azimuth_from_crater_deg": crater_to_event_azimuth,
                    "azimuth_bin": azimuth_bin,
                }
            )

    return events


def _select_azimuthally_distributed_events(events):
    selected = []
    n_bins = int(np.ceil(360.0 / AZIMUTH_BIN_WIDTH_DEG))
    for bin_index in range(n_bins):
        bin_events = [event for event in events if event["azimuth_bin"] == bin_index]
        bin_events.sort(key=lambda event: event["magnitude"], reverse=True)
        selected.extend(bin_events[:EVENTS_PER_AZIMUTH_BIN])

    selected.sort(key=lambda event: event["magnitude"], reverse=True)
    return selected[:MAX_SELECTED_EVENTS]


def _cache_paths(event, network, station, location, channel):
    clean_location = location if location else "blank"
    base = WAVEFORM_CACHE_DIR / event["publicid"]
    stub = f"{network}_{station}_{clean_location}_{channel}"
    return base / f"{stub}.mseed", base / f"{stub}.xml"


def _download_or_read_trace(client, event, station_config):
    network, station, location, channel = station_config
    mseed_path, xml_path = _cache_paths(event, network, station, location, channel)
    start = event["time"] - SECONDS_BEFORE_ORIGIN
    end = event["time"] + SECONDS_AFTER_ORIGIN

    if mseed_path.exists() and xml_path.exists():
        stream = read(str(mseed_path))
        inventory = read_inventory(str(xml_path))
    else:
        stream = client.get_waveforms(network, station, location, channel, start, end)
        inventory = client.get_stations(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=start,
            endtime=end,
            level="response",
        )
        mseed_path.parent.mkdir(parents=True, exist_ok=True)
        stream.write(str(mseed_path), format="MSEED")
        inventory.write(str(xml_path), format="STATIONXML")

    stream = stream.copy()
    stream.merge(method=1, fill_value=0.0)
    stream.trim(start, end, pad=True, fill_value=0.0)
    stream.detrend("demean")
    stream.detrend("linear")
    stream.taper(max_percentage=0.03)
    stream.remove_response(
        inventory=inventory,
        output=RESPONSE_OUTPUT,
        pre_filt=RESPONSE_PREFILT_HZ,
        water_level=60,
    )

    trace = stream[0]
    channel_info = inventory[0][0][0]
    station_latitude = float(channel_info.latitude)
    station_longitude = float(channel_info.longitude)
    return trace, station_latitude, station_longitude


def _arrival_geometry(event, station_latitude, station_longitude):
    horizontal_distance_m, event_to_station_azimuth, station_to_event_azimuth = gps2dist_azimuth(
        event["latitude"],
        event["longitude"],
        station_latitude,
        station_longitude,
    )
    horizontal_distance_km = horizontal_distance_m / 1000.0
    hypocentral_distance_km = np.hypot(horizontal_distance_km, event["depth_km"])
    p_time_s = hypocentral_distance_km / GUIDE_VP_KM_S
    s_time_s = hypocentral_distance_km / GUIDE_VS_KM_S

    return {
        "horizontal_distance_km": horizontal_distance_km,
        "hypocentral_distance_km": hypocentral_distance_km,
        "event_to_station_azimuth": event_to_station_azimuth,
        "station_to_event_azimuth": station_to_event_azimuth,
        "p_time_s": p_time_s,
        "s_time_s": s_time_s,
    }


def _rms(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan
    return float(np.sqrt(np.nanmean(values**2)))


def _window_rms(trace, start_s, end_s):
    times_s = trace.times() - SECONDS_BEFORE_ORIGIN
    mask = (times_s >= start_s) & (times_s < end_s)
    if not np.any(mask):
        return np.nan
    return _rms(trace.data[mask])


def _bandpassed_trace(trace, freqmin, freqmax):
    tr = trace.copy()
    tr.detrend("demean")
    tr.detrend("linear")
    tr.taper(max_percentage=0.03)
    tr.filter(
        "bandpass",
        freqmin=freqmin,
        freqmax=freqmax,
        corners=FILTER_CORNERS,
        zerophase=True,
    )
    return tr


def _station_event_measurements(client, event, station_config):
    network, station, location, channel = station_config
    trace, station_latitude, station_longitude = _download_or_read_trace(client, event, station_config)
    geometry = _arrival_geometry(event, station_latitude, station_longitude)

    p_time_s = geometry["p_time_s"]
    s_time_s = geometry["s_time_s"]
    noise_end_s = max(0.0, p_time_s - NOISE_WINDOW_END_BEFORE_P_S)
    noise_start_s = noise_end_s - NOISE_WINDOW_LENGTH_S
    coda_start_s = CODA_START_MULTIPLE_OF_S * s_time_s

    rows = []
    for freqmin, freqmax in FREQUENCY_BANDS_HZ:
        band_trace = _bandpassed_trace(trace, freqmin, freqmax)
        noise_rms = _window_rms(band_trace, noise_start_s, noise_end_s)

        for window_index in range(N_CODA_WINDOWS):
            window_start_s = coda_start_s + window_index * CODA_WINDOW_STEP_S
            window_end_s = window_start_s + CODA_WINDOW_LENGTH_S
            coda_rms = _window_rms(band_trace, window_start_s, window_end_s)

            snr = np.nan
            if np.isfinite(noise_rms) and noise_rms > 0.0 and np.isfinite(coda_rms):
                snr = coda_rms / noise_rms

            rows.append(
                {
                    "event_id": event["publicid"],
                    "origintime": event["origintime"],
                    "event_magnitude": event["magnitude"],
                    "event_azimuth_from_crater_deg": event["azimuth_from_crater_deg"],
                    "event_distance_from_crater_km": event["distance_from_crater_km"],
                    "network": network,
                    "station": station,
                    "location": location,
                    "channel": channel,
                    "latitude": station_latitude,
                    "longitude": station_longitude,
                    "hypocentral_distance_km": geometry["hypocentral_distance_km"],
                    "epicentral_distance_km": geometry["horizontal_distance_km"],
                    "p_guide_s": p_time_s,
                    "s_guide_s": s_time_s,
                    "coda_start_s": coda_start_s,
                    "frequency_min_hz": freqmin,
                    "frequency_max_hz": freqmax,
                    "frequency_band": _band_label(freqmin, freqmax),
                    "window_index": window_index,
                    "window_start_s": window_start_s,
                    "window_end_s": window_end_s,
                    "noise_rms": noise_rms,
                    "coda_rms": coda_rms,
                    "snr": snr,
                    "passes_snr": bool(np.isfinite(snr) and snr >= MIN_CODA_SNR),
                }
            )

    return rows


def _normalise_measurements(measurement_rows):
    groups = defaultdict(list)
    for row in measurement_rows:
        if not row["passes_snr"]:
            continue
        if not np.isfinite(row["coda_rms"]) or row["coda_rms"] <= 0.0:
            continue
        key = (row["event_id"], row["frequency_band"], row["window_index"])
        groups[key].append(row)

    ratio_rows = []
    for (event_id, frequency_band, window_index), rows in groups.items():
        denominator = np.nan
        reference_label = NORMALIZATION_MODE

        if NORMALIZATION_MODE == "reference":
            ref_rows = [row for row in rows if row["station"] == REFERENCE_STATION]
            if not ref_rows:
                continue
            denominator = ref_rows[0]["coda_rms"]
            reference_label = REFERENCE_STATION
        elif NORMALIZATION_MODE == "network_median":
            amplitudes = [row["coda_rms"] for row in rows if np.isfinite(row["coda_rms"]) and row["coda_rms"] > 0.0]
            if not amplitudes:
                continue
            denominator = float(np.nanmedian(amplitudes))
            reference_label = "network_median"
        else:
            raise ValueError(f"Unknown NORMALIZATION_MODE: {NORMALIZATION_MODE}")

        if not np.isfinite(denominator) or denominator <= 0.0:
            continue

        for row in rows:
            ratio = row["coda_rms"] / denominator
            ratio_rows.append(
                {
                    "event_id": event_id,
                    "origintime": row["origintime"],
                    "event_magnitude": row["event_magnitude"],
                    "event_azimuth_from_crater_deg": row["event_azimuth_from_crater_deg"],
                    "station": row["station"],
                    "frequency_band": frequency_band,
                    "frequency_min_hz": row["frequency_min_hz"],
                    "frequency_max_hz": row["frequency_max_hz"],
                    "window_index": window_index,
                    "reference": reference_label,
                    "coda_rms": row["coda_rms"],
                    "denominator_rms": denominator,
                    "site_ratio": ratio,
                    "snr": row["snr"],
                }
            )

    return ratio_rows


def _summarise_ratios(ratio_rows):
    groups = defaultdict(list)
    band_limits = {}
    for row in ratio_rows:
        key = (row["station"], row["frequency_band"])
        if np.isfinite(row["site_ratio"]) and row["site_ratio"] > 0.0:
            groups[key].append(row["site_ratio"])
            band_limits[row["frequency_band"]] = (row["frequency_min_hz"], row["frequency_max_hz"])

    summary_rows = []
    for (station, frequency_band), ratios in sorted(groups.items()):
        if len(ratios) < MIN_RATIOS_PER_STATION_BAND:
            continue

        ratios = np.asarray(ratios, dtype=float)
        log_ratios = np.log(ratios)
        freqmin, freqmax = band_limits[frequency_band]
        summary_rows.append(
            {
                "station": station,
                "frequency_band": frequency_band,
                "frequency_min_hz": freqmin,
                "frequency_max_hz": freqmax,
                "n_ratios": int(ratios.size),
                "site_amplification": float(np.exp(np.nanmean(log_ratios))),
                "multiplicative_std": float(np.exp(np.nanstd(log_ratios))),
                "median_ratio": float(np.nanmedian(ratios)),
                "p16_ratio": float(np.nanpercentile(ratios, 16)),
                "p84_ratio": float(np.nanpercentile(ratios, 84)),
            }
        )

    return summary_rows


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_summary(summary_rows, selected_events):
    stations = [station for _, station, _, _ in STATIONS]
    bands = [_band_label(freqmin, freqmax) for freqmin, freqmax in FREQUENCY_BANDS_HZ]
    values = np.full((len(bands), len(stations)), np.nan)

    for row in summary_rows:
        if row["station"] not in stations or row["frequency_band"] not in bands:
            continue
        band_idx = bands.index(row["frequency_band"])
        station_idx = stations.index(row["station"])
        values[band_idx, station_idx] = row["site_amplification"]

    fig = plt.figure(figsize=(14, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], width_ratios=[1.3, 1.0])
    ax_bar = fig.add_subplot(grid[0, 0])
    ax_map = fig.add_subplot(grid[0, 1])
    ax_heatmap = fig.add_subplot(grid[1, :])

    broad_band = _band_label(*PRIMARY_ASL_BAND_HZ)
    broad_rows = [row for row in summary_rows if row["frequency_band"] == broad_band]
    broad_rows.sort(key=lambda row: stations.index(row["station"]) if row["station"] in stations else 999)
    x = np.arange(len(broad_rows))
    bar_values = np.asarray([row["site_amplification"] for row in broad_rows], dtype=float)
    print(bar_values)
    lower_err = bar_values - np.asarray([row["p16_ratio"] for row in broad_rows], dtype=float)
    upper_err = np.asarray([row["p84_ratio"] for row in broad_rows], dtype=float) - bar_values
    yerr = np.vstack([np.maximum(lower_err, 0.0), np.maximum(upper_err, 0.0)])

    ax_bar.bar(
        x,
        bar_values,
        yerr=yerr,
        capsize=3,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
        color="tab:blue",
        alpha=0.78,
    )
    ax_bar.axhline(1.0, color="black", linestyle="--", linewidth=1.1)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([row["station"] for row in broad_rows], rotation=45, ha="right")
    ax_bar.set_ylabel("AF relative ratio", fontsize=LABEL_FONTSIZE)
    ax_bar.set_title(f"Site amplification, {broad_band}", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax_bar.grid(True, axis="y", alpha=0.25)
    ax_bar.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    event_lons = np.asarray([event["longitude"] for event in selected_events], dtype=float)
    event_lats = np.asarray([event["latitude"] for event in selected_events], dtype=float)
    event_mags = np.asarray([event["magnitude"] for event in selected_events], dtype=float)
    scatter = ax_map.scatter(
        event_lons,
        event_lats,
        c=event_mags,
        s=35 + 18 * np.maximum(event_mags - np.nanmin(event_mags), 0.0),
        cmap="viridis",
        edgecolor="black",
        linewidth=0.5,
        label="Earthquakes",
    )
    ax_map.scatter(
        [CRATER_LON],
        [CRATER_LAT],
        marker="^",
        s=95,
        color="tab:red",
        edgecolor="black",
        linewidth=0.8,
        label="Ngauruhoe",
        zorder=5,
    )
    ax_map.set_xlabel("Longitude", fontsize=LABEL_FONTSIZE)
    ax_map.set_ylabel("Latitude", fontsize=LABEL_FONTSIZE)
    ax_map.set_title("Earthquakes used for coda ratios", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax_map.grid(True, alpha=0.25)
    ax_map.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax_map.legend(loc="best", fontsize=TICK_FONTSIZE, frameon=True)
    map_cbar = fig.colorbar(scatter, ax=ax_map, pad=0.02)
    map_cbar.set_label("Magnitude", fontsize=LABEL_FONTSIZE)
    map_cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    finite_values = values[np.isfinite(values)]
    if finite_values.size:
        vmin = np.nanpercentile(finite_values, 5)
        vmax = np.nanpercentile(finite_values, 95)
    else:
        vmin, vmax = 0.0, 1.0
    image = ax_heatmap.imshow(values, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax_heatmap.set_yticks(np.arange(len(bands)))
    ax_heatmap.set_yticklabels(bands)
    ax_heatmap.set_xticks(np.arange(len(stations)))
    ax_heatmap.set_xticklabels(stations, rotation=45, ha="right")
    ax_heatmap.set_xlabel("Station", fontsize=LABEL_FONTSIZE)
    ax_heatmap.set_ylabel("Frequency band", fontsize=LABEL_FONTSIZE)
    ax_heatmap.set_title("Frequency-dependent site amplification", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax_heatmap.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    cbar = fig.colorbar(image, ax=ax_heatmap, pad=0.01)
    cbar.set_label("AF relative ratio", fontsize=LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    fig.suptitle(
        f"Coda-normalized site amplification ({NORMALIZATION_MODE}, reference={REFERENCE_STATION})",
        fontsize=TITLE_FONTSIZE + 2,
        fontweight="bold",
    )
    return fig


def _plot_amplification_factors(summary_rows):
    stations = [station for _, station, _, _ in STATIONS]
    primary_band = _band_label(*PRIMARY_ASL_BAND_HZ)
    rows = [row for row in summary_rows if row["frequency_band"] == primary_band]
    rows.sort(key=lambda row: stations.index(row["station"]) if row["station"] in stations else 999)

    x = np.arange(len(rows))
    values = np.asarray([row["site_amplification"] for row in rows], dtype=float)
    lower_err = values - np.asarray([row["p16_ratio"] for row in rows], dtype=float)
    upper_err = np.asarray([row["p84_ratio"] for row in rows], dtype=float) - values
    yerr = np.vstack([np.maximum(lower_err, 0.0), np.maximum(upper_err, 0.0)])

    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    ax.bar(
        x,
        values,
        yerr=yerr,
        capsize=3,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
        color="tab:blue",
        alpha=0.82,
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([row["station"] for row in rows], rotation=45, ha="right")
    ax.set_ylabel("Site amplification factor", fontsize=LABEL_FONTSIZE)
    
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    return fig


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    catalogue = _load_catalogue(CATALOGUE_FILE)
    selected_events = _select_azimuthally_distributed_events(catalogue)
    if not selected_events:
        raise RuntimeError("No catalogue events passed the site-amplification filters.")

    client = FDSNClient(FDSN_BASE_URL)
    all_measurement_rows = []
    skipped = []

    for event in tqdm(selected_events, desc="Processing events", unit="event"):
        for station_config in tqdm(
            STATIONS,
            desc=f"{event['publicid']} stations",
            unit="station",
            leave=False,
        ):
            _, station, _, _ = station_config
            try:
                all_measurement_rows.extend(_station_event_measurements(client, event, station_config))
            except Exception as exc:
                skipped.append((event["publicid"], station, str(exc)))

    ratio_rows = _normalise_measurements(all_measurement_rows)
    summary_rows = _summarise_ratios(ratio_rows)

    _write_csv(
        SAVE_EVENT_CSV,
        selected_events,
        [
            "publicid",
            "origintime",
            "latitude",
            "longitude",
            "depth_km",
            "magnitude",
            "magnitudetype",
            "distance_from_crater_km",
            "azimuth_from_crater_deg",
            "azimuth_bin",
        ],
    )
    _write_csv(
        SAVE_MEASUREMENT_CSV,
        all_measurement_rows,
        [
            "event_id",
            "origintime",
            "event_magnitude",
            "event_azimuth_from_crater_deg",
            "event_distance_from_crater_km",
            "network",
            "station",
            "location",
            "channel",
            "latitude",
            "longitude",
            "hypocentral_distance_km",
            "epicentral_distance_km",
            "p_guide_s",
            "s_guide_s",
            "coda_start_s",
            "frequency_min_hz",
            "frequency_max_hz",
            "frequency_band",
            "window_index",
            "window_start_s",
            "window_end_s",
            "noise_rms",
            "coda_rms",
            "snr",
            "passes_snr",
        ],
    )
    _write_csv(
        SAVE_RATIO_CSV,
        ratio_rows,
        [
            "event_id",
            "origintime",
            "event_magnitude",
            "event_azimuth_from_crater_deg",
            "station",
            "frequency_band",
            "frequency_min_hz",
            "frequency_max_hz",
            "window_index",
            "reference",
            "coda_rms",
            "denominator_rms",
            "site_ratio",
            "snr",
        ],
    )
    _write_csv(
        SAVE_SUMMARY_CSV,
        summary_rows,
        [
            "station",
            "frequency_band",
            "frequency_min_hz",
            "frequency_max_hz",
            "n_ratios",
            "site_amplification",
            "multiplicative_std",
            "median_ratio",
            "p16_ratio",
            "p84_ratio",
        ],
    )

    if summary_rows:
        fig = _plot_summary(summary_rows, selected_events)
        fig.savefig(SAVE_FIG, dpi=300, bbox_inches="tight")
        bar_fig = _plot_amplification_factors(summary_rows)
        bar_fig.savefig(SAVE_BAR_FIG, dpi=300, bbox_inches="tight")

    print(f"Catalogue events passing filters: {len(catalogue)}")
    print(f"Selected events: {len(selected_events)}")
    print(f"Raw coda measurements: {len(all_measurement_rows)}")
    print(f"Usable normalized ratios: {len(ratio_rows)}")
    print(f"Station-band summaries: {len(summary_rows)}")
    print(f"Normalization mode: {NORMALIZATION_MODE}")
    print(f"Reference station: {REFERENCE_STATION}")
    print(f"Coda start: {CODA_START_MULTIPLE_OF_S:.1f} x predicted S travel time")
    print(f"Minimum coda SNR: {MIN_CODA_SNR:.1f}")

    if skipped:
        print("Skipped event/station items:")
        for event_id, station, message in skipped:
            print(f"  {event_id} {station}: {message}")

    print(f"Saved selected events -> {SAVE_EVENT_CSV}")
    print(f"Saved measurements -> {SAVE_MEASUREMENT_CSV}")
    print(f"Saved ratios -> {SAVE_RATIO_CSV}")
    print(f"Saved summary -> {SAVE_SUMMARY_CSV}")
    if summary_rows:
        print(f"Saved figure -> {SAVE_FIG}")
        print(f"Saved amplification-factor figure -> {SAVE_BAR_FIG}")

    plt.show()
