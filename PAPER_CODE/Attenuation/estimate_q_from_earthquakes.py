"""
Estimate effective attenuation from GeoNet earthquake waveforms.

The fitted model is, for each frequency band,

    log(A_es / S_s) + b log(r_es) = event_term_e + station_term_s - B r_es

where A_es is the band-limited RMS amplitude for event e at station s,
r_es is hypocentral distance in metres, S_s is an optional independent site
amplification correction, b is the fixed geometric spreading exponent, and B
is the intrinsic attenuation coefficient in 1/m.

The primary output for the ASL sensitivity analysis is B. Q is then reported as

    Q = pi f / (B beta)

for the chosen beta. If B <= 0, Q is left blank because that implies no
measurable positive attenuation in the fitted band.

This script downloads waveforms from GeoNet using ObsPy/FDSN and caches the
miniSEED files locally so failed/partial runs can be resumed.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime, read
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


HERE = Path(__file__).resolve().parent
CATALOGUE_CSV = HERE / "earthquakes_with_crater_distance.csv"
SITE_AMPLIFICATION_CSV = (
    HERE.parent / "SiteAmplification" / "site_amplification_summary.csv"
)
OUTPUT_DIR = HERE / "RESULTS"
WAVEFORM_DIR = OUTPUT_DIR / "waveforms"
STATION_XML = OUTPUT_DIR / "geonet_station_metadata.xml"
STATION_CSV = OUTPUT_DIR / "geonet_station_metadata.csv"

GEONET_CLIENT = "GEONET"
NETWORK = "NZ"
LOCATION = "*"
CHANNEL_PRIORITY = ["HHZ", "EHZ", "BHZ", "HNZ"]

CRATER_LAT = -39.1566302543244
CRATER_LON = 175.63253480007924

STATIONS = [
    "COVZ",
    "ETVZ",
    "FWVZ",
    "KRVZ",
    "NGZ",
    "NOVZ",
    "NTVZ",
    "OTVZ",
    "SNVZ",
    "TMVZ",
    "TUVZ",
    "TWVZ",
    "WHVZ",
    "WTVZ",
    "MAVZ",
    "TRVZ",
]

# Catalogue filters. Set any of these to None to disable that filter.
MIN_MAGNITUDE = 0
MAX_EVENT_DISTANCE_FROM_CRATER_KM = None
MAX_EVENT_DISTANCE_FROM_NEAREST_STATION_KM = 200.0
MAX_DEPTH_KM = 40.0
MAX_EVENTS = None

# The arrival velocity is only used to place an approximate S/coda window.
# It does not set the beta used to convert fitted B to Q.
P_ARRIVAL_VELOCITY_M_S = 6000.0
S_ARRIVAL_VELOCITY_M_S = 3500.0
SIGNAL_START_AFTER_S_S = 0.0
SIGNAL_WINDOW_LENGTH_S = 10.0
NOISE_END_BEFORE_P_S = 5.0
NOISE_WINDOW_LENGTH_S = 10.0
DOWNLOAD_PRE_ORIGIN_S = 60.0
DOWNLOAD_POST_SIGNAL_PADDING_S = 20.0

FREQUENCY_BANDS_HZ = [(1.0, 10.0)]

GEOMETRIC_SPREADING_EXPONENT = 1.0
BETA_FOR_Q_M_S = 3500.0
Q_REPORT_FREQUENCY_HZ = 3.7
MIN_SNR = 3.0
MIN_USABLE_STATIONS_PER_EVENT = 6
MIN_RECORDS_PER_BAND = 30
MIN_EVENTS_PER_BAND = 5

APPLY_SITE_AMPLIFICATION = True
SITE_AMPLIFICATION_COLUMN = "site_amplification"
FIT_RESIDUAL_STATION_TERMS = True

REMOVE_RESPONSE = True
RESPONSE_PREFILT = (0.5, 0.8, 12.0, 15.0)

BOOTSTRAP_ITERATIONS = 200
BOOTSTRAP_CENTRAL_INTERVAL_PERCENT = 68.0
RANDOM_SEED = 20260321

PLOT_DIAGNOSTICS = True
PLOT_DIR = OUTPUT_DIR / "diagnostic_plots"
SAVE_DIAGNOSTIC_PLOTS = True
SPYDER_SHOW_BLOCK = False


CATALOGUE_COLUMNS = {
    "event_id": ["eventid", "event_id", "publicid", "public_id", "contributorid"],
    "time": ["time", "origintime", "origin_time", "datetime"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "long"],
    "depth_km": ["depth/km", "depth_km", "depth", "depthkm"],
    "magnitude": ["magnitude", "mag", "m"],
}


def _normalise_column_name(name):
    return str(name).strip().lower().replace(" ", "").replace("-", "_")


def _find_column(columns, aliases, required=True):
    normalised = {_normalise_column_name(col): col for col in columns}
    for alias in aliases:
        key = _normalise_column_name(alias)
        if key in normalised:
            return normalised[key]
    if required:
        raise ValueError(f"Could not find any of these columns in catalogue: {aliases}")
    return None


def load_catalogue(path):
    path = Path(path)
    df = pd.read_csv(path)
    if len(df.columns) == 1 and "|" in df.columns[0]:
        df = pd.read_csv(path, sep="|")
    df.columns = [str(col).strip() for col in df.columns]

    selected = {}
    for key, aliases in CATALOGUE_COLUMNS.items():
        selected[key] = _find_column(df.columns, aliases, required=(key != "magnitude"))

    out = pd.DataFrame({
        "event_id": df[selected["event_id"]].astype(str).str.strip(),
        "time": pd.to_datetime(df[selected["time"]], utc=True, errors="coerce"),
        "latitude": pd.to_numeric(df[selected["latitude"]], errors="coerce"),
        "longitude": pd.to_numeric(df[selected["longitude"]], errors="coerce"),
        "depth_km": pd.to_numeric(df[selected["depth_km"]], errors="coerce"),
    })
    if selected["magnitude"] is not None:
        out["magnitude"] = pd.to_numeric(df[selected["magnitude"]], errors="coerce")
    else:
        out["magnitude"] = np.nan

    out = out.dropna(subset=["time", "latitude", "longitude", "depth_km"]).copy()
    out["distance_from_crater_km"] = [
        gps2dist_azimuth(CRATER_LAT, CRATER_LON, lat, lon)[0] / 1000.0
        for lat, lon in zip(out["latitude"], out["longitude"])
    ]

    if MIN_MAGNITUDE is not None:
        out = out[(out["magnitude"].isna()) | (out["magnitude"] >= MIN_MAGNITUDE)]
    if MAX_EVENT_DISTANCE_FROM_CRATER_KM is not None:
        out = out[out["distance_from_crater_km"] <= MAX_EVENT_DISTANCE_FROM_CRATER_KM]
    if MAX_DEPTH_KM is not None:
        out = out[out["depth_km"] <= MAX_DEPTH_KM]

    out = out.sort_values("time").reset_index(drop=True)
    if MAX_EVENTS is not None:
        out = out.head(MAX_EVENTS).copy()
    return out


def get_station_metadata(client, catalogue):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if STATION_XML.exists():
        return read_inventory_compat(STATION_XML)

    start = UTCDateTime(catalogue["time"].min().to_pydatetime()) - 86400
    end = UTCDateTime(catalogue["time"].max().to_pydatetime()) + 86400
    inventory = client.get_stations(
        network=NETWORK,
        station=",".join(STATIONS),
        location=LOCATION,
        channel=",".join(CHANNEL_PRIORITY),
        starttime=start,
        endtime=end,
        level="response" if REMOVE_RESPONSE else "channel",
    )
    inventory.write(str(STATION_XML), format="STATIONXML")
    return inventory


def read_inventory_compat(path):
    from obspy import read_inventory

    return read_inventory(str(path))


def station_lookup_from_inventory(inventory):
    rows = {}
    for net in inventory:
        for sta in net:
            if sta.code not in STATIONS:
                continue
            if len(sta) > 0:
                cha = sta[0]
                lat = cha.latitude if cha.latitude is not None else sta.latitude
                lon = cha.longitude if cha.longitude is not None else sta.longitude
                elev = cha.elevation if cha.elevation is not None else sta.elevation
            else:
                lat, lon, elev = sta.latitude, sta.longitude, sta.elevation
            rows[sta.code] = {"latitude": lat, "longitude": lon, "elevation_m": elev}
    missing = sorted(set(STATIONS) - set(rows))
    if missing:
        warnings.warn(f"Missing station metadata for: {missing}")
    return rows


def save_station_metadata_csv(station_lookup):
    station_rows = []
    for station, info in sorted(station_lookup.items()):
        station_rows.append({
            "station": station,
            "latitude": float(info["latitude"]),
            "longitude": float(info["longitude"]),
            "elevation_m": float(info["elevation_m"]),
            "distance_from_crater_km": gps2dist_azimuth(
                CRATER_LAT,
                CRATER_LON,
                float(info["latitude"]),
                float(info["longitude"]),
            )[0] / 1000.0,
        })
    station_df = pd.DataFrame(station_rows)
    station_df.to_csv(STATION_CSV, index=False)
    return station_df


def add_station_distance_summary(catalogue, station_lookup):
    catalogue = catalogue.copy()
    station_items = list(station_lookup.items())
    min_distances = []
    max_distances = []
    median_distances = []

    for row in catalogue.itertuples(index=False):
        distances_km = []
        for _, station_info in station_items:
            epicentral_m = gps2dist_azimuth(
                float(row.latitude),
                float(row.longitude),
                float(station_info["latitude"]),
                float(station_info["longitude"]),
            )[0]
            vertical_m = float(row.depth_km) * 1000.0 + float(station_info["elevation_m"])
            distances_km.append(float(np.sqrt(epicentral_m ** 2 + vertical_m ** 2)) / 1000.0)

        distances_km = np.asarray(distances_km, dtype=float)
        min_distances.append(float(np.min(distances_km)))
        max_distances.append(float(np.max(distances_km)))
        median_distances.append(float(np.median(distances_km)))

    catalogue["min_station_distance_km"] = min_distances
    catalogue["max_station_distance_km"] = max_distances
    catalogue["median_station_distance_km"] = median_distances

    if MAX_EVENT_DISTANCE_FROM_NEAREST_STATION_KM is not None:
        catalogue = catalogue[
            catalogue["min_station_distance_km"] <= MAX_EVENT_DISTANCE_FROM_NEAREST_STATION_KM
        ].copy()

    return catalogue.reset_index(drop=True)


def event_id_for_filename(event_id):
    keep = []
    for char in str(event_id):
        keep.append(char if char.isalnum() or char in ("_", "-") else "_")
    return "".join(keep)


def event_origin_utc(event_row):
    return UTCDateTime(event_row["time"].to_pydatetime())


def hypocentral_distance_m(event_row, station_info):
    epicentral_m = gps2dist_azimuth(
        float(event_row["latitude"]),
        float(event_row["longitude"]),
        float(station_info["latitude"]),
        float(station_info["longitude"]),
    )[0]
    vertical_m = float(event_row["depth_km"]) * 1000.0 + float(station_info["elevation_m"])
    return float(np.sqrt(epicentral_m ** 2 + vertical_m ** 2))


def event_distances(event_row, station_lookup):
    return {
        station: hypocentral_distance_m(event_row, station_info)
        for station, station_info in station_lookup.items()
    }


def waveform_cache_path(event_id):
    return WAVEFORM_DIR / f"{event_id_for_filename(event_id)}.mseed"


def download_event_waveforms(client, event_row, distances_m):
    WAVEFORM_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = waveform_cache_path(event_row["event_id"])
    if cache_path.exists():
        return read(str(cache_path))

    origin = event_origin_utc(event_row)
    max_s_time = max(distances_m.values()) / S_ARRIVAL_VELOCITY_M_S
    min_p_time = min(distances_m.values()) / P_ARRIVAL_VELOCITY_M_S
    required_pre_origin_s = max(
        DOWNLOAD_PRE_ORIGIN_S,
        NOISE_WINDOW_LENGTH_S + NOISE_END_BEFORE_P_S - min_p_time,
    )
    start = origin - required_pre_origin_s
    end = (
        origin
        + max_s_time
        + SIGNAL_START_AFTER_S_S
        + SIGNAL_WINDOW_LENGTH_S
        + DOWNLOAD_POST_SIGNAL_PADDING_S
    )

    stream = client.get_waveforms(
        network=NETWORK,
        station=",".join(STATIONS),
        location=LOCATION,
        channel=",".join(CHANNEL_PRIORITY),
        starttime=start,
        endtime=end,
        attach_response=False,
    )
    stream.merge(method=1, fill_value="interpolate")
    stream.write(str(cache_path), format="MSEED")
    return stream


def choose_station_trace(stream, station):
    station_stream = stream.select(station=station)
    if len(station_stream) == 0:
        return None
    for channel in CHANNEL_PRIORITY:
        candidates = station_stream.select(channel=channel)
        if len(candidates) > 0:
            return max(candidates, key=lambda tr: float(tr.stats.sampling_rate)).copy()
    return max(station_stream, key=lambda tr: float(tr.stats.sampling_rate)).copy()


def preprocess_trace(trace, inventory):
    trace = trace.copy()
    trace.detrend("linear")
    trace.detrend("demean")
    trace.taper(max_percentage=0.05, type="hann")
    if REMOVE_RESPONSE:
        trace.remove_response(
            inventory=inventory,
            output="VEL",
            pre_filt=RESPONSE_PREFILT,
            water_level=60,
        )
    return trace


def rms_in_window(trace, start, end):
    if start < trace.stats.starttime or end > trace.stats.endtime:
        return np.nan
    window = trace.copy().trim(starttime=start, endtime=end, pad=False)
    data = window.data.astype(float)
    if data.size < max(8, int(0.5 * float(trace.stats.sampling_rate))):
        return np.nan
    return float(np.sqrt(np.mean(data ** 2)))


def band_limited_rms(trace, freq_min, freq_max, signal_start, signal_end, noise_start, noise_end):
    filtered = trace.copy()
    filtered.filter("bandpass", freqmin=freq_min, freqmax=freq_max, corners=4, zerophase=True)
    signal_rms = rms_in_window(filtered, signal_start, signal_end)
    noise_rms = rms_in_window(filtered, noise_start, noise_end)
    if not np.isfinite(signal_rms) or not np.isfinite(noise_rms) or noise_rms <= 0.0:
        return np.nan, np.nan, np.nan
    return signal_rms, noise_rms, signal_rms / noise_rms


def measure_event_amplitudes(event_row, stream, inventory, distances_m):
    rows = []
    origin = event_origin_utc(event_row)

    usable_stations = 0
    for station in STATIONS:
        if station not in distances_m:
            continue

        trace = choose_station_trace(stream, station)
        if trace is None:
            continue

        try:
            trace = preprocess_trace(trace, inventory)
        except Exception as exc:
            warnings.warn(f"{event_row['event_id']} {station}: preprocessing failed: {exc}")
            continue

        distance_m = distances_m[station]
        p_arrival = origin + distance_m / P_ARRIVAL_VELOCITY_M_S
        s_arrival = origin + distance_m / S_ARRIVAL_VELOCITY_M_S
        signal_start = s_arrival + SIGNAL_START_AFTER_S_S
        signal_end = signal_start + SIGNAL_WINDOW_LENGTH_S
        noise_end = p_arrival - NOISE_END_BEFORE_P_S
        noise_start = noise_end - NOISE_WINDOW_LENGTH_S

        station_had_usable_band = False
        for freq_min, freq_max in FREQUENCY_BANDS_HZ:
            signal_rms, noise_rms, snr = band_limited_rms(
                trace,
                freq_min,
                freq_max,
                signal_start,
                signal_end,
                noise_start,
                noise_end,
            )
            if not np.isfinite(snr) or snr < MIN_SNR:
                continue

            station_had_usable_band = True
            centre_frequency = float(np.sqrt(freq_min * freq_max))
            rows.append({
                "event_id": event_row["event_id"],
                "origin_time": event_row["time"].isoformat(),
                "event_latitude": float(event_row["latitude"]),
                "event_longitude": float(event_row["longitude"]),
                "event_depth_km": float(event_row["depth_km"]),
                "event_magnitude": float(event_row["magnitude"]) if np.isfinite(event_row["magnitude"]) else np.nan,
                "station": station,
                "channel": trace.stats.channel,
                "distance_m": distance_m,
                "distance_km": distance_m / 1000.0,
                "freq_min_hz": freq_min,
                "freq_max_hz": freq_max,
                "centre_frequency_hz": centre_frequency,
                "signal_rms": signal_rms,
                "noise_rms": noise_rms,
                "snr": snr,
                "log_amplitude": float(np.log(max(signal_rms, 1e-30))),
            })

        if station_had_usable_band:
            usable_stations += 1

    if usable_stations < MIN_USABLE_STATIONS_PER_EVENT:
        return []
    return rows


def collect_amplitude_measurements(catalogue, client, inventory, station_lookup):
    all_rows = []
    failures = []
    iterator = catalogue.iterrows()
    if tqdm is not None:
        iterator = tqdm(
            iterator,
            total=len(catalogue),
            desc="Downloading/measuring earthquakes",
            unit="event",
        )

    for _, event_row in iterator:
        distances_m = event_distances(event_row, station_lookup)
        try:
            stream = download_event_waveforms(client, event_row, distances_m)
        except Exception as exc:
            failures.append({"event_id": event_row["event_id"], "stage": "download", "error": str(exc)})
            warnings.warn(f"{event_row['event_id']}: waveform download failed: {exc}")
            continue

        rows = measure_event_amplitudes(event_row, stream, inventory, distances_m)
        all_rows.extend(rows)
        message = f"{event_row['event_id']}: kept {len(rows)} event-station-band measurements"
        if tqdm is not None:
            iterator.set_postfix_str(message)
        else:
            print(message)

    amplitudes = pd.DataFrame(all_rows)
    failures_df = pd.DataFrame(failures)
    return amplitudes, failures_df


def load_site_amplification_table(path):
    path = Path(path)
    if not path.exists():
        if APPLY_SITE_AMPLIFICATION:
            warnings.warn(f"Site amplification file not found: {path}. Using no site correction.")
        return pd.DataFrame()

    site_df = pd.read_csv(path)
    required = {
        "station",
        "frequency_min_hz",
        "frequency_max_hz",
        SITE_AMPLIFICATION_COLUMN,
    }
    missing = sorted(required - set(site_df.columns))
    if missing:
        raise ValueError(f"Missing columns in site amplification table {path}: {missing}")

    site_df = site_df.copy()
    site_df["station"] = site_df["station"].astype(str)
    site_df["frequency_min_hz"] = pd.to_numeric(site_df["frequency_min_hz"], errors="coerce")
    site_df["frequency_max_hz"] = pd.to_numeric(site_df["frequency_max_hz"], errors="coerce")
    site_df[SITE_AMPLIFICATION_COLUMN] = pd.to_numeric(
        site_df[SITE_AMPLIFICATION_COLUMN],
        errors="coerce",
    )
    site_df = site_df.dropna(
        subset=["station", "frequency_min_hz", "frequency_max_hz", SITE_AMPLIFICATION_COLUMN]
    )
    site_df = site_df[site_df[SITE_AMPLIFICATION_COLUMN] > 0.0].copy()
    return site_df


def _select_site_amplification(site_df, station, freq_min, freq_max):
    if site_df.empty:
        return 1.0, "none", np.nan, np.nan

    station_df = site_df[site_df["station"] == station]
    if station_df.empty:
        warnings.warn(f"No site amplification entry for station {station}; using 1.0.")
        return 1.0, "missing_station", np.nan, np.nan

    exact = station_df[
        np.isclose(station_df["frequency_min_hz"], freq_min)
        & np.isclose(station_df["frequency_max_hz"], freq_max)
    ]
    if not exact.empty:
        row = exact.iloc[0]
        return (
            float(row[SITE_AMPLIFICATION_COLUMN]),
            "exact",
            float(row["frequency_min_hz"]),
            float(row["frequency_max_hz"]),
        )

    analysis_width = max(float(freq_max) - float(freq_min), 1e-12)
    candidates = []
    analysis_centre = np.sqrt(float(freq_min) * float(freq_max))
    for row in station_df.itertuples(index=False):
        site_min = float(row.frequency_min_hz)
        site_max = float(row.frequency_max_hz)
        overlap = min(float(freq_max), site_max) - max(float(freq_min), site_min)
        if overlap <= 0.0:
            continue
        site_centre = np.sqrt(site_min * site_max)
        candidates.append((
            overlap / analysis_width,
            -abs(np.log(site_centre / analysis_centre)),
            row,
        ))

    if not candidates:
        warnings.warn(
            f"No overlapping site amplification band for {station} "
            f"{freq_min:g}-{freq_max:g} Hz; using 1.0."
        )
        return 1.0, "missing_band", np.nan, np.nan

    _, _, row = max(candidates, key=lambda item: (item[0], item[1]))
    return (
        float(getattr(row, SITE_AMPLIFICATION_COLUMN)),
        "overlap",
        float(row.frequency_min_hz),
        float(row.frequency_max_hz),
    )


def apply_site_amplification_corrections(amplitudes, site_df):
    amplitudes = amplitudes.copy()
    if amplitudes.empty:
        return amplitudes

    site_amplification = []
    site_match_type = []
    site_freq_min = []
    site_freq_max = []

    for row in amplitudes.itertuples(index=False):
        if APPLY_SITE_AMPLIFICATION:
            amplification, match_type, matched_min, matched_max = _select_site_amplification(
                site_df=site_df,
                station=row.station,
                freq_min=float(row.freq_min_hz),
                freq_max=float(row.freq_max_hz),
            )
        else:
            amplification, match_type, matched_min, matched_max = 1.0, "disabled", np.nan, np.nan

        site_amplification.append(amplification)
        site_match_type.append(match_type)
        site_freq_min.append(matched_min)
        site_freq_max.append(matched_max)

    amplitudes["site_amplification"] = np.asarray(site_amplification, dtype=float)
    amplitudes["site_amplification_match"] = site_match_type
    amplitudes["site_amplification_freq_min_hz"] = np.asarray(site_freq_min, dtype=float)
    amplitudes["site_amplification_freq_max_hz"] = np.asarray(site_freq_max, dtype=float)
    amplitudes["site_amplification_log"] = np.log(
        np.maximum(amplitudes["site_amplification"].to_numpy(float), 1e-30)
    )
    amplitudes["log_amplitude_site_corrected"] = (
        amplitudes["log_amplitude"].to_numpy(float)
        - amplitudes["site_amplification_log"].to_numpy(float)
    )
    return amplitudes


def _fit_band_table(band_df):
    band_df = band_df.copy()
    log_amplitude_column = (
        "log_amplitude_site_corrected"
        if APPLY_SITE_AMPLIFICATION and "log_amplitude_site_corrected" in band_df.columns
        else "log_amplitude"
    )
    band_df["corrected_log_amplitude"] = (
        band_df[log_amplitude_column]
        + GEOMETRIC_SPREADING_EXPONENT * np.log(np.maximum(band_df["distance_m"].to_numpy(float), 1.0))
    )

    events = sorted(band_df["event_id"].unique())
    stations = sorted(band_df["station"].unique())
    if len(events) < MIN_EVENTS_PER_BAND or len(band_df) < MIN_RECORDS_PER_BAND or len(stations) < 3:
        return None

    event_index = {event_id: i for i, event_id in enumerate(events)}
    station_index = {station: i for i, station in enumerate(stations[1:])} if FIT_RESIDUAL_STATION_TERMS else {}

    n_rows = len(band_df)
    n_station_terms = max(0, len(stations) - 1) if FIT_RESIDUAL_STATION_TERMS else 0
    n_cols = len(events) + n_station_terms + 1
    design = np.zeros((n_rows, n_cols), dtype=float)

    for row_i, row in enumerate(band_df.itertuples(index=False)):
        design[row_i, event_index[row.event_id]] = 1.0
        if FIT_RESIDUAL_STATION_TERMS and row.station in station_index:
            design[row_i, len(events) + station_index[row.station]] = 1.0
        design[row_i, -1] = float(row.distance_m)

    y = band_df["corrected_log_amplitude"].to_numpy(float)
    coef, residuals, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ coef
    residual = y - predicted
    dof = max(1, n_rows - rank)
    residual_std = float(np.sqrt(np.sum(residual ** 2) / dof))

    covariance = residual_std ** 2 * np.linalg.pinv(design.T @ design)
    distance_coef = float(coef[-1])
    distance_coef_std = float(np.sqrt(max(covariance[-1, -1], 0.0)))
    attenuation_b = -distance_coef
    attenuation_b_std = distance_coef_std

    centre_frequency = float(np.median(band_df["centre_frequency_hz"]))
    q_frequency = float(Q_REPORT_FREQUENCY_HZ) if Q_REPORT_FREQUENCY_HZ is not None else centre_frequency
    q_value = np.nan
    if attenuation_b > 0.0:
        q_value = float(np.pi * q_frequency / (attenuation_b * BETA_FOR_Q_M_S))

    event_terms = pd.DataFrame({
        "event_id": events,
        "event_term": coef[:len(events)],
    })
    station_terms = pd.DataFrame({
        "station": [stations[0]] + stations[1:],
        "station_term": (
            np.r_[0.0, coef[len(events):-1]]
            if FIT_RESIDUAL_STATION_TERMS
            else np.zeros(len(stations), dtype=float)
        ),
    })

    fit = {
        "freq_min_hz": float(band_df["freq_min_hz"].iloc[0]),
        "freq_max_hz": float(band_df["freq_max_hz"].iloc[0]),
        "centre_frequency_hz": centre_frequency,
        "records": int(n_rows),
        "events": int(len(events)),
        "stations": int(len(stations)),
        "rank": int(rank),
        "residual_std": residual_std,
        "site_amplification_applied": bool(APPLY_SITE_AMPLIFICATION),
        "site_amplification_column": SITE_AMPLIFICATION_COLUMN,
        "residual_station_terms_fit": bool(FIT_RESIDUAL_STATION_TERMS),
        "log_amplitude_column": log_amplitude_column,
        "attenuation_B_per_m": attenuation_b,
        "attenuation_B_per_km": attenuation_b * 1000.0,
        "attenuation_B_std_per_m": attenuation_b_std,
        "attenuation_B_std_per_km": attenuation_b_std * 1000.0,
        "beta_for_Q_m_s": float(BETA_FOR_Q_M_S),
        "frequency_for_Q_hz": q_frequency,
        "Q": q_value,
    }
    return fit, event_terms, station_terms


def bootstrap_band_b(band_df, rng):
    events = np.array(sorted(band_df["event_id"].unique()))
    if BOOTSTRAP_ITERATIONS <= 0 or len(events) < MIN_EVENTS_PER_BAND:
        return np.nan, np.nan

    estimates = []
    grouped = {event_id: group for event_id, group in band_df.groupby("event_id")}
    iterator = range(BOOTSTRAP_ITERATIONS)
    if tqdm is not None:
        freq_min = float(band_df["freq_min_hz"].iloc[0])
        freq_max = float(band_df["freq_max_hz"].iloc[0])
        iterator = tqdm(
            iterator,
            desc=f"Bootstrapping {freq_min:g}-{freq_max:g} Hz",
            unit="sample",
            leave=False,
        )

    for _ in iterator:
        sampled_events = rng.choice(events, size=len(events), replace=True)
        sampled = pd.concat([grouped[event_id] for event_id in sampled_events], ignore_index=True)
        result = _fit_band_table(sampled)
        if result is None:
            continue
        fit, _, _ = result
        b_value = fit["attenuation_B_per_m"]
        if np.isfinite(b_value):
            estimates.append(b_value)

    if not estimates:
        return np.nan, np.nan
    estimates = np.asarray(estimates, dtype=float)
    lower_percentile = (100.0 - BOOTSTRAP_CENTRAL_INTERVAL_PERCENT) / 2.0
    upper_percentile = 100.0 - lower_percentile
    return (
        float(np.percentile(estimates, lower_percentile)),
        float(np.percentile(estimates, upper_percentile)),
    )


def fit_attenuation(amplitudes):
    if amplitudes.empty:
        raise ValueError("No amplitude measurements available for fitting.")

    rng = np.random.default_rng(RANDOM_SEED)
    fits = []
    all_event_terms = []
    all_station_terms = []

    band_keys = ["freq_min_hz", "freq_max_hz"]
    for (freq_min, freq_max), band_df in amplitudes.groupby(band_keys):
        result = _fit_band_table(band_df)
        if result is None:
            warnings.warn(f"Skipping {freq_min}-{freq_max} Hz: not enough records/events/stations.")
            continue

        fit, event_terms, station_terms = result
        b_lo, b_hi = bootstrap_band_b(band_df, rng)
        b_half_width = 0.5 * (b_hi - b_lo) if np.isfinite(b_lo) and np.isfinite(b_hi) else np.nan
        fit["bootstrap_central_interval_percent"] = float(BOOTSTRAP_CENTRAL_INTERVAL_PERCENT)
        fit["attenuation_B_bootstrap_lower_per_m"] = b_lo
        fit["attenuation_B_bootstrap_upper_per_m"] = b_hi
        fit["attenuation_B_bootstrap_half_width_per_m"] = b_half_width
        fit["attenuation_B_bootstrap_lower_per_km"] = b_lo * 1000.0 if np.isfinite(b_lo) else np.nan
        fit["attenuation_B_bootstrap_upper_per_km"] = b_hi * 1000.0 if np.isfinite(b_hi) else np.nan
        fit["attenuation_B_bootstrap_half_width_per_km"] = (
            b_half_width * 1000.0 if np.isfinite(b_half_width) else np.nan
        )
        fits.append(fit)

        event_terms["freq_min_hz"] = freq_min
        event_terms["freq_max_hz"] = freq_max
        station_terms["freq_min_hz"] = freq_min
        station_terms["freq_max_hz"] = freq_max
        all_event_terms.append(event_terms)
        all_station_terms.append(station_terms)

    if not fits:
        band_summaries = []
        for (freq_min, freq_max), band_df in amplitudes.groupby(band_keys):
            band_summaries.append(
                f"{freq_min:g}-{freq_max:g} Hz: {len(band_df)} records, "
                f"{band_df['event_id'].nunique()} events, "
                f"{band_df['station'].nunique()} stations"
            )
        details = "; ".join(band_summaries) or "no frequency bands"
        raise ValueError(
            "No attenuation band met the fitting requirements "
            f"(minimum {MIN_RECORDS_PER_BAND} records, "
            f"{MIN_EVENTS_PER_BAND} events, and 3 stations). "
            f"Available data: {details}."
        )

    fit_df = pd.DataFrame(fits)
    event_terms_df = pd.concat(all_event_terms, ignore_index=True) if all_event_terms else pd.DataFrame()
    station_terms_df = pd.concat(all_station_terms, ignore_index=True) if all_station_terms else pd.DataFrame()
    return fit_df, event_terms_df, station_terms_df


def _save_and_show(fig, save_path):
    if SAVE_DIAGNOSTIC_PLOTS and save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved diagnostic plot -> {save_path}")
    plt.show(block=SPYDER_SHOW_BLOCK)


def plot_site_amplification_diagnostics(site_df):
    if not PLOT_DIAGNOSTICS or site_df.empty:
        return None

    freq_min, freq_max = FREQUENCY_BANDS_HZ[0]
    rows = []
    for station in STATIONS:
        amplification, match_type, matched_min, matched_max = _select_site_amplification(
            site_df=site_df,
            station=station,
            freq_min=freq_min,
            freq_max=freq_max,
        )
        rows.append({
            "station": station,
            "site_amplification": amplification,
            "match_type": match_type,
            "matched_band": (
                f"{matched_min:g}-{matched_max:g} Hz"
                if np.isfinite(matched_min) and np.isfinite(matched_max)
                else "none"
            ),
        })

    plot_df = pd.DataFrame(rows)
    x = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x, plot_df["site_amplification"], color="0.35", edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="tab:red", linestyle="--", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["station"], rotation=45, ha="right")
    ax.set_ylabel("Site amplification factor")
    ax.set_title(f"Site amplification corrections used for {freq_min:g}-{freq_max:g} Hz")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    _save_and_show(fig, PLOT_DIR / "site_amplification_factors.pdf")
    return fig


def plot_measurement_coverage(amplitudes):
    if not PLOT_DIAGNOSTICS or amplitudes.empty:
        return None

    event_counts = (
        amplitudes.groupby("event_id")["station"]
        .nunique()
        .sort_values(ascending=False)
    )
    station_counts = (
        amplitudes.groupby("station")["event_id"]
        .nunique()
        .reindex(STATIONS)
        .fillna(0)
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    ax_event, ax_station = axes

    ax_event.plot(np.arange(1, len(event_counts) + 1), event_counts.to_numpy(), "o-", markersize=3)
    ax_event.axhline(MIN_USABLE_STATIONS_PER_EVENT, color="tab:red", linestyle="--", linewidth=1.1)
    ax_event.set_ylabel("Usable stations")
    ax_event.set_title("Usable station count by earthquake")
    ax_event.grid(True, alpha=0.3)

    x = np.arange(len(station_counts))
    ax_station.bar(x, station_counts.to_numpy(), color="0.35", edgecolor="black", linewidth=0.5)
    ax_station.set_xticks(x)
    ax_station.set_xticklabels(station_counts.index, rotation=45, ha="right")
    ax_station.set_ylabel("Usable earthquakes")
    ax_station.set_title("Usable earthquake count by station")
    ax_station.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    _save_and_show(fig, PLOT_DIR / "measurement_coverage.pdf")
    return fig


def plot_snr_distance_diagnostics(amplitudes):
    if not PLOT_DIAGNOSTICS or amplitudes.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax_scatter, ax_hist = axes

    for station in STATIONS:
        station_df = amplitudes[amplitudes["station"] == station]
        if station_df.empty:
            continue
        ax_scatter.scatter(
            station_df["distance_km"],
            station_df["snr"],
            s=18,
            alpha=0.65,
            label=station,
        )
    ax_scatter.axhline(MIN_SNR, color="tab:red", linestyle="--", linewidth=1.1)
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel("Hypocentral distance (km)")
    ax_scatter.set_ylabel("SNR")
    ax_scatter.set_title("SNR versus event-station distance")
    ax_scatter.grid(True, alpha=0.3)

    ax_hist.hist(amplitudes["snr"], bins=30, color="0.35", edgecolor="black", linewidth=0.4)
    ax_hist.axvline(MIN_SNR, color="tab:red", linestyle="--", linewidth=1.1)
    ax_hist.set_xlabel("SNR")
    ax_hist.set_ylabel("Measurements")
    ax_hist.set_title("Retained measurement SNR distribution")
    ax_hist.grid(True, axis="y", alpha=0.3)

    handles, labels = ax_scatter.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center right", fontsize=8, ncols=1)
        fig.tight_layout(rect=[0.0, 0.0, 0.9, 1.0])
    else:
        fig.tight_layout()

    _save_and_show(fig, PLOT_DIR / "snr_distance_diagnostics.pdf")
    return fig


def plot_attenuation_fit_diagnostic(amplitudes, fit_df, event_terms_df, station_terms_df):
    if not PLOT_DIAGNOSTICS or amplitudes.empty or fit_df.empty:
        return None

    freq_min, freq_max = FREQUENCY_BANDS_HZ[0]
    band_df = amplitudes[
        np.isclose(amplitudes["freq_min_hz"], freq_min)
        & np.isclose(amplitudes["freq_max_hz"], freq_max)
    ].copy()
    if band_df.empty:
        return None

    fit_row = fit_df[
        np.isclose(fit_df["freq_min_hz"], freq_min)
        & np.isclose(fit_df["freq_max_hz"], freq_max)
    ]
    if fit_row.empty:
        return None
    fit = fit_row.iloc[0]

    event_terms = event_terms_df[
        np.isclose(event_terms_df["freq_min_hz"], freq_min)
        & np.isclose(event_terms_df["freq_max_hz"], freq_max)
    ][["event_id", "event_term"]]
    station_terms = station_terms_df[
        np.isclose(station_terms_df["freq_min_hz"], freq_min)
        & np.isclose(station_terms_df["freq_max_hz"], freq_max)
    ][["station", "station_term"]]

    band_df = band_df.merge(event_terms, on="event_id", how="left")
    band_df = band_df.merge(station_terms, on="station", how="left")
    band_df["event_term"] = band_df["event_term"].fillna(0.0)
    band_df["station_term"] = band_df["station_term"].fillna(0.0)

    log_amplitude_column = (
        "log_amplitude_site_corrected"
        if APPLY_SITE_AMPLIFICATION and "log_amplitude_site_corrected" in band_df.columns
        else "log_amplitude"
    )
    band_df["residualized_log_amplitude"] = (
        band_df[log_amplitude_column].to_numpy(float)
        + GEOMETRIC_SPREADING_EXPONENT * np.log(np.maximum(band_df["distance_m"].to_numpy(float), 1.0))
        - band_df["event_term"].to_numpy(float)
        - band_df["station_term"].to_numpy(float)
    )

    distance_km = band_df["distance_km"].to_numpy(float)
    distance_line_km = np.linspace(distance_km.min(), distance_km.max(), 200)
    attenuation_b_per_km = float(fit["attenuation_B_per_km"])
    fit_line = -attenuation_b_per_km * distance_line_km

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for station in STATIONS:
        station_df = band_df[band_df["station"] == station]
        if station_df.empty:
            continue
        ax.scatter(
            station_df["distance_km"],
            station_df["residualized_log_amplitude"],
            s=22,
            alpha=0.7,
            label=station,
        )
    ax.plot(
        distance_line_km,
        fit_line,
        color="black",
        linewidth=2.0,
        label=f"Fit: B = {attenuation_b_per_km:.4f} km$^{{-1}}$",
    )
    ax.set_xlabel("Hypocentral distance (km)")
    ax.set_ylabel("Residualized log amplitude")
    ax.set_title(f"Attenuation fit after event and station corrections ({freq_min:g}-{freq_max:g} Hz)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncols=2)
    fig.tight_layout()

    _save_and_show(fig, PLOT_DIR / "attenuation_fit_residualized.pdf")
    return fig


def plot_event_terms_vs_magnitude(amplitudes, event_terms_df):
    if not PLOT_DIAGNOSTICS or amplitudes.empty or event_terms_df.empty:
        return None

    freq_min, freq_max = FREQUENCY_BANDS_HZ[0]
    event_terms = event_terms_df[
        np.isclose(event_terms_df["freq_min_hz"], freq_min)
        & np.isclose(event_terms_df["freq_max_hz"], freq_max)
    ][["event_id", "event_term"]].copy()
    if event_terms.empty:
        return None

    event_meta = (
        amplitudes.groupby("event_id", as_index=False)
        .agg(
            event_magnitude=("event_magnitude", "first"),
            event_depth_km=("event_depth_km", "first"),
            median_distance_km=("distance_km", "median"),
            usable_stations=("station", "nunique"),
            measurements=("station", "size"),
        )
    )
    plot_df = event_terms.merge(event_meta, on="event_id", how="left")
    plot_df = plot_df[np.isfinite(plot_df["event_magnitude"]) & np.isfinite(plot_df["event_term"])].copy()
    if plot_df.empty:
        return None

    x = plot_df["event_magnitude"].to_numpy(float)
    y = plot_df["event_term"].to_numpy(float)
    if len(plot_df) >= 2 and not np.allclose(x, x[0]):
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        corr = float(np.corrcoef(x, y)[0, 1])
    else:
        slope, intercept, x_line, y_line, corr = np.nan, np.nan, np.array([]), np.array([]), np.nan

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    scatter = ax.scatter(
        plot_df["event_magnitude"],
        plot_df["event_term"],
        c=plot_df["median_distance_km"],
        s=25 + 6 * plot_df["usable_stations"],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.35,
    )
    if x_line.size > 0:
        ax.plot(
            x_line,
            y_line,
            color="black",
            linewidth=1.5,
            label=f"Linear fit, r = {corr:.2f}",
        )
        ax.legend(loc="best", fontsize=9)

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Median event-station distance (km)")
    ax.set_xlabel("Catalogue magnitude")
    ax.set_ylabel("Event term")
    ax.set_title(f"Event term versus magnitude ({freq_min:g}-{freq_max:g} Hz)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _save_and_show(fig, PLOT_DIR / "event_term_vs_magnitude.pdf")
    return fig


def plot_diagnostics(amplitudes, site_df, fit_df, event_terms_df, station_terms_df):
    if not PLOT_DIAGNOSTICS:
        return
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_site_amplification_diagnostics(site_df)
    plot_measurement_coverage(amplitudes)
    plot_snr_distance_diagnostics(amplitudes)
    plot_attenuation_fit_diagnostic(amplitudes, fit_df, event_terms_df, station_terms_df)
    plot_event_terms_vs_magnitude(amplitudes, event_terms_df)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WAVEFORM_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading catalogue: {CATALOGUE_CSV}")
    catalogue = load_catalogue(CATALOGUE_CSV)
    catalogue_path = OUTPUT_DIR / "earthquakes_filtered_for_q.csv"
    print(f"Kept {len(catalogue)} catalogue events after magnitude/depth/crater filters")

    client = Client(GEONET_CLIENT)
    inventory = get_station_metadata(client, catalogue)
    station_lookup = station_lookup_from_inventory(inventory)
    station_df = save_station_metadata_csv(station_lookup)
    print(f"Saved station metadata -> {STATION_CSV}")
    print(station_df.to_string(index=False))

    catalogue = add_station_distance_summary(catalogue, station_lookup)
    catalogue.to_csv(catalogue_path, index=False)
    print(f"Kept {len(catalogue)} catalogue events after station-distance filters -> {catalogue_path}")

    amplitudes, failures = collect_amplitude_measurements(catalogue, client, inventory, station_lookup)
    site_df = load_site_amplification_table(SITE_AMPLIFICATION_CSV)
    amplitudes = apply_site_amplification_corrections(amplitudes, site_df)

    amplitudes_path = OUTPUT_DIR / "earthquake_band_amplitudes.csv"
    amplitudes.to_csv(amplitudes_path, index=False)
    print(f"Saved amplitude measurements -> {amplitudes_path}")

    if not failures.empty:
        failures_path = OUTPUT_DIR / "download_failures.csv"
        failures.to_csv(failures_path, index=False)
        print(f"Saved download failures -> {failures_path}")

    fit_df, event_terms_df, station_terms_df = fit_attenuation(amplitudes)
    fit_path = OUTPUT_DIR / "attenuation_B_Q_fits.csv"
    event_terms_path = OUTPUT_DIR / "attenuation_event_terms.csv"
    station_terms_path = OUTPUT_DIR / "attenuation_station_terms.csv"

    fit_df.to_csv(fit_path, index=False)
    event_terms_df.to_csv(event_terms_path, index=False)
    station_terms_df.to_csv(station_terms_path, index=False)

    print(f"Saved attenuation fits -> {fit_path}")
    print(f"Saved event terms -> {event_terms_path}")
    print(f"Saved station terms -> {station_terms_path}")
    print("\nBand fits:")
    print(fit_df.to_string(index=False))

    plot_diagnostics(amplitudes, site_df, fit_df, event_terms_df, station_terms_df)
