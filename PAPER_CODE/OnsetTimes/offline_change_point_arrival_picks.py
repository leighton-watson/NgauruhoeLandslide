from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime, read
from obspy.clients.fdsn import Client as FDSNClient
from obspy.core.inventory import Inventory


# ---------------------------------------------------------------------------
# User-editable parameters
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "DATA"
CACHE_DIR = Path(__file__).resolve().parent / "offline_change_point_arrival_picks_cache"

SEISMIC_MSEED = DATA_DIR / "seismic_20260321_133600.mseed"
INFRASOUND_MSEED = DATA_DIR / "infrasound_20260321_133600.mseed"

OUTPUT_CSV = Path(__file__).resolve().parent / "offline_change_point_arrival_picks.csv"
OUTPUT_RESULTS_NPZ = Path(__file__).resolve().parent / "offline_change_point_arrival_picks_results.npz"
OUTPUT_FIG = Path(__file__).resolve().parent / "offline_change_point_arrival_picks.png"
OUTPUT_DIAGNOSTIC_FIG = Path(__file__).resolve().parent / "offline_change_point_diagnostics.png"
OUTPUT_STATION_DIAGNOSTIC_DIR = Path(__file__).resolve().parent / "offline_change_point_station_diagnostics"

# Set RUN_ANALYSIS = False after one successful run to reload
# OUTPUT_RESULTS_NPZ and redraw figures without recomputing picks.
RUN_ANALYSIS = True
PLOT_RESULTS = True
SAVE_ANALYSIS_RESULTS = True
SHOW_PLOTS = True
SAVE_FIGURE = True
SAVE_DIAGNOSTIC_FIGURES = True

PLOT_START_S = -50.0
PLOT_END_S = 50.0
TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 14
ANNOTATION_FONTSIZE = 14
STATION_LABEL_FONTSIZE = 15

DOWNLOAD_DATA = True
FORCE_DOWNLOAD = False
FDSN_BASE_URL = "https://service.geonet.org.nz"

# All offsets below are relative to this fixed absolute time.
REFERENCE_TIME_UTC = UTCDateTime("2026-03-21T13:36:05")

STATIONS = ["OTVZ", "SNVZ", "NGZ", "NOVZ", "ETVZ"]

TAB20 = plt.get_cmap("tab20")
STATION_COLORS = {
    "OTVZ": TAB20(0),
    "SNVZ": TAB20(1),
    "NGZ": TAB20(2),
    "NOVZ": TAB20(3),
    "ETVZ": TAB20(6),
}

START_OFFSET_S = -60.0
END_OFFSET_S = 60.0

# Baseline/reference window shown in the plots and recorded in the CSV.
NOISE_START_S = -50.0
NOISE_END_S = -10.0

# Offline change-point model window. Candidate onsets are tested inside the
# detection windows below, but each candidate is scored using data from
# MODEL_START_S to MODEL_END_S.
MODEL_START_S = -30.0
MODEL_END_S = 60.0
MIN_PRE_CHANGE_S = 5.0
MIN_POST_CHANGE_S = 5.0

# Characteristic function: log of smoothed analytic envelope normalized by the
# noise-window median envelope amplitude.
SEISMIC_SMOOTH_S = 2.0
INFRASOUND_SMOOTH_S = 2.0
LOG_EPS_FRACTION = 1.0e-6

# Optional prior that mildly favours earlier picks inside the detection window.
# Set to 0.0 for a uniform prior over candidate onset times.
SEISMIC_EARLY_PRIOR_DECAY_S = 0.0
INFRASOUND_EARLY_PRIOR_DECAY_S = 0.0

SEISMIC_FREQ_MIN = 1.0
SEISMIC_FREQ_MAX = 10.0
INFRASOUND_FREQ_MIN = 2.0
INFRASOUND_FREQ_MAX = 15.0

SEISMIC_CHANNELS = {
    "OTVZ": ("NZ", "10", "HHZ"),
    "SNVZ": ("NZ", "10", "EHZ"),
    "NGZ": ("NZ", "10", "HHZ"),
    "NOVZ": ("NZ", "10", "EHZ"),
    "ETVZ": ("NZ", "10", "HHZ"),
}

INFRASOUND_CHANNELS = {
    "OTVZ": ("NZ", "31", "HDF"),
    "SNVZ": ("NZ", "30", "HDF"),
    "NGZ": ("NZ", "30", "HDF"),
    "NOVZ": ("NZ", "30", "HDF"),
    "ETVZ": ("NZ", "31", "HDF"),
}

# Edit these if you want to restrict candidate onsets to plausible windows.
DETECTION_WINDOWS_S = {
    "seismic": {
        "OTVZ": (0.0, 60.0),
        "SNVZ": (0.0, 60.0),
        "NGZ": (0.0, 60.0),
        "NOVZ": (0.0, 60.0),
        "ETVZ": (0.0, 60.0),
    },
    "infrasound": {
        "OTVZ": (0.0, 60.0),
        "SNVZ": (0.0, 60.0),
        "NGZ": (0.0, 60.0),
        "NOVZ": (0.0, 60.0),
        "ETVZ": (0.0, 60.0),
    },
}


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PickConfig:
    kind: str
    mseed_path: Path
    freq_min: float
    freq_max: float
    smooth_s: float
    early_prior_decay_s: float


@dataclass
class PickResult:
    station: str
    kind: str
    pick_s: float
    credible_start_s: float
    credible_end_s: float
    posterior_probability: float
    envelope_noise_median: float
    pre_mean: float
    post_mean: float
    amplitude_shift: float
    status: str
    detection_start_s: float
    detection_end_s: float
    time_s: np.ndarray
    data: np.ndarray
    log_envelope: np.ndarray
    candidate_times: np.ndarray
    posterior: np.ndarray
    config: PickConfig


def absolute_start_time() -> UTCDateTime:
    return REFERENCE_TIME_UTC + START_OFFSET_S


def absolute_end_time() -> UTCDateTime:
    return REFERENCE_TIME_UTC + END_OFFSET_S


def cache_paths(kind: str) -> tuple[Path, Path]:
    start = absolute_start_time().strftime("%Y%m%d_%H%M%S")
    end = absolute_end_time().strftime("%Y%m%d_%H%M%S")
    reference = REFERENCE_TIME_UTC.strftime("%Y%m%d_%H%M%S")
    stub = f"{kind}_reference_{reference}_window_{start}_{end}"
    return CACHE_DIR / f"{stub}.mseed", CACHE_DIR / f"{stub}.xml"


def download_stream(channel_map: dict[str, tuple[str, str, str]], mseed_path: Path, xml_path: Path, label: str) -> None:
    client = FDSNClient(FDSN_BASE_URL)
    stream = Stream()
    inventory = Inventory(networks=[], source="GeoNet via ObsPy")
    start = absolute_start_time()
    end = absolute_end_time()

    print(f"Downloading {label} data from {start} to {end}")
    for station in STATIONS:
        if station not in channel_map:
            print(f"  skipped {station}: no channel mapping")
            continue

        network, location, channel = channel_map[station]
        seed_id = f"{network}.{station}.{location}.{channel}"
        print(f"  {seed_id}")
        try:
            stream += client.get_waveforms(network, station, location, channel, start, end)
            inventory += client.get_stations(
                network=network,
                station=station,
                location=location,
                channel=channel,
                starttime=start,
                endtime=end,
                level="response",
            )
        except Exception as exc:
            print(f"    skipped: {exc}")

    if len(stream) == 0:
        raise RuntimeError(f"No {label} waveforms were downloaded.")

    stream.sort(keys=["station", "channel"])
    corrected = Stream()
    skipped = []
    for trace in stream:
        tr = trace.copy()
        try:
            tr.remove_sensitivity(inventory=inventory)
            tr.detrend("demean")
            tr.detrend("linear")
            corrected += tr
        except Exception as exc:
            skipped.append((tr.id, str(exc)))

    if len(corrected) == 0:
        details = "; ".join(f"{seed_id}: {message}" for seed_id, message in skipped)
        raise RuntimeError(f"No {label} traces had usable sensitivity metadata. {details}")

    if skipped:
        print(f"Skipped {len(skipped)} {label} trace(s) without usable sensitivity:")
        for seed_id, message in skipped:
            print(f"  {seed_id}: {message}")

    mseed_path.parent.mkdir(parents=True, exist_ok=True)
    corrected.write(str(mseed_path), format="MSEED")
    inventory.write(str(xml_path), format="STATIONXML")
    print(f"Saved {label} cache to {mseed_path}")


def ensure_input_files() -> dict[str, Path]:
    if not DOWNLOAD_DATA:
        return {
            "seismic": SEISMIC_MSEED,
            "infrasound": INFRASOUND_MSEED,
        }

    seismic_mseed, seismic_xml = cache_paths("seismic")
    infrasound_mseed, infrasound_xml = cache_paths("infrasound")

    if FORCE_DOWNLOAD or not seismic_mseed.exists() or not seismic_xml.exists():
        download_stream(SEISMIC_CHANNELS, seismic_mseed, seismic_xml, "seismic")
    if FORCE_DOWNLOAD or not infrasound_mseed.exists() or not infrasound_xml.exists():
        download_stream(INFRASOUND_CHANNELS, infrasound_mseed, infrasound_xml, "infrasound")

    return {
        "seismic": seismic_mseed,
        "infrasound": infrasound_mseed,
    }


def read_filtered_trace(config: PickConfig, station: str):
    stream = read(str(config.mseed_path))
    selected = stream.select(station=station)
    if len(selected) == 0:
        raise ValueError(f"Station {station!r} was not found in {config.mseed_path.name}")

    trace = selected[0].copy()
    trace.trim(absolute_start_time(), absolute_end_time())
    trace.detrend("demean")
    trace.detrend("linear")
    trace.taper(max_percentage=0.05)
    trace.filter(
        "bandpass",
        freqmin=config.freq_min,
        freqmax=config.freq_max,
        corners=4,
        zerophase=True,
    )

    sampling_rate = float(trace.stats.sampling_rate)
    time_s = np.arange(trace.stats.npts, dtype=float) / sampling_rate
    time_s += float(trace.stats.starttime - REFERENCE_TIME_UTC)
    return time_s, trace.data.astype(float), sampling_rate


def analytic_envelope(data: np.ndarray) -> np.ndarray:
    npts = data.size
    spectrum = np.fft.fft(data)
    multiplier = np.zeros(npts)
    if npts % 2 == 0:
        multiplier[0] = 1.0
        multiplier[npts // 2] = 1.0
        multiplier[1 : npts // 2] = 2.0
    else:
        multiplier[0] = 1.0
        multiplier[1 : (npts + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(spectrum * multiplier))


def causal_moving_mean(data: np.ndarray, npts: int) -> np.ndarray:
    npts = max(1, int(npts))
    values = data.astype(float)
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    end = np.arange(1, values.size + 1)
    start = np.maximum(0, end - npts)
    window_sum = cumsum[end] - cumsum[start]
    window_n = end - start
    return window_sum / window_n


def log_smoothed_envelope(
    time_s: np.ndarray,
    data: np.ndarray,
    sampling_rate: float,
    smooth_s: float,
) -> tuple[np.ndarray, float]:
    envelope = analytic_envelope(data)
    smoothed = causal_moving_mean(envelope, round(smooth_s * sampling_rate))
    noise_mask = (
        (time_s >= NOISE_START_S)
        & (time_s <= NOISE_END_S)
        & np.isfinite(smoothed)
        & (smoothed > 0.0)
    )
    if np.count_nonzero(noise_mask) == 0:
        raise ValueError(
            f"Noise window {NOISE_START_S:g} to {NOISE_END_S:g} s does not overlap positive envelope values."
        )

    noise_median = float(np.nanmedian(smoothed[noise_mask]))
    if not np.isfinite(noise_median) or noise_median <= 0.0:
        raise ValueError("Noise-window median envelope is not finite and positive.")

    eps = LOG_EPS_FRACTION * noise_median
    eps = max(eps, np.finfo(float).eps)
    return np.log((smoothed + eps) / (noise_median + eps)), noise_median


def detection_window_for_station(station: str, config: PickConfig) -> tuple[float, float]:
    return DETECTION_WINDOWS_S.get(config.kind, {}).get(station, (NOISE_END_S, END_OFFSET_S))


def gaussian_log_likelihood(values: np.ndarray) -> tuple[float, float, float]:
    finite = values[np.isfinite(values)]
    npts = finite.size
    if npts < 2:
        return -np.inf, np.nan, np.nan

    mean = float(np.nanmean(finite))
    variance = float(np.nanvar(finite))
    variance = max(variance, np.finfo(float).eps)
    log_likelihood = -0.5 * npts * (np.log(2.0 * np.pi * variance) + 1.0)
    return float(log_likelihood), mean, variance


def posterior_from_log_scores(log_scores: np.ndarray) -> np.ndarray:
    posterior = np.zeros_like(log_scores, dtype=float)
    finite = np.isfinite(log_scores)
    if not np.any(finite):
        return posterior

    shifted = log_scores[finite] - np.nanmax(log_scores[finite])
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return posterior

    posterior[finite] = weights / total
    return posterior


def posterior_interval(candidate_times: np.ndarray, posterior: np.ndarray, low: float = 0.05, high: float = 0.95):
    if candidate_times.size == 0 or np.sum(posterior) <= 0.0:
        return np.nan, np.nan

    order = np.argsort(candidate_times)
    sorted_times = candidate_times[order]
    sorted_posterior = posterior[order]
    cdf = np.cumsum(sorted_posterior)
    cdf /= cdf[-1]
    return (
        float(np.interp(low, cdf, sorted_times)),
        float(np.interp(high, cdf, sorted_times)),
    )


def offline_change_point(
    time_s: np.ndarray,
    log_envelope: np.ndarray,
    detection_start_s: float,
    detection_end_s: float,
    early_prior_decay_s: float,
) -> tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
    model_mask = (
        (time_s >= MODEL_START_S)
        & (time_s <= MODEL_END_S)
        & np.isfinite(log_envelope)
    )
    model_idx = np.flatnonzero(model_mask)
    if model_idx.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.array([]), np.array([])

    dt = float(np.nanmedian(np.diff(time_s)))
    min_pre_n = max(2, int(round(MIN_PRE_CHANGE_S / dt)))
    min_post_n = max(2, int(round(MIN_POST_CHANGE_S / dt)))

    candidate_idx = np.flatnonzero(
        (time_s >= detection_start_s)
        & (time_s <= detection_end_s)
        & np.isfinite(log_envelope)
    )
    valid_candidates = []
    log_scores = []
    pre_means = []
    post_means = []

    for idx in candidate_idx:
        pre_mask = model_mask & (time_s < time_s[idx])
        post_mask = model_mask & (time_s >= time_s[idx])
        if np.count_nonzero(pre_mask) < min_pre_n or np.count_nonzero(post_mask) < min_post_n:
            continue

        pre_ll, pre_mean, _ = gaussian_log_likelihood(log_envelope[pre_mask])
        post_ll, post_mean, _ = gaussian_log_likelihood(log_envelope[post_mask])
        if not np.isfinite(pre_ll + post_ll):
            continue

        log_prior = 0.0
        if early_prior_decay_s > 0.0:
            log_prior = -(time_s[idx] - detection_start_s) / early_prior_decay_s

        valid_candidates.append(idx)
        log_scores.append(pre_ll + post_ll + log_prior)
        pre_means.append(pre_mean)
        post_means.append(post_mean)

    if not valid_candidates:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.array([]), np.array([])

    valid_candidates = np.asarray(valid_candidates, dtype=int)
    log_scores = np.asarray(log_scores, dtype=float)
    pre_means = np.asarray(pre_means, dtype=float)
    post_means = np.asarray(post_means, dtype=float)
    posterior = posterior_from_log_scores(log_scores)

    best_local_idx = int(np.nanargmax(posterior))
    pick_s = float(time_s[valid_candidates[best_local_idx]])
    credible_start_s, credible_end_s = posterior_interval(time_s[valid_candidates], posterior)
    return (
        pick_s,
        credible_start_s,
        credible_end_s,
        float(posterior[best_local_idx]),
        float(post_means[best_local_idx] - pre_means[best_local_idx]),
        time_s[valid_candidates],
        posterior,
    )


def normalise_for_plot(data: np.ndarray) -> np.ndarray:
    scale = np.nanmax(np.abs(data))
    if not np.isfinite(scale) or scale <= 0.0:
        return data.astype(float)
    return data.astype(float) / scale


def pick_trace(station: str, config: PickConfig) -> PickResult:
    time_s, data, sampling_rate = read_filtered_trace(config, station)
    log_envelope, envelope_noise_median = log_smoothed_envelope(time_s, data, sampling_rate, config.smooth_s)
    detection_start_s, detection_end_s = detection_window_for_station(station, config)
    (
        pick_s,
        credible_start_s,
        credible_end_s,
        posterior_probability,
        amplitude_shift,
        candidate_times,
        posterior,
    ) = offline_change_point(
        time_s,
        log_envelope,
        detection_start_s,
        detection_end_s,
        config.early_prior_decay_s,
    )

    if np.isfinite(pick_s):
        pre_mask = (time_s >= MODEL_START_S) & (time_s < pick_s) & np.isfinite(log_envelope)
        post_mask = (time_s >= pick_s) & (time_s <= MODEL_END_S) & np.isfinite(log_envelope)
        pre_mean = float(np.nanmean(log_envelope[pre_mask]))
        post_mean = float(np.nanmean(log_envelope[post_mask]))
        status = "ok"
    else:
        pre_mean = np.nan
        post_mean = np.nan
        status = "no_pick"

    return PickResult(
        station=station,
        kind=config.kind,
        pick_s=pick_s,
        credible_start_s=credible_start_s,
        credible_end_s=credible_end_s,
        posterior_probability=posterior_probability,
        envelope_noise_median=envelope_noise_median,
        pre_mean=pre_mean,
        post_mean=post_mean,
        amplitude_shift=amplitude_shift,
        status=status,
        detection_start_s=detection_start_s,
        detection_end_s=detection_end_s,
        time_s=time_s,
        data=data,
        log_envelope=log_envelope,
        candidate_times=candidate_times,
        posterior=posterior,
        config=config,
    )


def results_to_table(results: list[PickResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "station": result.station,
                "kind": result.kind,
                "pick_s": result.pick_s,
                "credible_start_s": result.credible_start_s,
                "credible_end_s": result.credible_end_s,
                "posterior_probability": result.posterior_probability,
                "envelope_noise_median": result.envelope_noise_median,
                "pre_mean_log_normalized_envelope": result.pre_mean,
                "post_mean_log_normalized_envelope": result.post_mean,
                "amplitude_shift_log_units": result.amplitude_shift,
                "status": result.status,
                "freq_min_hz": result.config.freq_min,
                "freq_max_hz": result.config.freq_max,
                "smooth_s": result.config.smooth_s,
                "early_prior_decay_s": result.config.early_prior_decay_s,
                "reference_time_utc": str(REFERENCE_TIME_UTC),
                "absolute_start_time_utc": str(absolute_start_time()),
                "absolute_end_time_utc": str(absolute_end_time()),
                "mseed_path": str(result.config.mseed_path),
                "start_offset_s": START_OFFSET_S,
                "end_offset_s": END_OFFSET_S,
                "noise_start_s": NOISE_START_S,
                "noise_end_s": NOISE_END_S,
                "model_start_s": MODEL_START_S,
                "model_end_s": MODEL_END_S,
                "min_pre_change_s": MIN_PRE_CHANGE_S,
                "min_post_change_s": MIN_POST_CHANGE_S,
                "detection_start_s": result.detection_start_s,
                "detection_end_s": result.detection_end_s,
            }
        )
    return pd.DataFrame(rows)


def save_analysis_results(results: list[PickResult], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    metadata = []

    for idx, result in enumerate(results):
        prefix = f"result_{idx}_"
        arrays[prefix + "time_s"] = result.time_s
        arrays[prefix + "data"] = result.data
        arrays[prefix + "log_envelope"] = result.log_envelope
        arrays[prefix + "candidate_times"] = result.candidate_times
        arrays[prefix + "posterior"] = result.posterior
        metadata.append(
            {
                "station": result.station,
                "kind": result.kind,
                "pick_s": result.pick_s,
                "credible_start_s": result.credible_start_s,
                "credible_end_s": result.credible_end_s,
                "posterior_probability": result.posterior_probability,
                "envelope_noise_median": result.envelope_noise_median,
                "pre_mean": result.pre_mean,
                "post_mean": result.post_mean,
                "amplitude_shift": result.amplitude_shift,
                "status": result.status,
                "detection_start_s": result.detection_start_s,
                "detection_end_s": result.detection_end_s,
                "config": {
                    "kind": result.config.kind,
                    "mseed_path": str(result.config.mseed_path),
                    "freq_min": result.config.freq_min,
                    "freq_max": result.config.freq_max,
                    "smooth_s": result.config.smooth_s,
                    "early_prior_decay_s": result.config.early_prior_decay_s,
                },
            }
        )

    arrays["_metadata_json"] = np.array(json.dumps(metadata))
    np.savez_compressed(save_path, **arrays)


def load_analysis_results(load_path: Path) -> list[PickResult]:
    results = []
    with np.load(load_path, allow_pickle=False) as data:
        metadata = json.loads(data["_metadata_json"].item())
        for idx, item in enumerate(metadata):
            prefix = f"result_{idx}_"
            config_data = item["config"]
            config = PickConfig(
                kind=config_data["kind"],
                mseed_path=Path(config_data["mseed_path"]),
                freq_min=float(config_data["freq_min"]),
                freq_max=float(config_data["freq_max"]),
                smooth_s=float(config_data["smooth_s"]),
                early_prior_decay_s=float(config_data["early_prior_decay_s"]),
            )
            results.append(
                PickResult(
                    station=item["station"],
                    kind=item["kind"],
                    pick_s=float(item["pick_s"]),
                    credible_start_s=float(item["credible_start_s"]),
                    credible_end_s=float(item["credible_end_s"]),
                    posterior_probability=float(item["posterior_probability"]),
                    envelope_noise_median=float(item.get("envelope_noise_median", np.nan)),
                    pre_mean=float(item["pre_mean"]),
                    post_mean=float(item["post_mean"]),
                    amplitude_shift=float(item["amplitude_shift"]),
                    status=item["status"],
                    detection_start_s=float(item["detection_start_s"]),
                    detection_end_s=float(item["detection_end_s"]),
                    time_s=data[prefix + "time_s"],
                    data=data[prefix + "data"],
                    log_envelope=data[prefix + "log_envelope"],
                    candidate_times=data[prefix + "candidate_times"],
                    posterior=data[prefix + "posterior"],
                    config=config,
                )
            )
    return results


def save_diagnostic_figure(results: list[PickResult], save_path: Path, show: bool = False) -> None:
    nrows = len(STATIONS)
    fig, axes = plt.subplots(nrows, 2, figsize=(16, max(2.6 * nrows, 9)), sharex=True)
    axes = np.atleast_2d(axes)

    by_key = {(result.station, result.kind): result for result in results}
    for row, station in enumerate(STATIONS):
        for col, kind in enumerate(("seismic", "infrasound")):
            result = by_key[(station, kind)]
            ax = axes[row, col]
            ax_env = ax.twinx()

            station_color = STATION_COLORS.get(station, "0.15")
            ax.plot(result.time_s, normalise_for_plot(result.data), color=station_color, linewidth=0.9)
            ax_env.plot(result.time_s, result.log_envelope, color="black", linewidth=1.0, alpha=0.9)

            if np.isfinite(result.pick_s):
                ax.axvline(result.pick_s, color="black", linewidth=1.2)
                pre_mask = (
                    (result.time_s >= MODEL_START_S)
                    & (result.time_s < result.pick_s)
                    & np.isfinite(result.log_envelope)
                )
                post_mask = (
                    (result.time_s >= result.pick_s)
                    & (result.time_s <= MODEL_END_S)
                    & np.isfinite(result.log_envelope)
                )
                ax_env.plot(
                    result.time_s[pre_mask],
                    np.full(np.count_nonzero(pre_mask), result.pre_mean),
                    color="black",
                    linestyle="--",
                    linewidth=1.0,
                )
                ax_env.plot(
                    result.time_s[post_mask],
                    np.full(np.count_nonzero(post_mask), result.post_mean),
                    color="black",
                    linestyle="--",
                    linewidth=1.0,
                )

            ax.set_ylim(-1.15, 1.15)
            ax.set_xlim(-30, PLOT_END_S)
            ax.set_yticks([-1,0,1])
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, colors="black")
            ax_env.tick_params(axis="y", labelsize=TICK_FONTSIZE, colors="black")
            ax_env.set_ylim(-1,6)
            
            ax.grid(True, alpha=0.25)
            #ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
            ax.tick_params(axis="y", left=False)

            if row == 0:
                title = "(a) Seismic" if col == 0 else "(b) Infrasound"
                ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
            if col == 0:
                ax.set_ylabel(station, fontsize=STATION_LABEL_FONTSIZE)

            label = f"Pick = {result.pick_s:.2f} s" if np.isfinite(result.pick_s) else "Pick = none"
            ax.text(
                0.03,
                0.92,
                label,
                transform=ax.transAxes,
                va="top",
                fontsize=ANNOTATION_FONTSIZE,
                color="black",
                fontweight="bold",
            )

    reference_label = REFERENCE_TIME_UTC.strftime("%Y-%m-%d %H:%M:%S")
    time_label = f"Time (s) since {reference_label} UTC"
    axes[-1, 0].set_xlabel(time_label, fontsize=LABEL_FONTSIZE)
    axes[-1, 1].set_xlabel(time_label, fontsize=LABEL_FONTSIZE)
    fig.tight_layout()
    if SAVE_FIGURE:
        fig.savefig(save_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


def save_change_point_explanation_figure(results: list[PickResult], save_path: Path, show: bool = False) -> None:
    nrows = len(STATIONS)
    fig, axes = plt.subplots(nrows, 2, figsize=(14, max(2.8 * nrows, 9)), sharex=True)
    axes = np.atleast_2d(axes)

    by_key = {(result.station, result.kind): result for result in results}
    for row, station in enumerate(STATIONS):
        for col, kind in enumerate(("seismic", "infrasound")):
            result = by_key[(station, kind)]
            ax = axes[row, col]
            station_color = STATION_COLORS.get(station, "0.15")

            ax.plot(result.time_s, result.log_envelope, color=station_color, linewidth=1.0)

            if np.isfinite(result.pick_s):
                pre_mask = (
                    (result.time_s >= MODEL_START_S)
                    & (result.time_s < result.pick_s)
                    & np.isfinite(result.log_envelope)
                )
                post_mask = (
                    (result.time_s >= result.pick_s)
                    & (result.time_s <= MODEL_END_S)
                    & np.isfinite(result.log_envelope)
                )
                ax.plot(
                    result.time_s[pre_mask],
                    np.full(np.count_nonzero(pre_mask), result.pre_mean),
                    color="black",
                    linewidth=1.4,
                )
                ax.plot(
                    result.time_s[post_mask],
                    np.full(np.count_nonzero(post_mask), result.post_mean),
                    color="black",
                    linewidth=1.4,
                )
                ax.axvline(result.pick_s, color="black", linewidth=1.2)
                label = (
                    f"Pick = {result.pick_s:.2f} s\n"
                    f"pre mean = {result.pre_mean:.2f}\n"
                    f"post mean = {result.post_mean:.2f}"
                )
            else:
                label = "Pick = none"

            ax.set_xlim(PLOT_START_S, PLOT_END_S)
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
            if row == 0:
                title = "(a) Seismic" if col == 0 else "(b) Infrasound"
                ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{station}\nlog normalized envelope", fontsize=STATION_LABEL_FONTSIZE)
            ax.text(
                0.01,
                0.96,
                label,
                transform=ax.transAxes,
                va="top",
                fontsize=ANNOTATION_FONTSIZE,
                color="black",
            )

    reference_label = REFERENCE_TIME_UTC.strftime("%Y-%m-%d %H:%M:%S")
    time_label = f"Time (s) since {reference_label} UTC"
    axes[-1, 0].set_xlabel(time_label, fontsize=LABEL_FONTSIZE)
    axes[-1, 1].set_xlabel(time_label, fontsize=LABEL_FONTSIZE)
    fig.tight_layout()
    if SAVE_FIGURE:
        fig.savefig(save_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


def save_station_change_point_diagnostics(results: list[PickResult], output_dir: Path, show: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        station_color = STATION_COLORS.get(result.station, "0.15")

        axes[0].plot(result.time_s, normalise_for_plot(result.data), color=station_color, linewidth=0.9)
        axes[0].set_ylabel("Norm. amp.", fontsize=LABEL_FONTSIZE)
        axes[0].set_title(
            f"{result.station} {result.kind} offline change-point diagnostics",
            fontsize=TITLE_FONTSIZE,
            fontweight="bold",
        )

        axes[1].plot(result.time_s, result.log_envelope, color=station_color, linewidth=1.0)
        if np.isfinite(result.pick_s):
            pre_mask = (
                (result.time_s >= MODEL_START_S)
                & (result.time_s < result.pick_s)
                & np.isfinite(result.log_envelope)
            )
            post_mask = (
                (result.time_s >= result.pick_s)
                & (result.time_s <= MODEL_END_S)
                & np.isfinite(result.log_envelope)
            )
            axes[1].plot(
                result.time_s[pre_mask],
                np.full(np.count_nonzero(pre_mask), result.pre_mean),
                color="black",
                linewidth=1.4,
                label="pre/post means",
            )
            axes[1].plot(
                result.time_s[post_mask],
                np.full(np.count_nonzero(post_mask), result.post_mean),
                color="black",
                linewidth=1.4,
            )
        axes[1].set_ylabel("Log normalized envelope", fontsize=LABEL_FONTSIZE)

        if result.candidate_times.size:
            axes[2].plot(result.candidate_times, result.posterior, color="tab:orange", linewidth=1.2)
            if np.isfinite(result.credible_start_s) and np.isfinite(result.credible_end_s):
                axes[2].axvspan(result.credible_start_s, result.credible_end_s, color="tab:orange", alpha=0.18)
        axes[2].set_ylabel("Posterior", fontsize=LABEL_FONTSIZE)

        for ax in axes:
            ax.set_xlim(PLOT_START_S, PLOT_END_S)
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
            if np.isfinite(result.pick_s):
                ax.axvline(result.pick_s, color="black", linewidth=1.2)

        label = f"Pick = {result.pick_s:.2f} s" if np.isfinite(result.pick_s) else "Pick = none"
        axes[0].text(
            0.01,
            0.92,
            label,
            transform=axes[0].transAxes,
            va="top",
            fontsize=ANNOTATION_FONTSIZE,
        )

        reference_label = REFERENCE_TIME_UTC.strftime("%Y-%m-%d %H:%M:%S")
        axes[-1].set_xlabel(f"Time (s) since {reference_label} UTC", fontsize=LABEL_FONTSIZE)
        fig.tight_layout()

        save_path = output_dir / f"{result.station}_{result.kind}_offline_change_point_diagnostic.png"
        if SAVE_FIGURE:
            fig.savefig(save_path, dpi=300)
        if show:
            plt.show()
        else:
            plt.close(fig)


def run_analysis() -> list[PickResult]:
    input_files = ensure_input_files()
    run_configs = {
        "seismic": PickConfig(
            kind="seismic",
            mseed_path=input_files["seismic"],
            freq_min=SEISMIC_FREQ_MIN,
            freq_max=SEISMIC_FREQ_MAX,
            smooth_s=SEISMIC_SMOOTH_S,
            early_prior_decay_s=SEISMIC_EARLY_PRIOR_DECAY_S,
        ),
        "infrasound": PickConfig(
            kind="infrasound",
            mseed_path=input_files["infrasound"],
            freq_min=INFRASOUND_FREQ_MIN,
            freq_max=INFRASOUND_FREQ_MAX,
            smooth_s=INFRASOUND_SMOOTH_S,
            early_prior_decay_s=INFRASOUND_EARLY_PRIOR_DECAY_S,
        ),
    }

    results = []
    for station in STATIONS:
        for kind in ("seismic", "infrasound"):
            results.append(pick_trace(station, run_configs[kind]))

    table = results_to_table(results)
    table.to_csv(OUTPUT_CSV, index=False)
    if SAVE_ANALYSIS_RESULTS:
        save_analysis_results(results, OUTPUT_RESULTS_NPZ)

    print(f"Saved offline change-point picks to {OUTPUT_CSV}")
    if SAVE_ANALYSIS_RESULTS:
        print(f"Saved analysis variables to {OUTPUT_RESULTS_NPZ}")
    return results


def plot_results(results: list[PickResult]) -> None:
    save_diagnostic_figure(results, OUTPUT_FIG, show=SHOW_PLOTS)
    if SAVE_DIAGNOSTIC_FIGURES:
        save_change_point_explanation_figure(results, OUTPUT_DIAGNOSTIC_FIG, show=False)
        save_station_change_point_diagnostics(results, OUTPUT_STATION_DIAGNOSTIC_DIR, show=False)

    if SAVE_FIGURE:
        print(f"Saved diagnostic figure to {OUTPUT_FIG}")
    if SAVE_DIAGNOSTIC_FIGURES:
        print(f"Saved explanation figure to {OUTPUT_DIAGNOSTIC_FIG}")
        print(f"Saved station diagnostics to {OUTPUT_STATION_DIAGNOSTIC_DIR}")


def main() -> None:
    if RUN_ANALYSIS:
        results = run_analysis()
    else:
        results = load_analysis_results(OUTPUT_RESULTS_NPZ)
        print(f"Loaded analysis variables from {OUTPUT_RESULTS_NPZ}")

    if PLOT_RESULTS:
        plot_results(results)


if __name__ == "__main__":
    main()
