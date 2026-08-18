from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from obspy import UTCDateTime, read, read_inventory
from pyproj import Transformer


COORD_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "lines.linewidth": 0.9,
})


@dataclass
class SearchGrid:
    x_idx: np.ndarray
    y_idx: np.ndarray
    x_grid: np.ndarray
    y_grid: np.ndarray
    z_grid: np.ndarray
    east_limits: tuple[float, float]
    north_limits: tuple[float, float]


def load_topography(topo_dir):
    topo_path = Path(topo_dir)
    data = np.load(topo_path / "topography_big.npz", mmap_mode="r")
    return {"X": data["X"], "Y": data["Y"], "C": data["C"]}


def _analytic_envelope(data):
    npts = data.size
    spectrum = np.fft.fft(data)
    h = np.zeros(npts)
    if npts % 2 == 0:
        h[0] = 1.0
        h[npts // 2] = 1.0
        h[1:npts // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(npts + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(spectrum * h))


def _moving_average(data, n):
    n = max(1, int(n))
    kernel = np.ones(n, dtype=float) / n
    return np.convolve(data, kernel, mode="same")


def _window_centres(npts, fs, win_length_s, win_step_s):
    win_len = max(3, int(round(win_length_s * fs)))
    win_step = max(1, int(round(win_step_s * fs)))
    centres = []
    start = 0
    while start + win_len <= npts:
        centres.append(start + win_len // 2)
        start += win_step
    return np.array(centres, dtype=int), win_len


def _extract_station_metadata(stations, coord_file):
    inventory = read_inventory(str(coord_file))
    sta_lookup = {}
    for net in inventory:
        for sta in net:
            if len(sta) > 0:
                cha = sta[0]
                lat = cha.latitude if cha.latitude is not None else sta.latitude
                lon = cha.longitude if cha.longitude is not None else sta.longitude
                elev = cha.elevation if cha.elevation is not None else sta.elevation
            else:
                lat, lon, elev = sta.latitude, sta.longitude, sta.elevation
            sta_lookup[sta.code] = (lat, lon, elev)

    rows = []
    for station in stations:
        lat, lon, elev = sta_lookup[station]
        east, north = COORD_TRANSFORMER.transform(lon, lat)
        rows.append({
            "station": station,
            "latitude": lat,
            "longitude": lon,
            "easting": east,
            "northing": north,
            "elevation": elev,
        })
    return pd.DataFrame(rows)


def load_event_data(
    stations,
    start_time,
    end_time,
    mseed_file,
    coord_file,
    freq_min=1.0,
    freq_max=10.0,
    win_length_s=5.0,
    win_step_s=2.0,
    envelope_smooth_s=1.0,
):
    if isinstance(start_time, str):
        start_time = UTCDateTime(start_time)
    if isinstance(end_time, str):
        end_time = UTCDateTime(end_time)

    stream = read(str(mseed_file))
    stream.trim(starttime=start_time, endtime=end_time)

    traces = []
    for station in stations:
        trace = stream.select(station=station)
        if len(trace) == 0:
            raise ValueError(f"Station '{station}' not found in waveform file.")
        tr = trace[0].copy()
        tr.filter("bandpass", freqmin=freq_min, freqmax=freq_max, corners=4, zerophase=True)
        traces.append(tr)

    fs_values = {float(tr.stats.sampling_rate) for tr in traces}
    if len(fs_values) != 1:
        raise ValueError("All stations must share a common sampling rate for the alternative methods.")
    fs = fs_values.pop()
    npts_values = {int(tr.stats.npts) for tr in traces}
    if len(npts_values) != 1:
        raise ValueError("All stations must share a common number of samples after trimming.")
    npts = npts_values.pop()

    time = np.arange(npts) / fs
    centres, win_len = _window_centres(npts, fs, win_length_s, win_step_s)

    filtered = np.vstack([tr.data.astype(float) for tr in traces])
    envelopes = np.vstack([
        _moving_average(_analytic_envelope(row), int(round(envelope_smooth_s * fs)))
        for row in filtered
    ])

    rms = np.empty((len(stations), len(centres)))
    for i, row in enumerate(filtered):
        for j, centre in enumerate(centres):
            i0 = centre - win_len // 2
            i1 = i0 + win_len
            window = row[i0:i1]
            rms[i, j] = np.sqrt(np.mean(window ** 2))

    return {
        "traces": traces,
        "filtered": filtered,
        "envelopes": envelopes,
        "rms": rms,
        "time": time,
        "window_centres": centres,
        "window_times": time[centres],
        "window_length_s": win_length_s,
        "window_step_s": win_step_s,
        "sampling_rate": fs,
        "station_coords": _extract_station_metadata(stations, coord_file),
        "filter_params": {
            "freq_min": freq_min,
            "freq_max": freq_max,
            "win_length_s": win_length_s,
            "win_step_s": win_step_s,
            "envelope_smooth_s": envelope_smooth_s,
        },
    }


def build_search_grid(topo, crater_lat, crater_lon, search_area=5000, search_box=None, spatial_step=200):
    x = topo["X"][0, :]
    y = topo["Y"][:, 0]
    xcrater, ycrater = COORD_TRANSFORMER.transform(crater_lon, crater_lat)

    if search_box is None:
        east_min, east_max = xcrater - search_area, xcrater + search_area
        north_min, north_max = ycrater - search_area, ycrater + search_area
    else:
        east_min, east_max = search_box["east_min"], search_box["east_max"]
        north_min, north_max = search_box["north_min"], search_box["north_max"]

    dx = x[1] - x[0]
    dy = abs(y[1] - y[0])
    x_skip = max(1, int(round(spatial_step / dx)))
    y_skip = max(1, int(round(spatial_step / dy)))

    x_start = np.where(x >= east_min)[0][0]
    x_stop = np.searchsorted(x, east_max)
    y_stop = np.searchsorted(-y, -north_min)
    y_start = np.searchsorted(-y, -north_max)

    x_idx = np.arange(x_start, x_stop, x_skip)
    y_idx = np.arange(y_start, y_stop, y_skip)

    return SearchGrid(
        x_idx=x_idx,
        y_idx=y_idx,
        x_grid=topo["X"][y_idx, :][:, x_idx],
        y_grid=topo["Y"][y_idx, :][:, x_idx],
        z_grid=topo["C"][y_idx, :][:, x_idx],
        east_limits=(east_min, east_max),
        north_limits=(north_min, north_max),
    )


def _station_array(station_coords):
    return station_coords[["easting", "northing", "elevation"]].to_numpy(float)


def _best_locations(score_cube, search_grid):
    best_flat = np.argmax(score_cube.reshape(-1, score_cube.shape[-1]), axis=0)
    ny, nx = score_cube.shape[:2]
    iy, ix = np.unravel_index(best_flat, (ny, nx))
    best_x = search_grid.x_grid[iy, ix]
    best_y = search_grid.y_grid[iy, ix]
    best_z = search_grid.z_grid[iy, ix]
    best_score = score_cube[iy, ix, np.arange(score_cube.shape[-1])]
    return best_x, best_y, best_z, best_score


def amplitude_ratio_localization(data, topo, crater_lat, crater_lon, search_area=5000, search_box=None, spatial_step=200, decay_exponent=1.0):
    search_grid = build_search_grid(topo, crater_lat, crater_lon, search_area, search_box, spatial_step)
    stations = _station_array(data["station_coords"])
    amplitudes = np.maximum(data["rms"], 1e-12)
    pair_idx = np.array(list(combinations(range(amplitudes.shape[0]), 2)), dtype=int)
    observed = np.log(amplitudes[pair_idx[:, 0], :]) - np.log(amplitudes[pair_idx[:, 1], :])
    residual_scale = np.maximum(np.median(np.abs(observed), axis=0), 0.25)

    score = np.zeros((search_grid.x_grid.shape[0], search_grid.x_grid.shape[1], amplitudes.shape[1]))

    for iy in range(search_grid.x_grid.shape[0]):
        for ix in range(search_grid.x_grid.shape[1]):
            src = np.array([
                search_grid.x_grid[iy, ix],
                search_grid.y_grid[iy, ix],
                search_grid.z_grid[iy, ix],
            ])
            dist = np.sqrt(np.sum((stations - src) ** 2, axis=1))
            predicted = -decay_exponent * (np.log(dist[pair_idx[:, 0]]) - np.log(dist[pair_idx[:, 1]]))
            residual = observed - predicted[:, None]
            misfit = np.median(np.abs(residual), axis=0) / residual_scale
            score[iy, ix, :] = np.exp(-0.5 * misfit ** 2)

    best_x, best_y, best_z, best_score = _best_locations(score, search_grid)
    xcrater, ycrater = COORD_TRANSFORMER.transform(crater_lon, crater_lat)
    return {
        "method": "amplitude_ratio",
        "score": score,
        "best_x": best_x,
        "best_y": best_y,
        "best_z": best_z,
        "best_score": best_score,
        "window_times": data["window_times"],
        "x_offset": best_x - xcrater,
        "y_offset": best_y - ycrater,
        "search_grid": search_grid,
        "crater_xy": (xcrater, ycrater),
        "station_coords": data["station_coords"],
        "metadata": {"decay_exponent": decay_exponent},
    }


def _best_pair_lag(x_i, x_j, max_lag_samples):
    x_i = x_i - np.mean(x_i)
    x_j = x_j - np.mean(x_j)
    std_i = np.std(x_i)
    std_j = np.std(x_j)
    if std_i == 0 or std_j == 0:
        return np.nan, 0.0

    best_corr = -np.inf
    best_lag = 0
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        if lag < 0:
            a = x_i[:lag]
            b = x_j[-lag:]
        elif lag > 0:
            a = x_i[lag:]
            b = x_j[:-lag]
        else:
            a = x_i
            b = x_j
        if len(a) < 10:
            continue
        std_a = np.std(a)
        std_b = np.std(b)
        if std_a == 0 or std_b == 0:
            continue
        corr = np.dot(a, b) / (len(a) * std_a * std_b)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    return best_lag, best_corr


def estimate_pairwise_lags(data, max_lag_s=6.0, min_corr=0.35):
    fs = data["sampling_rate"]
    envelopes = data["envelopes"]
    centres = data["window_centres"]
    win_len = int(round(data["window_length_s"] * fs))
    max_lag_samples = int(round(max_lag_s * fs))
    pair_idx = np.array(list(combinations(range(envelopes.shape[0]), 2)), dtype=int)

    lags = np.full((pair_idx.shape[0], len(centres)), np.nan)
    corr = np.zeros((pair_idx.shape[0], len(centres)))

    for iw, centre in enumerate(centres):
        i0 = centre - win_len // 2
        i1 = i0 + win_len
        for ip, (i, j) in enumerate(pair_idx):
            lag_samples, peak_corr = _best_pair_lag(envelopes[i, i0:i1], envelopes[j, i0:i1], max_lag_samples)
            corr[ip, iw] = peak_corr
            if peak_corr >= min_corr:
                lags[ip, iw] = lag_samples / fs

    return {"pair_idx": pair_idx, "lags": lags, "correlation": corr}


def tdoa_envelope_localization(
    data,
    topo,
    crater_lat,
    crater_lon,
    search_area=5000,
    search_box=None,
    spatial_step=200,
    velocity=1500.0,
    max_lag_s=6.0,
    min_corr=0.35,
):
    lag_info = estimate_pairwise_lags(data, max_lag_s=max_lag_s, min_corr=min_corr)
    search_grid = build_search_grid(topo, crater_lat, crater_lon, search_area, search_box, spatial_step)
    stations = _station_array(data["station_coords"])
    pair_idx = lag_info["pair_idx"]
    observed = lag_info["lags"]

    score = np.zeros((search_grid.x_grid.shape[0], search_grid.x_grid.shape[1], observed.shape[1]))

    for iy in range(search_grid.x_grid.shape[0]):
        for ix in range(search_grid.x_grid.shape[1]):
            src = np.array([
                search_grid.x_grid[iy, ix],
                search_grid.y_grid[iy, ix],
                search_grid.z_grid[iy, ix],
            ])
            dist = np.sqrt(np.sum((stations - src) ** 2, axis=1))
            predicted = (dist[pair_idx[:, 0]] - dist[pair_idx[:, 1]]) / velocity
            residual = observed - predicted[:, None]
            valid = np.isfinite(residual)
            valid_counts = valid.sum(axis=0)
            residual = np.where(valid, residual, np.nan)
            rms = np.sqrt(np.nanmean(residual ** 2, axis=0))
            rms[valid_counts == 0] = np.inf
            score[iy, ix, :] = np.exp(-0.5 * (rms / 0.75) ** 2) * np.clip(valid_counts / max(1, pair_idx.shape[0]), 0.0, 1.0)

    best_x, best_y, best_z, best_score = _best_locations(score, search_grid)
    xcrater, ycrater = COORD_TRANSFORMER.transform(crater_lon, crater_lat)
    return {
        "method": "tdoa_envelope",
        "score": score,
        "best_x": best_x,
        "best_y": best_y,
        "best_z": best_z,
        "best_score": best_score,
        "window_times": data["window_times"],
        "x_offset": best_x - xcrater,
        "y_offset": best_y - ycrater,
        "search_grid": search_grid,
        "crater_xy": (xcrater, ycrater),
        "station_coords": data["station_coords"],
        "lag_info": lag_info,
        "metadata": {"velocity": velocity, "max_lag_s": max_lag_s, "min_corr": min_corr},
    }


def hybrid_localization(
    data,
    topo,
    crater_lat,
    crater_lon,
    search_area=5000,
    search_box=None,
    spatial_step=200,
    decay_exponent=1.0,
    velocity=1500.0,
    max_lag_s=6.0,
    min_corr=0.35,
    ratio_weight=0.4,
    tdoa_weight=0.6,
):
    ratio_result = amplitude_ratio_localization(
        data=data,
        topo=topo,
        crater_lat=crater_lat,
        crater_lon=crater_lon,
        search_area=search_area,
        search_box=search_box,
        spatial_step=spatial_step,
        decay_exponent=decay_exponent,
    )
    tdoa_result = tdoa_envelope_localization(
        data=data,
        topo=topo,
        crater_lat=crater_lat,
        crater_lon=crater_lon,
        search_area=search_area,
        search_box=search_box,
        spatial_step=spatial_step,
        velocity=velocity,
        max_lag_s=max_lag_s,
        min_corr=min_corr,
    )

    score = ratio_weight * ratio_result["score"] + tdoa_weight * tdoa_result["score"]
    best_x, best_y, best_z, best_score = _best_locations(score, ratio_result["search_grid"])
    xcrater, ycrater = ratio_result["crater_xy"]
    return {
        "method": "hybrid",
        "score": score,
        "best_x": best_x,
        "best_y": best_y,
        "best_z": best_z,
        "best_score": best_score,
        "window_times": data["window_times"],
        "x_offset": best_x - xcrater,
        "y_offset": best_y - ycrater,
        "search_grid": ratio_result["search_grid"],
        "crater_xy": ratio_result["crater_xy"],
        "station_coords": data["station_coords"],
        "ratio_result": ratio_result,
        "tdoa_result": tdoa_result,
        "metadata": {
            "decay_exponent": decay_exponent,
            "velocity": velocity,
            "ratio_weight": ratio_weight,
            "tdoa_weight": tdoa_weight,
        },
    }


def seismoacoustic_differential_localization(
    arrival_picks,
    station_coords,
    topo,
    crater_lat,
    crater_lon,
    search_area=5000,
    search_box=None,
    spatial_step=200,
    seismic_velocity=3000.0,
    infrasound_velocity=330.0,
):
    if isinstance(arrival_picks, pd.DataFrame):
        picks = arrival_picks.copy()
    else:
        picks = pd.DataFrame(arrival_picks)

    required = {"station", "seismic_arrival_s", "infrasound_arrival_s"}
    missing = required - set(picks.columns)
    if missing:
        raise ValueError(f"Arrival picks are missing required columns: {sorted(missing)}")

    picks = picks.dropna(subset=["seismic_arrival_s", "infrasound_arrival_s"]).copy()
    if picks.empty:
        raise ValueError("No valid station picks remain after dropping missing arrival times.")

    picks["observed_dt_s"] = picks["infrasound_arrival_s"] - picks["seismic_arrival_s"]
    if "uncertainty_s" not in picks.columns:
        picks["uncertainty_s"] = 1.0
    picks["uncertainty_s"] = np.maximum(picks["uncertainty_s"].astype(float), 1e-3)

    merged = picks.merge(station_coords, on="station", how="inner")
    if len(merged) != len(picks):
        missing_meta = sorted(set(picks["station"]) - set(merged["station"]))
        raise ValueError(f"Missing station metadata for: {missing_meta}")

    search_grid = build_search_grid(topo, crater_lat, crater_lon, search_area, search_box, spatial_step)
    stations = _station_array(merged)
    observed = merged["observed_dt_s"].to_numpy(float)

    rms_grid = np.full(search_grid.x_grid.shape, np.nan)
    lag_grid = np.full(search_grid.x_grid.shape, np.nan)

    for iy in range(search_grid.x_grid.shape[0]):
        for ix in range(search_grid.x_grid.shape[1]):
            src = np.array([
                search_grid.x_grid[iy, ix],
                search_grid.y_grid[iy, ix],
                search_grid.z_grid[iy, ix],
            ])
            dist = np.sqrt(np.sum((stations - src) ** 2, axis=1))
            predicted = dist * (1.0 / infrasound_velocity - 1.0 / seismic_velocity)

            # Fit a constant source lag so the location is controlled by inter-station
            # differences rather than any absolute timing offset between signal types.
            fitted_lag = np.mean(observed - predicted)
            residual = observed - (predicted + fitted_lag)
            rms = np.sqrt(np.mean(residual ** 2))

            lag_grid[iy, ix] = fitted_lag
            rms_grid[iy, ix] = rms

    best_idx = np.unravel_index(np.nanargmin(rms_grid), rms_grid.shape)
    best_src = np.array([
        search_grid.x_grid[best_idx],
        search_grid.y_grid[best_idx],
        search_grid.z_grid[best_idx],
    ])
    best_dist = np.sqrt(np.sum((stations - best_src) ** 2, axis=1))
    best_predicted = best_dist * (1.0 / infrasound_velocity - 1.0 / seismic_velocity)
    best_lag = lag_grid[best_idx]
    best_residual = observed - (best_predicted + best_lag)

    xcrater, ycrater = COORD_TRANSFORMER.transform(crater_lon, crater_lat)
    station_results = merged.copy()
    station_results["predicted_dt_s"] = best_predicted + best_lag
    station_results["distance_m"] = best_dist
    station_results["residual_s"] = best_residual

    return {
        "method": "seismoacoustic_differential",
        "search_grid": search_grid,
        "rms_grid": rms_grid,
        "lag_grid": lag_grid,
        "best_x": float(best_src[0]),
        "best_y": float(best_src[1]),
        "best_z": float(best_src[2]),
        "best_rms_s": float(rms_grid[best_idx]),
        "best_source_lag_s": float(best_lag),
        "crater_xy": (xcrater, ycrater),
        "station_results": station_results,
        "metadata": {
            "seismic_velocity": seismic_velocity,
            "infrasound_velocity": infrasound_velocity,
        },
    }


def summarise_result(result, quality_threshold=0.55):
    good = result["best_score"] >= quality_threshold
    if np.any(good):
        mean_x = float(np.mean(result["best_x"][good]))
        mean_y = float(np.mean(result["best_y"][good]))
        mean_z = float(np.mean(result["best_z"][good]))
        mean_score = float(np.mean(result["best_score"][good]))
        frac = float(np.mean(good))
    else:
        mean_x = float(np.mean(result["best_x"]))
        mean_y = float(np.mean(result["best_y"]))
        mean_z = float(np.mean(result["best_z"]))
        mean_score = float(np.mean(result["best_score"]))
        frac = 0.0

    return {
        "method": result["method"],
        "mean_easting": mean_x,
        "mean_northing": mean_y,
        "mean_elevation": mean_z,
        "mean_score": mean_score,
        "good_fraction": frac,
    }


def plot_result(result, topo, plot_time_min=None, plot_time_max=None, quality_threshold=0.55, num_frames=12, save_fig=None):
    X, Y, C = topo["X"], topo["Y"], topo["C"]
    t = result["window_times"]
    good = result["best_score"] >= quality_threshold
    search_grid = result["search_grid"]
    xcrater, ycrater = result["crater_xy"]

    if plot_time_min is None:
        plot_time_min = float(t[0])
    if plot_time_max is None:
        plot_time_max = float(t[-1])

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.2], hspace=0.35, wspace=0.25)
    ax_q = fig.add_subplot(gs[0, 0])
    ax_e = fig.add_subplot(gs[1, 0])
    ax_n = fig.add_subplot(gs[2, 0])
    ax_m = fig.add_subplot(gs[:, 1])

    ax_q.plot(t, result["best_score"], "o-", color="tab:purple", markersize=4)
    ax_q.axhline(quality_threshold, color="red", linestyle="--", linewidth=1.2)
    ax_q.set_ylabel("Score")
    ax_q.set_title(f"{result['method']} quality")
    ax_q.grid(True, alpha=0.3)
    ax_q.tick_params(labelbottom=False)

    ax_e.plot(t, result["x_offset"], "o-", color="tab:orange", markersize=4)
    ax_e.scatter(t[good], result["x_offset"][good], color="green", edgecolor="black", linewidth=0.4, zorder=5)
    ax_e.axhline(0, color="0.5", linestyle=":")
    ax_e.set_ylabel("Delta Easting (m)")
    ax_e.grid(True, alpha=0.3)
    ax_e.tick_params(labelbottom=False)

    ax_n.plot(t, result["y_offset"], "o-", color="tab:green", markersize=4)
    ax_n.scatter(t[good], result["y_offset"][good], color="green", edgecolor="black", linewidth=0.4, zorder=5)
    ax_n.axhline(0, color="0.5", linestyle=":")
    ax_n.set_ylabel("Delta Northing (m)")
    ax_n.set_xlabel("Time (s)")
    ax_n.grid(True, alpha=0.3)

    ax_m.contour(X, Y, C, colors="k", levels=18, linewidths=0.45, alpha=0.35)
    ax_m.set_xlim(search_grid.x_grid.min(), search_grid.x_grid.max())
    ax_m.set_ylim(search_grid.y_grid.min(), search_grid.y_grid.max())
    ax_m.set_aspect("equal", adjustable="box")

    station_coords = result["station_coords"]
    ax_m.scatter(
        station_coords["easting"],
        station_coords["northing"],
        c="red",
        marker="v",
        s=42,
        edgecolor="black",
        linewidths=0.5,
        zorder=10,
        label="Stations",
    )
    ax_m.scatter(xcrater, ycrater, marker="X", s=110, color="tab:blue", edgecolor="black", linewidths=0.8, label="Crater", zorder=12)

    mask = (t >= plot_time_min) & (t <= plot_time_max)
    idx = np.where(mask)[0]
    if idx.size > 0:
        chosen = np.unique(np.linspace(idx[0], idx[-1], min(num_frames, idx.size), dtype=int))
        norm = Normalize(vmin=plot_time_min, vmax=plot_time_max)
        cmap = plt.cm.plasma
        for k in chosen:
            if result["best_score"][k] < quality_threshold:
                continue
            ax_m.scatter(
                result["best_x"][k],
                result["best_y"][k],
                s=55,
                c=[cmap(norm(t[k]))],
                edgecolor="black",
                linewidth=0.4,
                zorder=8,
            )
        plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax_m, label="Time (s)", pad=0.02)

    ax_m.set_xlabel("Easting (m)")
    ax_m.set_ylabel("Northing (m)")
    ax_m.set_title(f"Best locations: {result['method']}")
    ax_m.legend(loc="upper right", fontsize=8)

    if save_fig:
        save_path = Path(save_fig)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def run_method(method, cfg):
    topo = load_topography(cfg["TOPOGRAPHY_DIR"])
    data = load_event_data(
        stations=cfg["STATIONS"],
        start_time=cfg["START_TIME"],
        end_time=cfg["END_TIME"],
        mseed_file=cfg["MSEED_FILE"],
        coord_file=cfg["STATION_COORDS_FILE"],
        freq_min=cfg.get("FREQ_MIN", 1.0),
        freq_max=cfg.get("FREQ_MAX", 10.0),
        win_length_s=cfg.get("WIN_LENGTH_S", 5.0),
        win_step_s=cfg.get("WIN_STEP_S", 2.0),
        envelope_smooth_s=cfg.get("ENVELOPE_SMOOTH_S", 1.0),
    )

    common = dict(
        data=data,
        topo=topo,
        crater_lat=cfg["CRATER_LAT"],
        crater_lon=cfg["CRATER_LON"],
        search_area=cfg.get("SEARCH_AREA", 5000),
        search_box=cfg.get("SEARCH_BOX"),
        spatial_step=cfg.get("SPATIAL_STEP", 200),
    )

    if method == "amplitude_ratio":
        result = amplitude_ratio_localization(
            **common,
            decay_exponent=cfg.get("DECAY_EXPONENT", 1.0),
        )
    elif method == "tdoa_envelope":
        result = tdoa_envelope_localization(
            **common,
            velocity=cfg.get("VELOCITY", 1500.0),
            max_lag_s=cfg.get("MAX_LAG_S", 6.0),
            min_corr=cfg.get("MIN_CORR", 0.35),
        )
    elif method == "hybrid":
        result = hybrid_localization(
            **common,
            decay_exponent=cfg.get("DECAY_EXPONENT", 1.0),
            velocity=cfg.get("VELOCITY", 1500.0),
            max_lag_s=cfg.get("MAX_LAG_S", 6.0),
            min_corr=cfg.get("MIN_CORR", 0.35),
            ratio_weight=cfg.get("RATIO_WEIGHT", 0.4),
            tdoa_weight=cfg.get("TDOA_WEIGHT", 0.6),
        )
    else:
        raise ValueError(f"Unknown method '{method}'.")

    return topo, data, result


# ============================================================================
# TEMPORARY COMPATIBILITY BRIDGE
# ============================================================================ 

from pathlib import Path as _CompatPath
import sys as _compat_sys

_COMPAT_ROOT = _CompatPath(__file__).resolve().parents[1]
if str(_COMPAT_ROOT) not in _compat_sys.path:
    _compat_sys.path.append(str(_COMPAT_ROOT))

from AnalysisCodes.sara import *  # noqa: F401,F403,E402
