"""
Amplitude Source Localisation core code and reusable result export helpers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Rectangle
from obspy import UTCDateTime, read, read_inventory
from pyproj import Transformer

from AnalysisCodes.io_utils import get_plot_window_seconds, save_results_bundle, subset_topography, traces_to_matrix


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
    "lines.linewidth": 0.8,
})


def load_topography(topo_dir):
    print(f"\n{'='*70}")
    print("Loading topography...")
    print(f"{'='*70}\n")

    topo_path = Path(topo_dir)
    data = np.load(topo_path / "topography.npz", mmap_mode="r")
    X, Y, C = data["X"], data["Y"], data["C"]

    print(f"  Topography grid : {X.shape[0]} x {X.shape[1]}")
    print(f"  X range         : {X.min():.1f} – {X.max():.1f} m")
    print(f"  Y range         : {Y.min():.1f} – {Y.max():.1f} m")
    print(f"  Elevation range : {C.min():.1f} – {C.max():.1f} m\n")

    return {"X": X, "Y": Y, "C": C}


def load_event_data(
    stations,
    start_time,
    end_time,
    mseed_file,
    coord_file,
    freq_min=1.0,
    freq_max=20.0,
    win_length_s=5.0,
    win_step_s=None,
    win_overlap=None,
):
    if isinstance(start_time, str):
        start_time = UTCDateTime(start_time)
    if isinstance(end_time, str):
        end_time = UTCDateTime(end_time)

    print(f"\n{'='*70}")
    print(f"Loading data  |  {len(stations)} stations")
    print(f"Time window   :  {start_time}  →  {end_time}")
    print(f"Filter        :  {freq_min}–{freq_max} Hz   |   RMS window: {win_length_s} s")
    print(f"{'='*70}\n")

    if win_step_s is None:
        if win_overlap is None:
            win_overlap = 0.9
        win_step_s = win_length_s * (1.0 - win_overlap)
    elif win_overlap is None and win_length_s > 0:
        win_overlap = 1.0 - (win_step_s / win_length_s)

    print(f"  Reading {mseed_file} ...")
    st_all = read(str(mseed_file))
    st_all.trim(starttime=start_time, endtime=end_time)
    available = sorted(set(tr.stats.station for tr in st_all))
    print(f"  Loaded {len(st_all)} traces  |  available stations: {available}\n")

    rms_data = []
    traces = []

    for station in stations:
        print(f"  Processing {station}...")
        st = st_all.select(station=station)
        if len(st) == 0:
            raise ValueError(
                f"No traces found for station '{station}' in {mseed_file}.\n"
                f"Available stations: {available}"
            )

        trace = st[0].copy()
        trace.filter("bandpass", freqmin=freq_min, freqmax=freq_max, corners=4, zerophase=True)
        traces.append(trace.copy())

        sampling_rate = float(trace.stats.sampling_rate)
        npts = int(trace.stats.npts)
        dt = 1.0 / sampling_rate
        time = np.arange(0, npts * dt, dt)[:npts]
        samples = trace.data.astype(float)

        win_len = int(round(sampling_rate * win_length_s))
        win_step = max(1, int(round(sampling_rate * win_step_s)))

        if win_len >= npts:
            tc = np.array([], dtype=float)
            rms_vals = np.array([], dtype=float)
        else:
            num_win = 1 + (npts - win_len) // win_step
            tc = np.empty(num_win, dtype=float)
            rms_vals = np.empty(num_win, dtype=float)
            for j in range(num_win):
                i0 = j * win_step
                i1 = i0 + win_len
                if i1 > npts:
                    tc = tc[:j]
                    rms_vals = rms_vals[:j]
                    break
                rms_vals[j] = np.sqrt(np.mean(samples[i0:i1] ** 2))
                tc[j] = time[i0 + win_len // 2]

        rms_data.append(np.column_stack([tc, rms_vals]))
        if len(tc) > 0:
            print(f"    → {len(tc)} RMS windows   ({tc[0]:.2f} – {tc[-1]:.2f} s)")
        else:
            print("    → 0 RMS windows")

    print(f"\n  Reading station metadata from {coord_file} ...")
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

    station_coords = []
    for station in stations:
        if station not in sta_lookup:
            raise ValueError(
                f"Station '{station}' not found in {coord_file}.\n"
                f"Available: {list(sta_lookup.keys())}"
            )
        lat, lon, elev = sta_lookup[station]
        easting, northing = COORD_TRANSFORMER.transform(lon, lat)
        station_coords.append({
            "station": station,
            "latitude": lat,
            "longitude": lon,
            "easting": easting,
            "northing": northing,
            "elevation": elev,
        })
        print(
            f"    {station:8s}  lat={lat:.4f}  lon={lon:.4f}  elev={elev:.1f} m  "
            f"→  E={easting:.1f}  N={northing:.1f}"
        )

    station_coords = pd.DataFrame(station_coords)

    tc_final = rms_data[0][:, 0]
    if len(tc_final) > 0:
        print(f"\n  Data loaded OK  |  time range: {tc_final[0]:.2f} – {tc_final[-1]:.2f} s\n")
    else:
        print("\n  Data loaded OK  |  no RMS windows available\n")

    return {
        "rms_data": rms_data,
        "traces": traces,
        "station_coords": station_coords,
        "filter_params": {
            "freq_min": freq_min,
            "freq_max": freq_max,
            "win_length_s": win_length_s,
            "win_step_s": win_step_s,
            "win_overlap": win_overlap,
        },
    }


def run_localization(
    data,
    topo,
    crater_lat,
    crater_lon,
    search_area=8000,
    search_box=None,
    spatial_step=160,
    temporal_step=1.0,
    b=1.0,
    store_full_fields=True,
):
    print(f"\n{'='*70}")
    print("Running source localisation")
    print(f"{'='*70}\n")

    rms_data = data["rms_data"]
    station_coords = data["station_coords"]
    X, Y, C = topo["X"], topo["Y"], topo["C"]

    xcrater, ycrater = COORD_TRANSFORMER.transform(crater_lon, crater_lat)
    print(f"  Crater   : {crater_lat}°N  {crater_lon}°E")
    print(f"  NZTM     : E={xcrater:.1f}  N={ycrater:.1f}")
    print(f"  Decay b  : {b}\n")

    stations = station_coords["station"].values
    easting = station_coords["easting"].values
    northing = station_coords["northing"].values
    elev = station_coords["elevation"].values
    sta_xyz = np.vstack((easting, northing, elev)).T

    rms_values = np.vstack([row[:, 1] for row in rms_data]).astype(float)
    tc = rms_data[0][:, 0].astype(float)
    u_rms = rms_values.T

    print(f"  Stations     : {len(stations)}")
    print(f"  Time points  : {len(tc)}  ({tc[0]:.2f} – {tc[-1]:.2f} s)")

    x = X[0, :]
    y = Y[:, 0]
    dx_grid = x[1] - x[0] if len(x) > 1 else 1.0
    dy_grid = abs(y[1] - y[0]) if len(y) > 1 else 1.0
    dx_skip = max(1, int(round(spatial_step / dx_grid)))
    dy_skip = max(1, int(round(spatial_step / dy_grid)))

    time_indices = []
    current_time = tc[0]
    for i, t_val in enumerate(tc):
        if t_val >= current_time:
            time_indices.append(i)
            current_time += temporal_step
    tvec = np.array(time_indices, dtype=int)

    print(f"  Spatial step : {spatial_step} m  (every {dx_skip} grid pts)")
    print(f"  Temporal step: {temporal_step} s  ({len(tvec)} slices)")

    if search_box is not None:
        east_min, east_max = search_box["east_min"], search_box["east_max"]
        north_min, north_max = search_box["north_min"], search_box["north_max"]
        print(f"\n  Custom box  –  E: {east_min:.0f}–{east_max:.0f} m   N: {north_min:.0f}–{north_max:.0f} m")
    else:
        north_min, north_max = ycrater - search_area, ycrater + search_area
        east_min, east_max = xcrater - search_area, xcrater + search_area
        print(
            f"\n  Square box ±{search_area} m  –  E: {east_min:.0f}–{east_max:.0f} m   "
            f"N: {north_min:.0f}–{north_max:.0f} m"
        )

    xidx1 = np.where(x >= east_min)[0][0]
    xidx2 = np.searchsorted(x, east_max)
    yidx2 = np.searchsorted(-y, -north_min)
    yidx1 = np.searchsorted(-y, -north_max)

    xvec = np.arange(xidx1, xidx2, dx_skip)
    yvec = np.arange(yidx1, yidx2, dy_skip)
    nt, nx, ny = len(tvec), len(xvec), len(yvec)
    print(f"\n  Search grid  : {nx} × {ny} × {nt} = {nx * ny * nt:,} computations\n")

    grid_x = X[np.ix_(yvec, xvec)].astype(float)
    grid_y = Y[np.ix_(yvec, xvec)].astype(float)
    grid_z = C[np.ix_(yvec, xvec)].astype(float)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])

    print("  Precomputing source-station geometry...")
    dist = np.sqrt(np.sum((grid_points[:, None, :] - sta_xyz[None, :, :]) ** 2, axis=2))
    logdist = np.log(np.maximum(dist, 1e-12))
    mean_logdist = np.mean(logdist, axis=1)
    x_centered = logdist - mean_logdist[:, None]
    x_ss = np.sum(x_centered ** 2, axis=1)[:, None]

    print("  Computing R²...")
    log_amplitude = np.log(np.maximum(u_rms[tvec, :], 1e-10))
    mean_log_amplitude = np.mean(log_amplitude, axis=1)
    y_centered = log_amplitude - mean_log_amplitude[:, None]
    sst = np.sum(y_centered ** 2, axis=1)
    cross = x_centered @ y_centered.T
    sse = sst[None, :] + 2.0 * b * cross + (b ** 2) * x_ss

    r2_flat = np.zeros((grid_points.shape[0], nt), dtype=float)
    valid_sst = sst > 0
    if np.any(valid_sst):
        r2_flat[:, valid_sst] = 1.0 - sse[:, valid_sst] / sst[None, valid_sst]
    print("    100.0%\n")

    intercept_flat = mean_log_amplitude[None, :] + b * mean_logdist[:, None]
    rmax = np.max(r2_flat, axis=0)
    near_max = r2_flat > (0.99 * rmax[None, :])
    near_max_count = near_max.sum(axis=0)
    valid_near_max = near_max_count > 0

    tplot = tc[tvec].astype(float)
    xplot = np.zeros(nt, dtype=float)
    yplot = np.zeros(nt, dtype=float)
    aplot = np.full(nt, 1e-8, dtype=float)
    if np.any(valid_near_max):
        xplot[valid_near_max] = (
            (near_max[:, valid_near_max] * grid_x.ravel()[:, None]).sum(axis=0)
            / near_max_count[valid_near_max]
        ) - xcrater
        yplot[valid_near_max] = (
            (near_max[:, valid_near_max] * grid_y.ravel()[:, None]).sum(axis=0)
            / near_max_count[valid_near_max]
        ) - ycrater
        aplot[valid_near_max] = np.exp(
            (near_max[:, valid_near_max] * intercept_flat[:, valid_near_max]).sum(axis=0)
            / near_max_count[valid_near_max]
        )

    r2 = None
    r2_norm = None
    if store_full_fields:
        r2 = r2_flat.reshape(ny, nx, nt)
        r2_norm = np.zeros_like(r2)
        valid_max = rmax > 0
        if np.any(valid_max):
            r2_norm[:, :, valid_max] = r2[:, :, valid_max] / rmax[None, None, valid_max]

    print(f"  Localisation done  |  max R² range: {rmax.min():.3f} – {rmax.max():.3f}\n")

    results = {
        "X": X,
        "Y": Y,
        "C": C,
        "xcrater": xcrater,
        "ycrater": ycrater,
        "tc": tc,
        "tvec": tvec,
        "tplot": tplot,
        "xplot": xplot,
        "yplot": yplot,
        "aplot": aplot,
        "R2_max": rmax,
        "rms": rms_data,
        "stations": stations,
        "easting": easting,
        "northing": northing,
        "elev": elev,
        "x_grid": grid_x,
        "y_grid": grid_y,
        "search_box_east": [east_min, east_max],
        "search_box_north": [north_min, north_max],
        "b": b,
        "filter_params": data["filter_params"],
    }
    if store_full_fields:
        results["R2"] = r2
        results["R2_norm"] = r2_norm
    return results


def plot_localization(
    results,
    traces,
    plot_time_min=None,
    plot_time_max=None,
    relative_threshold=0.99,
    absolute_threshold=0.7,
    num_frames=15,
    save_fig=None,
):
    print(f"\n{'='*70}")
    print("Generating plot...")
    print(f"{'='*70}\n")

    X, Y, C = results["X"], results["Y"], results["C"]
    xcrater, ycrater = results["xcrater"], results["ycrater"]
    tc = results["tc"]
    tplot, xplot, yplot = results["tplot"], results["xplot"], results["yplot"]
    if "R2_norm" not in results:
        raise ValueError(
            "Full R² fields are not available in this result. "
            "Rerun ASL with STORE_FULL_FIELDS=True to use plot_localization()."
        )

    r2_norm = results["R2_norm"]
    r2_max = results["R2_max"]
    rms = results["rms"]
    stations = results["stations"]
    easting, northing = results["easting"], results["northing"]
    x_grid, y_grid = results["x_grid"], results["y_grid"]
    search_box_east = results["search_box_east"]
    search_box_north = results["search_box_north"]
    b = results["b"]
    filter_params = results["filter_params"]

    if plot_time_min is None:
        plot_time_min = tc[0]
    if plot_time_max is None:
        plot_time_max = tc[-1]

    mean_rms = np.array([np.mean(rms[i][:, 1]) for i in range(len(rms))])
    max_idx = np.argmax(mean_rms)
    trace = traces[max_idx]
    sampling_rate = float(trace.stats.sampling_rate)
    npts = int(trace.stats.npts)
    t_raw = np.arange(0, npts / sampling_rate, 1.0 / sampling_rate)[:npts]
    raw_data = trace.data.astype(float)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 3, width_ratios=[1.0, 1.45, 0.06], hspace=0.28, wspace=0.18)
    ax_wave = fig.add_subplot(gs[0, 0])
    ax_r2 = fig.add_subplot(gs[1, 0])
    ax_east = fig.add_subplot(gs[2, 0])
    ax_north = fig.add_subplot(gs[3, 0])
    ax_map = fig.add_subplot(gs[:, 1])
    cax = fig.add_subplot(gs[:, 2])

    good_mask = np.array(r2_max) > absolute_threshold

    ax_wave.plot(t_raw, raw_data, "k-", linewidth=0.3, alpha=0.7)
    ax_wave.axvspan(plot_time_min, plot_time_max, color="0.9", alpha=0.5, zorder=0)
    ax_wave.set_ylabel("Amplitude (m/s)")
    ax_wave.set_xlim(tc[0], tc[-1])
    ax_wave.grid(True, alpha=0.3)
    ax_wave.set_title(
        f"(a) Filtered: {stations[max_idx]} "
        f"({filter_params['freq_min']:.0f}–{filter_params['freq_max']:.0f} Hz)"
    )
    ax_wave.tick_params(labelbottom=False)

    ax_rms = ax_wave.twinx()
    ax_rms.plot(rms[max_idx][:, 0], rms[max_idx][:, 1], "r-", linewidth=1.5)
    ax_rms.set_ylabel("RMS", color="r")
    ax_rms.tick_params(axis="y", labelcolor="r")

    ax_r2.plot(tplot, r2_max, "o-", color="tab:purple", markersize=4, linewidth=1.5)
    ax_r2.axhline(absolute_threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold = {absolute_threshold}")
    ax_r2.axvspan(plot_time_min, plot_time_max, color="0.9", alpha=0.5, zorder=0)
    ax_r2.set_ylabel("Max R²")
    ax_r2.set_ylim(0, 1.0)
    ax_r2.set_xlim(tc[0], tc[-1])
    ax_r2.grid(True, alpha=0.3)
    ax_r2.set_title(f"(b) Localisation Quality  (b = {b})")
    ax_r2.legend(loc="upper right", fontsize=8)
    ax_r2.tick_params(labelbottom=False)
    if np.any(good_mask):
        ax_r2.fill_between(tplot, 0, r2_max, where=good_mask, alpha=0.2, color="green")

    ax_east.plot(tplot, xplot, "o-", color="tab:orange", markersize=4)
    ax_east.scatter(
        np.array(tplot)[good_mask],
        np.array(xplot)[good_mask],
        s=60,
        c="green",
        marker="o",
        edgecolor="black",
        linewidth=0.5,
        zorder=5,
        label=f"R² > {absolute_threshold}",
    )
    ax_east.axvspan(plot_time_min, plot_time_max, color="0.9", alpha=0.5)
    ax_east.set_ylabel("ΔEasting (m)")
    ax_east.set_xlim(tc[0], tc[-1])
    ax_east.grid(True, alpha=0.3)
    ax_east.set_title("(c) Source Easting offset from crater")
    ax_east.legend(loc="best", fontsize=8)
    ax_east.tick_params(labelbottom=False)

    ax_north.plot(tplot, yplot, "o-", color="tab:green", markersize=4)
    ax_north.scatter(
        np.array(tplot)[good_mask],
        np.array(yplot)[good_mask],
        s=60,
        c="green",
        marker="o",
        edgecolor="black",
        linewidth=0.5,
        zorder=5,
        label=f"R² > {absolute_threshold}",
    )
    ax_north.axvspan(plot_time_min, plot_time_max, color="0.9", alpha=0.5)
    ax_north.set_ylabel("ΔNorthing (m)")
    ax_north.set_xlabel("Time (s)")
    ax_north.set_xlim(tc[0], tc[-1])
    ax_north.grid(True, alpha=0.3)
    ax_north.set_title("(d) Source Northing offset from crater")
    ax_north.legend(loc="best", fontsize=8)

    ax_map.contour(X, Y, C, colors="k", levels=15, linewidths=0.5, alpha=0.4)
    ax_map.set_xlim(x_grid.min(), x_grid.max())
    ax_map.set_ylim(y_grid.min(), y_grid.max())
    ax_map.set_aspect("equal", adjustable="box")

    for sx, sy in zip(easting, northing):
        if x_grid.min() <= sx <= x_grid.max() and y_grid.min() <= sy <= y_grid.max():
            ax_map.scatter(sx, sy, c="red", marker="v", s=40, edgecolor="black", linewidths=0.5, zorder=10)

    ax_map.scatter(xcrater, ycrater, marker="X", s=100, color="tab:blue", edgecolor="black", linewidths=1, label="Crater", zorder=20)

    rect = Rectangle(
        (search_box_east[0], search_box_north[0]),
        search_box_east[1] - search_box_east[0],
        search_box_north[1] - search_box_north[0],
        linewidth=2,
        edgecolor="cyan",
        facecolor="none",
        linestyle="--",
        label="Search Area",
        zorder=15,
    )
    ax_map.add_patch(rect)

    tmin = np.searchsorted(tplot, plot_time_min)
    tmax = np.searchsorted(tplot, plot_time_max)
    frame_indices = np.linspace(tmin, tmax, num_frames, dtype=int)

    time_cmap = plt.cm.plasma
    time_norm = Normalize(vmin=plot_time_min, vmax=plot_time_max)

    n_plotted = 0
    for frame in frame_indices:
        if frame >= r2_norm.shape[2]:
            continue
        if r2_max[frame] < absolute_threshold:
            continue
        rgba = list(time_cmap(time_norm(tplot[frame])))
        rgba[3] = 0.7
        mask = r2_norm[:, :, frame] >= relative_threshold
        Z = np.ma.masked_where(~mask, r2_norm[:, :, frame])
        ax_map.pcolormesh(
            x_grid,
            y_grid,
            Z,
            cmap=ListedColormap([tuple(rgba)]),
            shading="nearest",
            zorder=5,
            rasterized=True,
        )
        n_plotted += 1

    print(f"  Plotted {n_plotted}/{len(frame_indices)} time slices (R² > {absolute_threshold})")

    sm = ScalarMappable(cmap=time_cmap, norm=time_norm)
    fig.colorbar(sm, cax=cax, label="Time (s)")

    ax_map.set_xlabel("Easting (m)")
    ax_map.set_ylabel("Northing (m)")
    ax_map.set_title(
        f"(e) Source Locations\n"
        f"(R² > {relative_threshold}×max,  max R² > {absolute_threshold})"
    )
    ax_map.legend(loc="upper right", fontsize=8)

    plt.suptitle("Seismic Source Localisation", fontsize=13, y=0.98)

    if save_fig:
        Path(save_fig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig, bbox_inches="tight")
        print(f"  Figure saved → {save_fig}")

    print("  Plot complete\n")
    plt.show()
    return fig


def save_kml(results, save_path, absolute_threshold=0.5, start_time=None):
    from pyproj import Transformer
    import colorsys

    print(f"\n{'='*70}")
    print("Saving KML...")
    print(f"{'='*70}\n")

    to_wgs84 = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)

    tplot = results["tplot"]
    xplot = results["xplot"]
    yplot = results["yplot"]
    r2_max = results["R2_max"]
    xcrater = results["xcrater"]
    ycrater = results["ycrater"]
    easting = results["easting"]
    northing = results["northing"]
    stations = results["stations"]

    src_easting = [xplot[k] + xcrater for k in range(len(tplot))]
    src_northing = [yplot[k] + ycrater for k in range(len(tplot))]

    abs_time = UTCDateTime(start_time) if start_time is not None else None

    good = [k for k in range(len(tplot)) if r2_max[k] > absolute_threshold]
    print(f"  {len(good)} of {len(tplot)} time steps exceed R² threshold ({absolute_threshold})")

    def r2_to_kml_colour(r2_val, r2_min, r2_max_val):
        norm = (r2_val - r2_min) / (r2_max_val - r2_min) if r2_max_val > r2_min else 0.5
        hue = 0.667 * (1.0 - norm)
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return f"ff{int(b * 255):02x}{int(g * 255):02x}{int(r * 255):02x}"

    r2_vals = [r2_max[k] for k in good]
    r2_lo = min(r2_vals) if r2_vals else 0
    r2_hi = max(r2_vals) if r2_vals else 1

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "<Document>",
        "  <name>ASL Source Locations</name>",
        "  <description>Amplitude Source Localisation results</description>",
        "",
        "  <!-- Styles for source location points -->",
    ]

    for k in good:
        colour = r2_to_kml_colour(r2_max[k], r2_lo, r2_hi)
        lines += [
            f'  <Style id="src_{k}">',
            "    <IconStyle>",
            f"      <color>{colour}</color>",
            "      <scale>0.8</scale>",
            "      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>",
            "    </IconStyle>",
            "    <LabelStyle><scale>0.7</scale></LabelStyle>",
            "  </Style>",
        ]

    lines += [
        '  <Style id="crater">',
        "    <IconStyle>",
        "      <color>ffff0000</color>",
        "      <scale>1.2</scale>",
        "      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/target.png</href></Icon>",
        "    </IconStyle>",
        "    <LabelStyle><scale>0.9</scale></LabelStyle>",
        "  </Style>",
        '  <Style id="station">',
        "    <IconStyle>",
        "      <color>ff00ff00</color>",
        "      <scale>0.9</scale>",
        "      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/triangle.png</href></Icon>",
        "    </IconStyle>",
        "    <LabelStyle><scale>0.8</scale></LabelStyle>",
        "  </Style>",
        "",
        "  <Folder>",
        "    <name>Source Locations</name>",
    ]

    for k in good:
        lon, lat = to_wgs84.transform(src_easting[k], src_northing[k])
        t_sec = tplot[k]
        r2_val = r2_max[k]
        name = f"t={t_sec:.1f}s  R²={r2_val:.3f}"
        desc = (
            f"Time offset: {t_sec:.2f} s\n"
            f"Max R²: {r2_val:.4f}\n"
            f"Lat: {lat:.6f}\n"
            f"Lon: {lon:.6f}"
        )

        lines.append("    <Placemark>")
        lines.append(f"      <name>{name}</name>")
        lines.append(f"      <description>{desc}</description>")
        lines.append(f"      <styleUrl>#src_{k}</styleUrl>")
        if abs_time is not None:
            ts = abs_time + t_sec
            lines.append(f'      <TimeStamp><when>{ts.strftime("%Y-%m-%dT%H:%M:%S")}Z</when></TimeStamp>')
        lines += [
            "      <Point>",
            f"        <coordinates>{lon:.8f},{lat:.8f},0</coordinates>",
            "      </Point>",
            "    </Placemark>",
        ]

    crater_lon_wgs, crater_lat_wgs = to_wgs84.transform(xcrater, ycrater)
    lines += [
        "  </Folder>",
        "",
        "  <Folder>",
        "    <name>Reference Points</name>",
        "    <Placemark>",
        "      <name>Crater</name>",
        "      <styleUrl>#crater</styleUrl>",
        f"      <Point><coordinates>{crater_lon_wgs:.8f},{crater_lat_wgs:.8f},0</coordinates></Point>",
        "    </Placemark>",
    ]

    for sta, east, north in zip(stations, easting, northing):
        lon, lat = to_wgs84.transform(east, north)
        lines += [
            "    <Placemark>",
            f"      <name>{sta}</name>",
            "      <styleUrl>#station</styleUrl>",
            f"      <Point><coordinates>{lon:.8f},{lat:.8f},0</coordinates></Point>",
            "    </Placemark>",
        ]

    lines += ["  </Folder>", "", "</Document>", "</kml>"]

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fobj:
        fobj.write("\n".join(lines))

    print(f"  KML saved → {save_path}")
    print(f"  Contains {len(good)} source points, 1 crater, {len(stations)} stations\n")


def build_output_bundle(cfg, topo, data, results):
    filtered_time, filtered_waveforms = traces_to_matrix(data["traces"])
    rms_time = data["rms_data"][0][:, 0].astype(float)
    rms_values = np.vstack([row[:, 1] for row in data["rms_data"]]).astype(float)
    plot_start_s, plot_end_s = get_plot_window_seconds(cfg)

    station_coords = data["station_coords"]
    map_topo = subset_topography(topo, results["search_box_east"], results["search_box_north"])

    return {
        "analysis_name": "ASL",
        "quality_label": "Max R^2",
        "quality_description": "Coefficient of determination from the amplitude-decay fit at each time step.",
        "start_time": str(cfg["START_TIME"]),
        "end_time": str(cfg["END_TIME"]),
        "plot_start_time": str(cfg.get("PLOT_START_TIME", cfg["START_TIME"])),
        "plot_end_time": str(cfg.get("PLOT_END_TIME", cfg["END_TIME"])),
        "plot_window_start_s": float(plot_start_s),
        "plot_window_end_s": float(plot_end_s),
        "config": cfg,
        "filter_params": data["filter_params"],
        "analysis_params": {
            "b": results["b"],
            "temporal_step": cfg.get("TEMPORAL_STEP", 1.0),
            "spatial_step": cfg.get("SPATIAL_STEP", 160),
        },
        "station_names": station_coords["station"].to_numpy(dtype=str),
        "station_latitude": station_coords["latitude"].to_numpy(dtype=float),
        "station_longitude": station_coords["longitude"].to_numpy(dtype=float),
        "station_easting": station_coords["easting"].to_numpy(dtype=float),
        "station_northing": station_coords["northing"].to_numpy(dtype=float),
        "station_elevation": station_coords["elevation"].to_numpy(dtype=float),
        "filtered_time": filtered_time.astype(float),
        "filtered_waveforms": filtered_waveforms.astype(float),
        "rms_time": rms_time,
        "rms_values": rms_values,
        "result_time": np.asarray(results["tplot"], dtype=float),
        "quality": np.asarray(results["R2_max"], dtype=float),
        "x_offset": np.asarray(results["xplot"], dtype=float),
        "y_offset": np.asarray(results["yplot"], dtype=float),
        "x_absolute": np.asarray(results["xplot"], dtype=float) + float(results["xcrater"]),
        "y_absolute": np.asarray(results["yplot"], dtype=float) + float(results["ycrater"]),
        "crater_easting": np.array([results["xcrater"]], dtype=float),
        "crater_northing": np.array([results["ycrater"]], dtype=float),
        "search_box_east": np.asarray(results["search_box_east"], dtype=float),
        "search_box_north": np.asarray(results["search_box_north"], dtype=float),
        "map_X": map_topo["X"].astype(float),
        "map_Y": map_topo["Y"].astype(float),
        "map_C": map_topo["C"].astype(float),
    }


def save_output_bundle(save_path, cfg, topo, data, results):
    bundle = build_output_bundle(cfg, topo, data, results)
    return save_results_bundle(save_path, bundle)


def run_analysis(cfg):
    topo = load_topography(topo_dir=cfg["TOPOGRAPHY_DIR"])
    data = load_event_data(
        stations=cfg["STATIONS"],
        start_time=cfg["START_TIME"],
        end_time=cfg["END_TIME"],
        mseed_file=cfg["MSEED_FILE"],
        coord_file=cfg["STATION_COORDS_FILE"],
        freq_min=cfg.get("FREQ_MIN", 1.0),
        freq_max=cfg.get("FREQ_MAX", 20.0),
        win_length_s=cfg.get("WIN_LENGTH_S", 5.0),
        win_step_s=cfg.get("WIN_STEP_S", None),
        win_overlap=cfg.get("WIN_OVERLAP", None),
    )

    results = run_localization(
        data=data,
        topo=topo,
        crater_lat=cfg["CRATER_LAT"],
        crater_lon=cfg["CRATER_LON"],
        search_area=cfg.get("SEARCH_AREA", 8000),
        search_box=cfg.get("SEARCH_BOX", None),
        spatial_step=cfg.get("SPATIAL_STEP", 160),
        temporal_step=cfg.get("TEMPORAL_STEP", 1.0),
        b=cfg.get("B", 1.0),
        store_full_fields=cfg.get("STORE_FULL_FIELDS", True),
    )

    return topo, data, results


def run_pipeline(cfg):
    analysis_cfg = dict(cfg)
    analysis_cfg["STORE_FULL_FIELDS"] = True
    topo, data, results = run_analysis(analysis_cfg)

    fig = plot_localization(
        results=results,
        traces=data["traces"],
        plot_time_min=cfg.get("PLOT_TIME_MIN", None),
        plot_time_max=cfg.get("PLOT_TIME_MAX", None),
        relative_threshold=cfg.get("RELATIVE_THRESHOLD", 0.99),
        absolute_threshold=cfg.get("ABSOLUTE_THRESHOLD", 0.7),
        num_frames=cfg.get("NUM_FRAMES", 15),
        save_fig=cfg.get("SAVE_FIG", None),
    )

    if cfg.get("SAVE_KML"):
        save_kml(
            results=results,
            save_path=cfg["SAVE_KML"],
            absolute_threshold=cfg.get("ABSOLUTE_THRESHOLD", 0.7),
            start_time=cfg.get("START_TIME"),
        )

    return topo, data, results, fig
