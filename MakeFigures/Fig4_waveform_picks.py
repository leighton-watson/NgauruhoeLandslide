"""
Plot seismic and infrasound waveforms with manual arrival picks.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import read


HERE = Path(__file__).resolve().parent

DATA_DIR = HERE.parent / "DATA"
ANALYSIS_CODES_DIR = HERE.parent / "AnalysisCodes"
SEISMIC_MSEED = DATA_DIR / "seismic_20260321_133600.mseed"
INFRASOUND_MSEED = DATA_DIR / "infrasound_20260321_133600.mseed"
PICKS_CSV = ANALYSIS_CODES_DIR / "picked_arrivals.csv"

START_OFFSET_S = 0.0
FILTER_END_OFFSET_S = 60.0
DISPLAY_END_OFFSET_S = 50.0

SEISMIC_FREQ_MIN = 1.0
SEISMIC_FREQ_MAX = 10.0
INFRASOUND_FREQ_MIN = 2.0
INFRASOUND_FREQ_MAX = 15.0

TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 18
TEXT_FONTSIZE = 20
SPACING = -2.0


def _load_picks(path: Path) -> pd.DataFrame:
    picks = pd.read_csv(path)
    if "use" in picks.columns:
        use_mask = picks["use"].astype(str).str.lower().eq("yes")
        picks = picks.loc[use_mask].copy()

    numeric_columns = [
        "seismic_arrival_s",
        "seismic_pick_1_s",
        "seismic_pick_2_s",
        "infrasound_arrival_s",
        "infrasound_pick_1_s",
        "infrasound_pick_2_s",
        "uncertainty_s",
    ]
    for column in numeric_columns:
        if column in picks.columns:
            picks[column] = pd.to_numeric(picks[column], errors="coerce")

    return picks.sort_values("station").reset_index(drop=True)


def _prepare_stream(path: Path, freq_min: float, freq_max: float):
    stream = read(str(path))
    stream = stream.copy()
    stream.trim(
        stream[0].stats.starttime + START_OFFSET_S,
        stream[0].stats.starttime + FILTER_END_OFFSET_S,
    )
    stream.detrend("demean")
    stream.detrend("linear")
    stream.taper(max_percentage=0.05)
    stream.filter("bandpass", freqmin=freq_min, freqmax=freq_max, corners=4, zerophase=True)
    stream.sort(keys=["station"])
    return stream


def _unique_station_order(stream):
    station_order = []
    for trace in stream:
        station = trace.stats.station
        if station not in station_order:
            station_order.append(station)
    return station_order


def _normalise(data):
    scale = np.max(np.abs(data))
    if not np.isfinite(scale) or scale == 0.0:
        return data.astype(float)
    return data.astype(float) / scale


def _trace_lookup(stream):
    lookup = {}
    for trace in stream:
        lookup.setdefault(trace.stats.station, trace)
    return lookup


def _format_pick_time(value):
    return f"{value:.2f} s"


def _label_pick(ax, pick_time, y_pos, *, ha, va="center", x_shift=0.0):
    return ax.text(
        pick_time + x_shift,
        y_pos,
        _format_pick_time(pick_time),
        color="black",
        fontsize=TEXT_FONTSIZE - 2,
        fontweight="bold",
        rotation=0,
        va=va,
        ha=ha,
    )


def _iter_picks_for_plot(pick_1, pick_2):
    picks = []
    if np.isfinite(pick_1):
        picks.append(float(pick_1))
    if np.isfinite(pick_2):
        picks.append(float(pick_2))
    return sorted(picks)


if __name__ == "__main__":
    picks = _load_picks(PICKS_CSV)
    st_seis = _prepare_stream(SEISMIC_MSEED, SEISMIC_FREQ_MIN, SEISMIC_FREQ_MAX)
    st_inf = _prepare_stream(INFRASOUND_MSEED, INFRASOUND_FREQ_MIN, INFRASOUND_FREQ_MAX)

    seismic_lookup = _trace_lookup(st_seis)
    infrasound_lookup = _trace_lookup(st_inf)

    # Match Fig2_waveforms.py by assigning colours from the seismic MiniSEED
    # file in ../DATA so station colours stay consistent across figures.
    color_reference_stream = _prepare_stream(SEISMIC_MSEED, SEISMIC_FREQ_MIN, SEISMIC_FREQ_MAX)
    seismic_station_order = _unique_station_order(color_reference_stream)
    station_order = [
        station for station in seismic_station_order
        if station in set(picks["station"]) and station in infrasound_lookup
    ]
    if not station_order:
        raise RuntimeError("No common stations were found between picks and waveform files.")

    cmap = plt.get_cmap("tab20")
    station_to_color = {
        station: cmap(i % cmap.N) for i, station in enumerate(seismic_station_order)
    }
    station_to_offset = {station: i * SPACING for i, station in enumerate(station_order)}

    fig, (ax_seis, ax_inf) = plt.subplots(1, 2, figsize=(18, 10), sharey=True)

    yticks = [station_to_offset[station] for station in station_order]
    ylabels = station_order

    for station in station_order:
        pick_row = picks.loc[picks["station"] == station].iloc[0]
        color = station_to_color[station]
        offset = station_to_offset[station]

        seis_trace = seismic_lookup[station]
        seis_time = seis_trace.times() + START_OFFSET_S
        seis_data = _normalise(seis_trace.data)
        ax_seis.plot(seis_time, seis_data + offset, linewidth=2.0, color=color)
        seis_pick_1 = pick_row.get("seismic_pick_1_s", np.nan)
        seis_pick_2 = pick_row.get("seismic_pick_2_s", np.nan)
        seismic_picks = _iter_picks_for_plot(seis_pick_1, seis_pick_2)
        for index, pick_time in enumerate(seismic_picks):
            ha = "right" if index == 0 else "left"
            x_shift = -0.2 if index == 0 else 0.2
            ax_seis.vlines(pick_time, offset - 0.7, offset + 0.7, color="black", linewidth=1.8)
            _label_pick(ax_seis, pick_time, offset + 0.8, ha=ha, x_shift=x_shift)

        inf_trace = infrasound_lookup[station]
        inf_time = inf_trace.times() + START_OFFSET_S
        inf_data = _normalise(inf_trace.data)
        ax_inf.plot(inf_time, inf_data + offset, linewidth=2.0, color=color)

        inf_pick_1 = pick_row.get("infrasound_pick_1_s", np.nan)
        inf_pick_2 = pick_row.get("infrasound_pick_2_s", np.nan)
        infrasound_picks = _iter_picks_for_plot(inf_pick_1, inf_pick_2)
        for index, pick_time in enumerate(infrasound_picks):
            ha = "right" if index == 0 else "left"
            x_shift = -0.2 if index == 0 else 0.2
            ax_inf.vlines(pick_time, offset - 0.7, offset + 0.7, color="black", linewidth=1.8)
            _label_pick(ax_inf, pick_time, offset + 0.8, ha=ha, x_shift=x_shift)

    for ax in (ax_seis, ax_inf):
        ax.set_xlim(START_OFFSET_S, DISPLAY_END_OFFSET_S)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)

    ax_seis.set_yticks(yticks)
    ax_seis.set_yticklabels(ylabels, fontsize=LABEL_FONTSIZE)
    ax_inf.set_yticks(yticks)
    ax_inf.set_yticklabels(ylabels, fontsize=LABEL_FONTSIZE)

    ax_seis.set_xlabel("Time (s) since 2026-03-21 13:36:05 (UTC)", fontsize=LABEL_FONTSIZE)
    ax_inf.set_xlabel("Time (s) since 2026-03-21 13:36:05 (UTC)", fontsize=LABEL_FONTSIZE)
    ax_seis.set_title("(a) Seismic", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax_inf.set_title("(b) Infrasound", fontsize=TITLE_FONTSIZE, fontweight="bold")

    ymax = -SPACING * 0.7
    ymin = (len(station_order) - 1) * SPACING + SPACING * 0.7
    ax_seis.set_ylim(ymin, ymax)

    fig.tight_layout()
    plt.show()
