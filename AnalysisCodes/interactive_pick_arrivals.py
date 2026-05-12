from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from obspy import read


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "DATA"
SEISMIC_MSEED = DATA_DIR / "seismic_20260321_133600.mseed"
INFRASOUND_MSEED = DATA_DIR / "infrasound_20260321_133600.mseed"
PICKS_CSV = HERE / "picked_arrivals.csv"

START_OFFSET_S = 0.0
END_OFFSET_S = 60

SEISMIC_FREQ_MIN = 1.0
SEISMIC_FREQ_MAX = 10.0
INFRASOUND_FREQ_MIN = 2.0
INFRASOUND_FREQ_MAX = 15.0

DEFAULT_STATIONS = ["OTVZ", "ETVZ", "NGZ", "NOVZ", "SNVZ"]


def _configure_matplotlib_backend():
    backend = matplotlib.get_backend().lower()
    if "inline" in backend:
        for candidate in ("QtAgg", "TkAgg"):
            try:
                matplotlib.use(candidate, force=True)
                print(f"Switched matplotlib backend from inline to {candidate} for interactive picking.")
                break
            except Exception:
                continue


_configure_matplotlib_backend()

import matplotlib.pyplot as plt


@dataclass
class StationData:
    station: str
    seismic_time: np.ndarray
    seismic_data: np.ndarray
    infrasound_time: np.ndarray
    infrasound_data: np.ndarray


def _load_pick_table(path: Path) -> pd.DataFrame:
    if path.exists():
        picks = pd.read_csv(path)
    else:
        picks = pd.DataFrame(columns=[
            "station", "seismic_arrival_s", "seismic_pick_1_s", "seismic_pick_2_s", "infrasound_arrival_s",
            "infrasound_pick_1_s", "infrasound_pick_2_s",
            "uncertainty_s", "use", "notes",
        ])

    for column, default in [
        ("seismic_arrival_s", np.nan),
        ("seismic_pick_1_s", np.nan),
        ("seismic_pick_2_s", np.nan),
        ("infrasound_arrival_s", np.nan),
        ("infrasound_pick_1_s", np.nan),
        ("infrasound_pick_2_s", np.nan),
        ("uncertainty_s", 1.5),
        ("use", "yes"),
        ("notes", ""),
    ]:
        if column not in picks.columns:
            picks[column] = default

    legacy_seismic_mask = (
        picks["seismic_pick_1_s"].isna()
        & picks["seismic_pick_2_s"].isna()
        & picks["seismic_arrival_s"].notna()
    )
    picks.loc[legacy_seismic_mask, "seismic_pick_1_s"] = picks.loc[legacy_seismic_mask, "seismic_arrival_s"]

    legacy_mask = (
        picks["infrasound_pick_1_s"].isna()
        & picks["infrasound_pick_2_s"].isna()
        & picks["infrasound_arrival_s"].notna()
    )
    picks.loc[legacy_mask, "infrasound_pick_1_s"] = picks.loc[legacy_mask, "infrasound_arrival_s"]

    _update_seismic_summary_columns(picks)
    _update_infrasound_summary_columns(picks)
    return picks


def _normalise(data: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(data))
    if not np.isfinite(scale) or scale == 0:
        return data.astype(float)
    return data.astype(float) / scale


def _update_infrasound_summary_columns(picks: pd.DataFrame):
    pick_1 = pd.to_numeric(picks["infrasound_pick_1_s"], errors="coerce")
    pick_2 = pd.to_numeric(picks["infrasound_pick_2_s"], errors="coerce")

    both_mask = pick_1.notna() & pick_2.notna()
    only_1_mask = pick_1.notna() & pick_2.isna()
    only_2_mask = pick_1.isna() & pick_2.notna()

    picks.loc[:, "infrasound_arrival_s"] = np.nan
    picks.loc[:, "uncertainty_s"] = pd.to_numeric(picks["uncertainty_s"], errors="coerce").fillna(1.5)

    picks.loc[both_mask, "infrasound_arrival_s"] = 0.5 * (pick_1[both_mask] + pick_2[both_mask])
    picks.loc[both_mask, "uncertainty_s"] = 0.5 * np.abs(pick_2[both_mask] - pick_1[both_mask])
    picks.loc[only_1_mask, "infrasound_arrival_s"] = pick_1[only_1_mask]
    picks.loc[only_2_mask, "infrasound_arrival_s"] = pick_2[only_2_mask]


def _update_seismic_summary_columns(picks: pd.DataFrame):
    pick_1 = pd.to_numeric(picks["seismic_pick_1_s"], errors="coerce")
    pick_2 = pd.to_numeric(picks["seismic_pick_2_s"], errors="coerce")

    both_mask = pick_1.notna() & pick_2.notna()
    only_1_mask = pick_1.notna() & pick_2.isna()
    only_2_mask = pick_1.isna() & pick_2.notna()

    picks.loc[:, "seismic_arrival_s"] = np.nan
    picks.loc[both_mask, "seismic_arrival_s"] = 0.5 * (pick_1[both_mask] + pick_2[both_mask])
    picks.loc[only_1_mask, "seismic_arrival_s"] = pick_1[only_1_mask]
    picks.loc[only_2_mask, "seismic_arrival_s"] = pick_2[only_2_mask]


def _read_filtered_trace(path: Path, station: str, freq_min: float, freq_max: float, start_offset_s: float, end_offset_s: float):
    stream = read(str(path))
    trace = stream.select(station=station)
    if len(trace) == 0:
        raise ValueError(f"Station '{station}' not found in {path.name}")

    tr = trace[0].copy()
    t0 = tr.stats.starttime + start_offset_s
    t1 = tr.stats.starttime + end_offset_s
    tr.trim(t0, t1)
    tr.detrend("demean")
    tr.detrend("linear")
    tr.taper(max_percentage=0.05)
    tr.filter("bandpass", freqmin=freq_min, freqmax=freq_max, corners=4, zerophase=True)

    time_s = np.arange(tr.stats.npts, dtype=float) / float(tr.stats.sampling_rate) + start_offset_s
    return time_s, tr.data.astype(float)


def load_station_data(stations: list[str]) -> list[StationData]:
    items: list[StationData] = []
    for station in stations:
        seismic_time, seismic_data = _read_filtered_trace(
            SEISMIC_MSEED, station, SEISMIC_FREQ_MIN, SEISMIC_FREQ_MAX, START_OFFSET_S, END_OFFSET_S,
        )
        infrasound_time, infrasound_data = _read_filtered_trace(
            INFRASOUND_MSEED, station, INFRASOUND_FREQ_MIN, INFRASOUND_FREQ_MAX, START_OFFSET_S, END_OFFSET_S,
        )
        items.append(StationData(
            station=station,
            seismic_time=seismic_time,
            seismic_data=_normalise(seismic_data),
            infrasound_time=infrasound_time,
            infrasound_data=_normalise(infrasound_data),
        ))
    return items


class InteractivePicker:
    def __init__(self, station_data: list[StationData], picks: pd.DataFrame, save_path: Path):
        self.station_data = station_data
        self.save_path = save_path
        self.fig, self.axes = plt.subplots(
            len(station_data), 2, figsize=(13, max(3.0 * len(station_data), 8)), sharex=False
        )
        self.axes = np.atleast_2d(self.axes)
        self.picks = picks.copy()
        self.station_to_index = {row.station: idx for idx, row in enumerate(station_data)}

        for station in [item.station for item in station_data]:
            if station not in set(self.picks["station"]):
                self.picks.loc[len(self.picks)] = [
                    station, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.5, "yes", ""
                ]

        self.picks = self.picks.drop_duplicates(subset=["station"], keep="first").reset_index(drop=True)
        self.lines = {}
        self.texts = {}

        self._draw()
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _pick_row(self, station: str) -> int:
        return int(self.picks.index[self.picks["station"] == station][0])

    def _draw(self):
        for row, item in enumerate(self.station_data):
            ax_seis = self.axes[row, 0]
            ax_inf = self.axes[row, 1]
            ax_seis.clear()
            ax_inf.clear()

            ax_seis.plot(item.seismic_time, item.seismic_data, color="black", linewidth=0.8)
            ax_inf.plot(item.infrasound_time, item.infrasound_data, color="black", linewidth=0.8)

            ax_seis.set_ylabel(item.station)
            ax_seis.set_title(f"{item.station} seismic ({SEISMIC_FREQ_MIN:.0f}-{SEISMIC_FREQ_MAX:.0f} Hz)")
            ax_inf.set_title(f"{item.station} infrasound ({INFRASOUND_FREQ_MIN:.0f}-{INFRASOUND_FREQ_MAX:.0f} Hz)")

            ax_seis.grid(True, alpha=0.3)
            ax_inf.grid(True, alpha=0.3)

            pick_row = self._pick_row(item.station)
            seis_pick_1 = self.picks.loc[pick_row, "seismic_pick_1_s"]
            seis_pick_2 = self.picks.loc[pick_row, "seismic_pick_2_s"]
            inf_pick_1 = self.picks.loc[pick_row, "infrasound_pick_1_s"]
            inf_pick_2 = self.picks.loc[pick_row, "infrasound_pick_2_s"]

            if np.isfinite(seis_pick_1):
                self.lines[(item.station, "seismic_pick_1")] = ax_seis.axvline(
                    seis_pick_1, color="tab:red", linewidth=1.2
                )
            if np.isfinite(seis_pick_2):
                self.lines[(item.station, "seismic_pick_2")] = ax_seis.axvline(
                    seis_pick_2, color="tab:orange", linewidth=1.2
                )
            if np.isfinite(seis_pick_1) and np.isfinite(seis_pick_2):
                lo = min(seis_pick_1, seis_pick_2)
                hi = max(seis_pick_1, seis_pick_2)
                ax_seis.axvspan(lo, hi, color="tab:red", alpha=0.12)
            if np.isfinite(inf_pick_1):
                self.lines[(item.station, "infrasound_pick_1")] = ax_inf.axvline(
                    inf_pick_1, color="tab:blue", linewidth=1.2
                )
            if np.isfinite(inf_pick_2):
                self.lines[(item.station, "infrasound_pick_2")] = ax_inf.axvline(
                    inf_pick_2, color="tab:cyan", linewidth=1.2
                )
            if np.isfinite(inf_pick_1) and np.isfinite(inf_pick_2):
                lo = min(inf_pick_1, inf_pick_2)
                hi = max(inf_pick_1, inf_pick_2)
                ax_inf.axvspan(lo, hi, color="tab:blue", alpha=0.12)

            self.texts[(item.station, "seismic")] = ax_seis.text(
                0.01, 0.92, self._pick_label(item.station, "seismic"), transform=ax_seis.transAxes, va="top",
            )
            self.texts[(item.station, "infrasound")] = ax_inf.text(
                0.01, 0.92, self._pick_label(item.station, "infrasound"), transform=ax_inf.transAxes, va="top",
            )

        self.axes[-1, 0].set_xlabel("Time since trace start (s)")
        self.axes[-1, 1].set_xlabel("Time since trace start (s)")
        self.fig.suptitle(
            "Click left panel for seismic picks and right panel for infrasound picks.\n"
            "Seismic uses two picks on the left panel. Infrasound uses two picks on the right panel.\n"
            "Use the matplotlib zoom tool first, then click to save picks. Press 's' to save, 'r' to reset panel pick.",
            y=0.995,
        )
        self.fig.tight_layout()

    def _pick_label(self, station: str, kind: str) -> str:
        pick_row = self._pick_row(station)
        if kind == "seismic":
            pick_1 = self.picks.loc[pick_row, "seismic_pick_1_s"]
            pick_2 = self.picks.loc[pick_row, "seismic_pick_2_s"]
            value = self.picks.loc[pick_row, "seismic_arrival_s"]
            if np.isfinite(pick_1) and np.isfinite(pick_2):
                return (
                    f"{kind}: {pick_1:.3f} s, {pick_2:.3f} s\n"
                    f"centre: {value:.3f} s"
                )
            if np.isfinite(pick_1):
                return f"{kind}: first pick {pick_1:.3f} s"
            if np.isfinite(pick_2):
                return f"{kind}: second pick {pick_2:.3f} s"
            if np.isfinite(value):
                return f"{kind}: {value:.3f} s"
            return f"{kind}: not picked"

        pick_1 = self.picks.loc[pick_row, "infrasound_pick_1_s"]
        pick_2 = self.picks.loc[pick_row, "infrasound_pick_2_s"]
        centre = self.picks.loc[pick_row, "infrasound_arrival_s"]
        uncertainty = self.picks.loc[pick_row, "uncertainty_s"]
        if np.isfinite(pick_1) and np.isfinite(pick_2):
            return (
                f"{kind}: {pick_1:.3f} s, {pick_2:.3f} s\n"
                f"centre: {centre:.3f} s, +/- {uncertainty:.3f} s"
            )
        if np.isfinite(pick_1):
            return f"{kind}: first pick {pick_1:.3f} s"
        if np.isfinite(pick_2):
            return f"{kind}: second pick {pick_2:.3f} s"
        return f"{kind}: not picked"

    def _set_pick(self, station: str, kind: str, xdata: float):
        pick_row = self._pick_row(station)
        if kind == "seismic":
            pick_1 = self.picks.loc[pick_row, "seismic_pick_1_s"]
            pick_2 = self.picks.loc[pick_row, "seismic_pick_2_s"]
            xdata = float(xdata)
            if not np.isfinite(pick_1):
                self.picks.loc[pick_row, "seismic_pick_1_s"] = xdata
            elif not np.isfinite(pick_2):
                self.picks.loc[pick_row, "seismic_pick_2_s"] = xdata
            else:
                self.picks.loc[pick_row, "seismic_pick_1_s"] = pick_2
                self.picks.loc[pick_row, "seismic_pick_2_s"] = xdata
        else:
            pick_1 = self.picks.loc[pick_row, "infrasound_pick_1_s"]
            pick_2 = self.picks.loc[pick_row, "infrasound_pick_2_s"]
            xdata = float(xdata)
            if not np.isfinite(pick_1):
                self.picks.loc[pick_row, "infrasound_pick_1_s"] = xdata
            elif not np.isfinite(pick_2):
                self.picks.loc[pick_row, "infrasound_pick_2_s"] = xdata
            else:
                self.picks.loc[pick_row, "infrasound_pick_1_s"] = pick_2
                self.picks.loc[pick_row, "infrasound_pick_2_s"] = xdata

        _update_seismic_summary_columns(self.picks)
        _update_infrasound_summary_columns(self.picks)
        self.picks.loc[pick_row, "use"] = "yes"

    def _reset_pick(self, station: str, kind: str):
        pick_row = self._pick_row(station)
        if kind == "seismic":
            self.picks.loc[pick_row, "seismic_pick_1_s"] = np.nan
            self.picks.loc[pick_row, "seismic_pick_2_s"] = np.nan
            _update_seismic_summary_columns(self.picks)
        else:
            self.picks.loc[pick_row, "infrasound_pick_1_s"] = np.nan
            self.picks.loc[pick_row, "infrasound_pick_2_s"] = np.nan
            _update_infrasound_summary_columns(self.picks)

    def on_click(self, event):
        if event.inaxes is None or event.xdata is None:
            return

        for row, item in enumerate(self.station_data):
            if event.inaxes == self.axes[row, 0]:
                self._set_pick(item.station, "seismic", event.xdata)
                self._draw()
                self.fig.canvas.draw_idle()
                return
            if event.inaxes == self.axes[row, 1]:
                self._set_pick(item.station, "infrasound", event.xdata)
                self._draw()
                self.fig.canvas.draw_idle()
                return

    def on_key(self, event):
        if event.key == "s":
            self.save()
        if event.key == "r" and event.inaxes is not None:
            for row, item in enumerate(self.station_data):
                if event.inaxes == self.axes[row, 0]:
                    self._reset_pick(item.station, "seismic")
                if event.inaxes == self.axes[row, 1]:
                    self._reset_pick(item.station, "infrasound")
            self._draw()
            self.fig.canvas.draw_idle()

    def save(self):
        _update_seismic_summary_columns(self.picks)
        _update_infrasound_summary_columns(self.picks)
        columns = [
            "station", "seismic_arrival_s", "seismic_pick_1_s", "seismic_pick_2_s", "infrasound_arrival_s",
            "infrasound_pick_1_s", "infrasound_pick_2_s",
            "uncertainty_s", "use", "notes",
        ]
        self.picks = self.picks[columns].sort_values("station").reset_index(drop=True)
        self.picks.to_csv(self.save_path, index=False)
        print(f"Saved picks to {self.save_path}")


if __name__ == "__main__":
    plt.ion()
    picks = _load_pick_table(PICKS_CSV)
    stations = [station for station in picks["station"].tolist() if station in DEFAULT_STATIONS]
    if not stations:
        stations = DEFAULT_STATIONS

    station_data = load_station_data(stations)
    picker = InteractivePicker(station_data, picks, PICKS_CSV)
    manager = plt.get_current_fig_manager()
    try:
        manager.window.showMaximized()
    except Exception:
        pass

    print("Interactive picker ready.")
    print("Zoom with the toolbar, click to pick, press 's' to save, press 'r' to clear the pick under the cursor.")
    print("Seismic and infrasound picks now each use two clicks to define a manual pick window.")
    print(f"If you are in Spyder, make sure the graphics backend is set to Qt or Tk, not Inline.")

    plt.show(block=True)
