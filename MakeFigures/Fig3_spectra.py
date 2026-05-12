"""
Create a 2x2 figure showing OTVZ seismic and infrasound spectrograms and
windowed spectra for the 2026-03-21 event.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.mlab import specgram as mlab_specgram
from obspy import Stream, UTCDateTime
from obspy.clients.fdsn import Client as FDSNClient


HERE = Path(__file__).resolve().parent

SAVE_FIG = HERE / "RESULTS" / "Fig3_spectra.pdf"

STATION = "OTVZ"
FDSN_BASE_URL = "https://service.geonet.org.nz"

DATA_START = UTCDateTime("2026-03-21T13:35:00")
DATA_END = UTCDateTime("2026-03-21T13:38:10")
PLOT_START = UTCDateTime("2026-03-21T13:35:05")
PLOT_END = UTCDateTime("2026-03-21T13:38:05")

WINDOW_LENGTH_S = 20.0
WINDOW_CENTERS = [
    UTCDateTime("2026-03-21T13:36:00"),
    UTCDateTime("2026-03-21T13:36:40"),
]
WINDOW_COLORS = ["tab:orange", "tab:blue"]
WINDOW_LABELS = [
    "20 s window centred at 13:36:00",
    "20 s window centred at 13:36:40",
]

SPEC_WINDOW_S = 8.0
SPEC_OVERLAP_FRAC = 0.50
SPECTRUM_FREQ_LIMS = (0.1, 40.0)
SEISMIC_FREQ_LIMS = (0.0, 40.0)
INFRASOUND_FREQ_LIMS = (0.0, 40.0)

SEISMIC_REFERENCE = 1e-9
INFRASOUND_REFERENCE = 20e-6

SEISMIC_GUIDE_LINES = (1.0, 10.0)
INFRASOUND_GUIDE_LINES = (2.0, 15.0)

COLORMAP = "magma"
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 12
PANEL_FONTSIZE = 18

SEISMIC_CHANNEL = ("NZ", STATION, "10", "HHZ")
INFRASOUND_CHANNEL = ("NZ", STATION, "31", "HDF")


def _download_trace(channel_info, label):
    network, station, location, channel = channel_info
    client = FDSNClient(FDSN_BASE_URL)

    print(
        f"Downloading fresh {label} data for "
        f"{network}.{station}.{location}.{channel} from {DATA_START} to {DATA_END}"
    )

    stream = client.get_waveforms(
        network,
        station,
        location,
        channel,
        DATA_START,
        DATA_END,
    )
    inventory = client.get_stations(
        network=network,
        station=station,
        location=location,
        channel=channel,
        starttime=DATA_START,
        endtime=DATA_END,
        level="response",
    )

    stream = stream.copy()
    stream.merge(method=1, fill_value=0.0)
    stream.trim(DATA_START, DATA_END, pad=True, fill_value=0.0)
    stream.detrend("demean")
    stream.detrend("linear")

    corrected = Stream()
    for trace in stream:
        tr = trace.copy()
        tr.remove_sensitivity(inventory=inventory)
        tr.detrend("demean")
        tr.detrend("linear")
        corrected += tr

    if len(corrected) == 0:
        raise RuntimeError(f"No usable {label} traces were downloaded.")

    trace = corrected[0].copy()
    trace.trim(DATA_START, DATA_END, pad=True, fill_value=0.0)
    return trace


def _spectrogram(trace, freq_lims, reference_level):
    sampling_rate = float(trace.stats.sampling_rate)
    nfft = max(64, int(round(SPEC_WINDOW_S * sampling_rate)))
    noverlap = min(nfft - 1, int(round(SPEC_OVERLAP_FRAC * nfft)))
    window = np.hanning(nfft)

    power, freqs, bins = mlab_specgram(
        trace.data.astype(float),
        NFFT=nfft,
        Fs=sampling_rate,
        noverlap=noverlap,
        window=window,
        scale_by_freq=True,
    )

    reference_power = reference_level**2
    power_db = 10.0 * np.log10(np.maximum(power, np.finfo(float).tiny) / reference_power)

    plot_offset_s = float(PLOT_START - DATA_START)
    plot_duration_s = float(PLOT_END - PLOT_START)
    freq_mask = (freqs >= freq_lims[0]) & (freqs <= freq_lims[1])
    time_mask = (bins >= plot_offset_s) & (bins <= plot_offset_s + plot_duration_s)

    return {
        "time_s": bins[time_mask] - plot_offset_s,
        "freq_hz": freqs[freq_mask],
        "power_db": power_db[freq_mask][:, time_mask],
    }


def _robust_limits(values):
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0, 1.0
    return (
        float(np.percentile(finite_values, 2.0)),
        float(np.percentile(finite_values, 99.0)),
    )


def _window_spectrum(trace, center_time):
    half_window = WINDOW_LENGTH_S / 2.0
    window_trace = trace.copy()
    window_trace.trim(
        center_time - half_window,
        center_time + half_window,
        pad=True,
        fill_value=0.0,
    )

    data = np.asarray(window_trace.data, dtype=float)
    data = data - np.mean(data)
    data = data * np.hanning(data.size)

    sampling_rate = float(window_trace.stats.sampling_rate)
    freqs = np.fft.rfftfreq(data.size, d=1.0 / sampling_rate)
    amplitude = np.abs(np.fft.rfft(data))

    valid = freqs > 0.0
    freqs = freqs[valid]
    amplitude = amplitude[valid]

    band_mask = (freqs >= SPECTRUM_FREQ_LIMS[0]) & (freqs <= SPECTRUM_FREQ_LIMS[1])
    band_amplitude = amplitude[band_mask]
    scale = np.nanmax(band_amplitude) if band_amplitude.size else np.nanmax(amplitude)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    return freqs[band_mask], amplitude[band_mask] / scale


def _panel_label(ax, label, color):
    ax.text(
        0.012,
        0.975,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_FONTSIZE,
        fontweight="bold",
        color=color,
    )


def _time_label():
    return f"Time (s) since {str(PLOT_START).replace('T', ' ')} UTC"


if __name__ == "__main__":
    seismic_trace = _download_trace(SEISMIC_CHANNEL, "seismic")
    infrasound_trace = _download_trace(INFRASOUND_CHANNEL, "infrasound")

    seismic_spec = _spectrogram(seismic_trace, SEISMIC_FREQ_LIMS, SEISMIC_REFERENCE)
    infrasound_spec = _spectrogram(infrasound_trace, INFRASOUND_FREQ_LIMS, INFRASOUND_REFERENCE)

    seismic_vmin, seismic_vmax = _robust_limits(seismic_spec["power_db"])
    infrasound_vmin, infrasound_vmax = _robust_limits(infrasound_spec["power_db"])

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15.5, 10.0),
        sharex="col",
        constrained_layout=True,
    )

    ax_seis_spec = axes[0, 0]
    ax_seis_spectrum = axes[0, 1]
    ax_inf_spec = axes[1, 0]
    ax_inf_spectrum = axes[1, 1]

    seismic_im = ax_seis_spec.pcolormesh(
        seismic_spec["time_s"],
        seismic_spec["freq_hz"],
        seismic_spec["power_db"],
        shading="auto",
        cmap=COLORMAP,
        vmin=seismic_vmin,
        vmax=seismic_vmax,
    )
    ax_seis_spec.set_ylim(*SEISMIC_FREQ_LIMS)
    ax_seis_spec.set_xlim(0.0, float(PLOT_END - PLOT_START))
    ax_seis_spec.set_ylabel("Frequency (Hz)", fontsize=LABEL_FONTSIZE)
    ax_seis_spec.set_title(f"{STATION} seismic spectrogram", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax_seis_spec.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    _panel_label(ax_seis_spec, "(a)", color="white")

    infrasound_im = ax_inf_spec.pcolormesh(
        infrasound_spec["time_s"],
        infrasound_spec["freq_hz"],
        infrasound_spec["power_db"],
        shading="auto",
        cmap=COLORMAP,
        vmin=infrasound_vmin,
        vmax=infrasound_vmax,
    )
    ax_inf_spec.set_ylim(*INFRASOUND_FREQ_LIMS)
    ax_inf_spec.set_xlim(0.0, float(PLOT_END - PLOT_START))
    ax_inf_spec.set_ylabel("Frequency (Hz)", fontsize=LABEL_FONTSIZE)
    ax_inf_spec.set_xlabel(_time_label(), fontsize=LABEL_FONTSIZE)
    ax_inf_spec.set_title(f"{STATION} infrasound spectrogram", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax_inf_spec.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    _panel_label(ax_inf_spec, "(c)", color="white")

    for center_time, color in zip(WINDOW_CENTERS, WINDOW_COLORS):
        x_pos = float(center_time - PLOT_START)
        ax_seis_spec.axvline(x_pos, color=color, linestyle="--", linewidth=2.5, alpha=0.95)
        ax_inf_spec.axvline(x_pos, color=color, linestyle="--", linewidth=2.5, alpha=0.95)

    for center_time, color, label in zip(WINDOW_CENTERS, WINDOW_COLORS, WINDOW_LABELS):
        freqs, amplitude = _window_spectrum(seismic_trace, center_time)
        ax_seis_spectrum.plot(freqs, amplitude, color=color, linewidth=2.3, label=label)

    for guide_frequency in SEISMIC_GUIDE_LINES:
        ax_seis_spectrum.axvline(guide_frequency, color="black", linestyle="--", linewidth=1.8)

    ax_seis_spectrum.set_xscale("log")
    ax_seis_spectrum.set_yscale("log")
    ax_seis_spectrum.set_xlim(*SPECTRUM_FREQ_LIMS)
    ax_seis_spectrum.set_ylim(1e-6, 1.6)
    ax_seis_spectrum.set_ylabel("Normalised amplitude", fontsize=LABEL_FONTSIZE)
    ax_seis_spectrum.set_title(f"{STATION} seismic spectra", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax_seis_spectrum.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax_seis_spectrum.legend(loc="lower left", fontsize=LEGEND_FONTSIZE)
    _panel_label(ax_seis_spectrum, "(b)", color="black")

    for center_time, color, label in zip(WINDOW_CENTERS, WINDOW_COLORS, WINDOW_LABELS):
        freqs, amplitude = _window_spectrum(infrasound_trace, center_time)
        ax_inf_spectrum.plot(freqs, amplitude, color=color, linewidth=2.3, label=label)

    for guide_frequency in INFRASOUND_GUIDE_LINES:
        ax_inf_spectrum.axvline(guide_frequency, color="black", linestyle="--", linewidth=1.8)

    ax_inf_spectrum.set_xscale("log")
    ax_inf_spectrum.set_yscale("log")
    ax_inf_spectrum.set_xlim(*SPECTRUM_FREQ_LIMS)
    ax_inf_spectrum.set_ylim(1e-6, 1.6)
    ax_inf_spectrum.set_xlabel("Frequency (Hz)", fontsize=LABEL_FONTSIZE)
    ax_inf_spectrum.set_ylabel("Normalised amplitude", fontsize=LABEL_FONTSIZE)
    ax_inf_spectrum.set_title(f"{STATION} infrasound spectra", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax_inf_spectrum.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax_inf_spectrum.legend(loc="lower left", fontsize=LEGEND_FONTSIZE)
    _panel_label(ax_inf_spectrum, "(d)", color="black")

    seismic_cbar = fig.colorbar(seismic_im, ax=ax_seis_spec, pad=0.02)
    seismic_cbar.set_label(r"PSD (dB re $(10^{-9}\ \mathrm{m/s})^2/\mathrm{Hz}$)", fontsize=LABEL_FONTSIZE)
    seismic_cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    infrasound_cbar = fig.colorbar(infrasound_im, ax=ax_inf_spec, pad=0.02)
    infrasound_cbar.set_label(r"PSD (dB re $(20\ \mu\mathrm{Pa})^2/\mathrm{Hz}$)", fontsize=LABEL_FONTSIZE)
    infrasound_cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    SAVE_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_FIG, dpi=300, bbox_inches="tight")

    print(f"Saved figure -> {SAVE_FIG}")

    plt.show()
