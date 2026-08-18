from obspy import UTCDateTime, read, read_inventory
from obspy.geodetics.base import gps2dist_azimuth
import matplotlib.pyplot as plt
import numpy as np
import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
SAVE_FIG = HERE / "FIGURES" / "Figure2.pdf"
ROOT = HERE.parent
CRATER_LAT = -39.1566302543244
CRATER_LON = 175.63253480007924

SITE_AMPLIFICATION_FILE = (
    HERE
    / "SiteAmplification"
    / "site_amplification_summary.csv"
)
SITE_AMPLIFICATION_BAND = "1-10Hz"
SNR_NOISE_START = UTCDateTime("2026-03-21T13:36:05")
SNR_NOISE_END = UTCDateTime("2026-03-21T13:36:15")
SNR_SIGNAL_START = UTCDateTime("2026-03-21T13:36:25")
SNR_SIGNAL_END = UTCDateTime("2026-03-21T13:36:45")


def load_station_distances(xml_paths):
    station_distances = {}
    for xml_path in xml_paths:
        if not xml_path.exists():
            print(f"Station metadata file not found: {xml_path}")
            continue

        inv = read_inventory(str(xml_path))
        for network in inv:
            for station in network:
                if station.code in station_distances:
                    continue

                distance_m, _, _ = gps2dist_azimuth(
                    CRATER_LAT,
                    CRATER_LON,
                    station.latitude,
                    station.longitude,
                )
                station_distances[station.code] = distance_m / 1000.0
    return station_distances


def load_site_amplification_factors(path, frequency_band):
    factors = {}
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row["frequency_band"] != frequency_band:
                continue
            factors[row["station"]] = float(row["site_amplification"])
    return factors


def apply_site_amplification_correction(stream, factors):
    corrected = stream.copy()
    for tr in corrected:
        factor = factors.get(tr.stats.station)
        if factor is None:
            print(f"No site amplification factor for {tr.stats.station}; leaving trace unchanged.")
            continue
        if not np.isfinite(factor) or factor <= 0.0:
            print(f"Invalid site amplification factor for {tr.stats.station}: {factor}; leaving trace unchanged.")
            continue
        tr.data = tr.data / factor
    return corrected


def window_rms(trace, start_time, end_time):
    start_s = start_time - trace.stats.starttime
    end_s = end_time - trace.stats.starttime
    time_s = trace.times()
    mask = (time_s >= start_s) & (time_s <= end_s) & np.isfinite(trace.data)
    if not np.any(mask):
        return np.nan
    return float(np.sqrt(np.mean(np.square(trace.data[mask]))))


def trace_snr(trace, noise_start, noise_end, signal_start, signal_end):
    noise_rms = window_rms(trace, noise_start, noise_end)
    signal_rms = window_rms(trace, signal_start, signal_end)
    if not np.isfinite(noise_rms) or noise_rms <= 0.0:
        return np.nan
    if not np.isfinite(signal_rms):
        return np.nan
    return signal_rms / noise_rms


def compute_snr_by_station(stream):
    return {
        tr.stats.station: trace_snr(
            tr,
            SNR_NOISE_START,
            SNR_NOISE_END,
            SNR_SIGNAL_START,
            SNR_SIGNAL_END,
        )
        for tr in stream
    }


def print_snr_by_station(label, snr_by_station, station_order):
    print(f"{label} signal-to-noise ratios")
    print(f"  Noise:  {SNR_NOISE_START} to {SNR_NOISE_END}")
    print(f"  Signal: {SNR_SIGNAL_START} to {SNR_SIGNAL_END}")
    for sta in station_order:
        snr = snr_by_station.get(sta, np.nan)
        snr_str = f"{snr:.2f}" if np.isfinite(snr) else "nan"
        print(f"  {sta}: {snr_str}")

# -------------------------
# Load data
# -------------------------
st_seis = read(str(HERE / "DATA" / "seismic_20260321_133600.mseed"))
st_inf = read(str(HERE / "DATA" / "infrasound_20260321_133600.mseed"))
station_distance_km = load_station_distances([
    HERE / "DATA" / "seismic_20260321_133600.xml",
    HERE / "DATA" / "infrasound_20260321_133600.xml",
])
site_amplification_factors = load_site_amplification_factors(
    SITE_AMPLIFICATION_FILE,
    SITE_AMPLIFICATION_BAND,
)

# -------------------------
# Preprocess seismic
# -------------------------
st_seis_plot = st_seis.copy()
st_seis_plot.detrend("demean")
st_seis_plot.detrend("linear")
st_seis_plot.taper(max_percentage=0.05)
st_seis_plot.filter("bandpass", freqmin=1, freqmax=10, corners=4, zerophase=True)
st_seis_plot = apply_site_amplification_correction(st_seis_plot, site_amplification_factors)
st_seis_plot.sort(keys=["station"])

# -------------------------
# Preprocess infrasound
# Adjust filter if needed for your data
# -------------------------
st_inf_plot = st_inf.copy()
st_inf_plot.detrend("demean")
st_inf_plot.detrend("linear")
st_inf_plot.taper(max_percentage=0.05)
st_inf_plot.filter("bandpass", freqmin=2, freqmax=10, corners=4, zerophase=True)
st_inf_plot.sort(keys=["station"])

# -------------------------
# Build station order from seismic
# This defines the row layout for BOTH panels
# -------------------------
seis_stations = [tr.stats.station for tr in st_seis_plot]
seis_station_order = []
for sta in seis_stations:
    if sta not in seis_station_order:
        seis_station_order.append(sta)

station_order = sorted(
    seis_station_order,
    key=lambda sta: (station_distance_km.get(sta, np.inf), sta),
)
        

# Choose a colormap with enough distinct colors
cmap = plt.get_cmap("tab20")  # good up to ~20 stations

# Map each station to a color
station_to_color = {
    sta: cmap(i % cmap.N) for i, sta in enumerate(station_order)
}

seismic_snr_by_station = compute_snr_by_station(st_seis_plot)
infrasound_snr_by_station = compute_snr_by_station(st_inf_plot)
print_snr_by_station("Seismic", seismic_snr_by_station, station_order)
print_snr_by_station("Infrasound", infrasound_snr_by_station, station_order)

# Fixed vertical offset for each station
spacing = -2.0
station_to_offset = {sta: i * spacing for i, sta in enumerate(station_order)}

# -------------------------
# Figure with two subplots
# -------------------------
fig, (ax_seis, ax_inf) = plt.subplots(
    1, 2, figsize=(18, 14), sharey=True
)

yticks = [station_to_offset[sta] for sta in station_order]
ylabels = station_order

# -------------------------
# Plot seismic
# -------------------------
for tr in st_seis_plot:
    sta = tr.stats.station
    offset = station_to_offset[sta]
    t = tr.times()-5

    amp_max = np.max(np.abs(tr.data))
    amp_max_scaled = amp_max * 1e6   # assuming m/s -> µm/s
    amp_str = f"{amp_max_scaled:.3f}".rstrip("0").rstrip(".")
    snr = seismic_snr_by_station.get(sta, np.nan)
    snr_str = f"SNR={snr:.1f}" if np.isfinite(snr) else "SNR=nan"

    if amp_max > 0:
        data_plot = tr.data / amp_max
    else:
        data_plot = tr.data

    ax_seis.plot(t, data_plot + offset, linewidth=1,
             color=station_to_color[sta])

    ax_seis.text(
        t[-1] - 8,
        offset + 0.7,
        f"{amp_str} µm/s",
        va="center",
        ha="right",
        fontsize=16
    )

    ax_seis.text(
        t[0] + 8,
        offset + 0.7,
        snr_str,
        va="center",
        ha="left",
        fontsize=14
    )

# -------------------------
# Plot infrasound
# Uses same station offsets, so missing stations become gaps
# -------------------------
for tr in st_inf_plot:
    sta = tr.stats.station

    # skip stations not present in seismic ordering
    # remove this check if you want extra infrasound-only stations appended another way
    if sta not in station_to_offset:
        continue

    offset = station_to_offset[sta]
    t = tr.times()-5

    amp_max = np.max(np.abs(tr.data))
    amp_str = f"{amp_max:.3f} Pa".rstrip("0").rstrip(".")
    snr = infrasound_snr_by_station.get(sta, np.nan)
    snr_str = f"SNR={snr:.1f}" if np.isfinite(snr) else "SNR=nan"

    if amp_max > 0:
        data_plot = tr.data / amp_max
    else:
        data_plot = tr.data

    ax_inf.plot(t, data_plot + offset, linewidth=1,
            color=station_to_color[sta])

    ax_inf.text(
        t[-1] - 8,
        offset + 0.7,
        amp_str,
        va="center",
        ha="right",
        fontsize=16
    )

    ax_inf.text(
        t[0] + 8,
        offset + 0.7,
        snr_str,
        va="center",
        ha="left",
        fontsize=14
    )

# -------------------------
# Axes formatting
# -------------------------
for ax in (ax_seis, ax_inf):
    ax.set_xlim(0, 120)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", labelsize=16)

ax_seis.set_yticks(yticks)
ax_seis.set_yticklabels(ylabels, fontsize=16)
ax_inf.set_yticks(yticks)
ax_inf.set_yticklabels(ylabels, fontsize=16)

for sta in station_order:
    distance_km = station_distance_km.get(sta)
    if distance_km is None:
        continue

    ax_inf.text(
        1.02,
        station_to_offset[sta],
        f"{distance_km:.1f} km",
        transform=ax_inf.get_yaxis_transform(),
        va="center",
        ha="left",
        fontsize=16,
        clip_on=False,
    )

ax_seis.set_xlabel("Time (s) since 2026-03-21 13:36:05 (UTC)", fontsize=16)
ax_inf.set_xlabel("Time (s) since 2026-03-21 13:36:05 (UTC)", fontsize=16)

ax_seis.set_title(
    f"(a) Seismic",
    fontsize=18,
    fontweight="bold",
)
ax_inf.set_title("(b) Infrasound", fontsize=18, fontweight="bold")

ymax = -spacing * 0.7
ymin = (len(station_order) - 1) * spacing + spacing * 0.7
ax_seis.set_ylim(ymin, ymax)
fig.savefig(SAVE_FIG, dpi=300, bbox_inches="tight")


plt.tight_layout(rect=(0, 0, 0.92, 1))
plt.show()
