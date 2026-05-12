from obspy import read
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Load data
# -------------------------
st_seis = read("../DATA/seismic_20260321_133600.mseed")
st_inf = read("../DATA/infrasound_20260321_133600.mseed")

# -------------------------
# Preprocess seismic
# -------------------------
st_seis_plot = st_seis.copy()
st_seis_plot.detrend("demean")
st_seis_plot.detrend("linear")
st_seis_plot.taper(max_percentage=0.05)
st_seis_plot.filter("bandpass", freqmin=1, freqmax=10, corners=4, zerophase=True)
st_seis_plot.sort(keys=["station"])

# -------------------------
# Preprocess infrasound
# Adjust filter if needed for your data
# -------------------------
st_inf_plot = st_inf.copy()
st_inf_plot.detrend("demean")
st_inf_plot.detrend("linear")
st_inf_plot.taper(max_percentage=0.05)
st_inf_plot.filter("bandpass", freqmin=2, freqmax=15, corners=4, zerophase=True)
st_inf_plot.sort(keys=["station"])

# -------------------------
# Build station order from seismic
# This defines the row layout for BOTH panels
# -------------------------
seis_stations = [tr.stats.station for tr in st_seis_plot]
station_order = []
for sta in seis_stations:
    if sta not in station_order:
        station_order.append(sta)
        

# Choose a colormap with enough distinct colors
cmap = plt.get_cmap("tab20")  # good up to ~20 stations

# Map each station to a color
station_to_color = {
    sta: cmap(i % cmap.N) for i, sta in enumerate(station_order)
}

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

ax_seis.set_xlabel("Time (s) since 2026-03-21 13:36:05 (UTC)", fontsize=16)
ax_inf.set_xlabel("Time (s) since 2026-03-21 13:36:05 (UTC)", fontsize=16)

ax_seis.set_title("(a) Seismic", fontsize=18, fontweight="bold")
ax_inf.set_title("(b) Infrasound", fontsize=18, fontweight="bold")

ymax = -spacing * 0.7
ymin = (len(station_order) - 1) * spacing + spacing * 0.7
ax_seis.set_ylim(ymin, ymax)

plt.tight_layout()
plt.show()

