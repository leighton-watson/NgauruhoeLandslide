import warnings
from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.core.inventory import Inventory
import matplotlib.pyplot as plt

# Initialize the FDSN client
client = FDSN_Client("https://service.geonet.org.nz")

# Times are already in UTC
t_start = UTCDateTime(2026, 3, 21, 13, 36, 0)
t_end   = UTCDateTime(2026, 3, 21, 13, 38, 10)

# Define stations and their channels
stations = [
    ("NZ", "OTVZ", "31", "HDF"),
    ("NZ", "SNVZ", "30", "HDF"),
    ("NZ", "ETVZ", "31", "HDF"),
    ("NZ", "NGZ",  "30", "HDF"),
    ("NZ", "WTVZ", "30", "HDF"),
    ("NZ", "NOVZ", "30", "HDF"),
    ("NZ", "KRVZ", "30", "HDF"),
    ("NZ", "TMVZ", "31", "HDF"),
    ("NZ", "NTVZ", "31", "HDF"),
    ("NZ", "FWVZ", "30", "HDF"),
    ("NZ", "WHVZ", "30", "HDF"),
    ("NZ", "COVZ", "30", "HDF"),
    ("NZ", "MAVZ", "30", "HDF"),
    ("NZ", "TRVZ", "31", "HDF"),
]

# -------------------------
# DOWNLOAD WAVEFORMS + METADATA
# -------------------------
st = Stream()
inventory = Inventory(networks=[], source="GeoNet via ObsPy")

for net, sta, loc, cha in stations:
    seed_id = f"{net}.{sta}.{loc}.{cha}"
    print(f"Downloading {seed_id} ...")
    try:
        st_tmp = client.get_waveforms(net, sta, loc, cha, t_start, t_end)
        st += st_tmp

        inv_tmp = client.get_stations(
            network=net,
            station=sta,
            location=loc,
            channel=cha,
            starttime=t_start,
            endtime=t_end,
            level="response"
        )
        inventory += inv_tmp

    except Exception as e:
        print(f"  Could not retrieve {seed_id}: {e}")

st.sort()

# -------------------------
# PREPROCESS RAW DATA
# -------------------------
st_raw = st.copy()

# Suppress expected cosmetic warnings for infrasound sensors:
#   - hPa / HPA unit: ObsPy doesn't recognise hectopascals natively,
#     but remove_sensitivity() only needs the scalar gain, so it works correctly.
#   - PolynomialResponseStage DC offset: infrasound sensors commonly use
#     polynomial responses; the DC offset is irrelevant for AC signals.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=".*hPa.*|.*HPA.*|.*PolynomialResponseStage.*|.*DC offset.*",
        category=UserWarning,
    )
    st_raw.remove_sensitivity(inventory=inventory)

# -------------------------
# PER-TRACE hPa -> Pa CONVERSION
# -------------------------
# Some GeoNet infrasound sensors store sensitivity in hPa (NGZ, WTVZ, FWVZ, WHVZ).
# After remove_sensitivity() those traces are in hPa and need x100.
# Stations with Pa sensitivities are already correct.
hpa_stations = set()
for network in inventory:
    for station in network:
        for channel in station:
            sens = channel.response.instrument_sensitivity
            if sens and sens.input_units.lower() == "hpa":
                hpa_stations.add(station.code)
            break  # one channel per station is enough

print("\n=== Unit Check & Conversion ===")
for tr in st_raw:
    if tr.stats.station in hpa_stations:
        tr.data = tr.data * 100.0
        print(f"  {tr.id}: hPa -> Pa (x100)")
    else:
        print(f"  {tr.id}: already in Pa")

st_raw.detrend("demean")
st_raw.detrend("linear")

# Cast to float32 to match MiniSEED FLOAT32 encoding and avoid
# the "encoding does not match dtype" warning on write.
for tr in st_raw:
    tr.data = tr.data.astype("float32")

# -------------------------
# PRINT STATION COORDINATES
# -------------------------
print("\n=== Station Coordinates ===")
print(f"{'Station':<8} {'Latitude':>10} {'Longitude':>12} {'Elevation(m)':>14}")
print("-" * 48)
seen = set()
for network in inventory:
    for station in network:
        for channel in station:
            key = (network.code, station.code, channel.location_code, channel.code)
            if key in seen:
                continue
            seen.add(key)
            print(
                f"{station.code:<8} "
                f"{channel.latitude:>10.4f} "
                f"{channel.longitude:>12.4f} "
                f"{channel.elevation:>14.1f}"
            )
            break

# -------------------------
# SAVE RAW DATA (UNFILTERED, SENSITIVITY-CORRECTED, float32)
# -------------------------
mseed_filename = f"infrasound_{t_start.strftime('%Y%m%d_%H%M%S')}.mseed"
st_raw.write(mseed_filename, format="MSEED", encoding="FLOAT32")
print(f"\nWaveforms saved to: {mseed_filename}")

xml_filename = f"infrasound_{t_start.strftime('%Y%m%d_%H%M%S')}.xml"
inventory.write(xml_filename, format="STATIONXML")
print(f"Station metadata saved to: {xml_filename}")

# -------------------------
# MAKE A FILTERED COPY FOR PLOTTING ONLY
# -------------------------
st_plot = st_raw.copy()

st_plot.taper(max_percentage=0.05, type="cosine")

# 1–10 Hz bandpass, zero-phase to avoid time shift.
# Lower freqmin to ~0.5 Hz to capture infrasound signals below 1 Hz.
st_plot.filter("bandpass", freqmin=1.0, freqmax=10.0, corners=4, zerophase=True)

# -------------------------
# PLOT FILTERED DATA
# -------------------------
st_plot.plot(equal_scale=False, size=(1200, 900))
plt.show()