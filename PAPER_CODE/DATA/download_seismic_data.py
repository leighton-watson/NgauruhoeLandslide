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
    ("NZ", "OTVZ", "10", "HHZ"),
    ("NZ", "SNVZ", "10", "EHZ"),
    ("NZ", "ETVZ", "10", "HHZ"),
    ("NZ", "NGZ",  "10", "HHZ"),
    ("NZ", "WTVZ", "10", "HHZ"),
    ("NZ", "NOVZ", "10", "EHZ"),
    ("NZ", "KRVZ", "10", "HHZ"),
    ("NZ", "TMVZ", "10", "HHZ"),
    ("NZ", "NTVZ", "10", "HHZ"),
    ("NZ", "COVZ", "11", "HHZ"),
    ("NZ", "FWVZ", "10", "HHZ"),
    ("NZ", "WHVZ", "10", "HHZ"),
    ("NZ", "TWVZ", "10", "HHZ"),
    ("NZ", "MAVZ", "11", "HHZ"),
    ("NZ", "TRVZ", "10", "HHZ"),
    ("NZ", "TUVZ", "10", "EHZ"),
]

# Containers
st = Stream()
inventory = Inventory(networks=[], source="GeoNet via ObsPy")

for net, sta, loc, cha in stations:
    seed_id = f"{net}.{sta}.{loc}.{cha}"
    print(f"Downloading {seed_id} ...")

    try:
        # 1) Get waveform only
        st_tmp = client.get_waveforms(net, sta, loc, cha, t_start, t_end)
        st += st_tmp

        # 2) Get full response metadata needed for sensitivity/response removal
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

# Optional: sort for cleaner output/plotting
st.sort()

# -------------------------
# PREPROCESS RAW DATA
# -------------------------
# This stream will remain unfiltered for saving
st_raw = st.copy()

# Remove instrument sensitivity using the Inventory
# This converts to physical units based on channel sensitivity metadata.
st_raw.remove_sensitivity(inventory=inventory)

# Basic detrending
st_raw.detrend("demean")
st_raw.detrend("linear")

# -------------------------
# MAKE A FILTERED COPY FOR PLOTTING ONLY
# -------------------------
st_plot = st_raw.copy()

# Optional taper before filtering
st_plot.taper(max_percentage=0.05, type="cosine")

# 1–10 Hz bandpass, zero-phase to avoid time shift
st_plot.filter("bandpass", freqmin=1.0, freqmax=10.0, corners=4, zerophase=True)

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
# SAVE RAW DATA (UNFILTERED, BUT SENSITIVITY-CORRECTED)
# -------------------------
mseed_filename = f"seismic_{t_start.strftime('%Y%m%d_%H%M%S')}.mseed"
st_raw.write(mseed_filename, format="MSEED")
print(f"\nWaveforms saved to: {mseed_filename}")

xml_filename = f"seismic_{t_start.strftime('%Y%m%d_%H%M%S')}.xml"
inventory.write(xml_filename, format="STATIONXML")
print(f"Station metadata saved to: {xml_filename}")

# -------------------------
# PLOT FILTERED DATA
# -------------------------
st_plot.plot(equal_scale=False, size=(1200, 900))
plt.show()