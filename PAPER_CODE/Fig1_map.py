from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import contextily as ctx
import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar
from obspy import read_inventory
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import LineString, Point


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "DATA"
TOPO_DIR = HERE / "DATA"
WEATHER_DIR = HERE / "DATA"
OUTPUT_FIGURE = HERE / "Figures" / "Figure1.pdf"

INVENTORY_FILE = DATA_DIR / "seismic_20260321_133600.xml"
INFRASOUND_INVENTORY_FILE = DATA_DIR / "infrasound_20260321_133600.xml"
DEM_FILE = TOPO_DIR / "topography.npz"
RAIN_CSV_FILE = WEATHER_DIR / "46771__Rain__hourly.csv"

CRATER_LAT = -39.1566302543244
CRATER_LON = 175.63253480007924
PROFILE_STATION = "OTVZ"
PROFILE_EXTENSION_M = 500.0
SEARCH_BOX_SIDE_M = 10000.0
EWS_LON = 175.64597
EWS_LAT = -39.13248
EWS_ELEV_M = 1903

RAIN_TIMEZONE = "UTC"
RAIN_START_DATE = date(2026, 3, 1)
RAIN_END_YEAR = 2026
RAIN_END_MONTH = 4
AXIS_LABEL_SIZE = 15
TICK_LABEL_SIZE = 13
TITLE_SIZE = 16
RAIN_EVENT_MARKERS = (
    {
        "label": "Event",
        "kind": "timestamp",
        "timestamp": datetime(2026, 3, 21, 13, 36, 15, tzinfo=ZoneInfo("UTC")),
        "source_timezone": "UTC",
        "style": dict(marker="*", color="crimson", markersize=20),
    },
    {
        "label": "Overflight",
        "kind": "local_date",
        "date": date(2026, 4, 10),
        "source_timezone": "Pacific/Auckland",
        "style": dict(marker="D", color="darkgreen", markersize=15),
    },
)


def add_north_arrow(ax, x=0.95, y=0.95, size=0.05):
    ax.annotate(
        "",
        xy=(x, y),
        xycoords="axes fraction",
        xytext=(x, y - size),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="white", lw=2),
        zorder=10,
    )
    ax.text(
        x,
        y + 0.01,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        color="white",
        path_effects=[pe.withStroke(linewidth=2, foreground="black")],
        zorder=10,
    )


def add_scale_bar(ax):
    scalebar = ScaleBar(
        1,
        units="m",
        dimension="si-length",
        length_fraction=0.18,
        location="lower right",
        color="white",
        box_color="black",
        box_alpha=0.4,
        font_properties={"size": 12},
        sep=3,
    )
    ax.add_artist(scalebar)

def load_stations(inventory_file: Path) -> list[dict[str, object]]:
    inv = read_inventory(str(inventory_file))
    stations = []
    for network in inv:
        for station in network:
            stations.append(
                {
                    "label": station.code,
                    "network": network.code,
                    "station": station.code,
                    "lat": station.latitude,
                    "lon": station.longitude,
                    "elev": station.elevation,
                }
            )
    return stations


def prepare_dem_interpolator(dem_file: Path):
    data = np.load(dem_file)
    x_grid = data["X"]
    y_grid = data["Y"]
    elevation_grid = data["C"]

    if x_grid.ndim == 2 and y_grid.ndim == 2:
        x1d = x_grid[0, :]
        y1d = y_grid[:, 0]
    elif x_grid.ndim == 1 and y_grid.ndim == 1:
        x1d = x_grid
        y1d = y_grid
    else:
        raise ValueError("Unsupported X/Y format in DEM")

    if np.any(np.diff(x1d) < 0):
        x1d = x1d[::-1]
        elevation_grid = elevation_grid[:, ::-1]

    if np.any(np.diff(y1d) < 0):
        y1d = y1d[::-1]
        elevation_grid = elevation_grid[::-1, :]

    return RegularGridInterpolator(
        (y1d, x1d),
        elevation_grid,
        bounds_error=False,
        fill_value=np.nan,
    )


def build_profile(
    crater_lon: float,
    crater_lat: float,
    target_lon: float,
    target_lat: float,
    extension_m: float,
    interpolator: RegularGridInterpolator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
    crater_x, crater_y = transformer.transform(crater_lon, crater_lat)
    target_x, target_y = transformer.transform(target_lon, target_lat)

    dx = target_x - crater_x
    dy = target_y - crater_y
    base_length = float(np.hypot(dx, dy))
    unit_x = dx / base_length
    unit_y = dy / base_length

    start_x = crater_x - extension_m * unit_x
    start_y = crater_y - extension_m * unit_y
    end_x = target_x + extension_m * unit_x
    end_y = target_y + extension_m * unit_y

    profile_length = base_length + 2 * extension_m
    step_m = 10.0
    distance_m = np.arange(0.0, profile_length + step_m, step_m)

    xs = start_x + distance_m * unit_x
    ys = start_y + distance_m * unit_y
    elevation_m = interpolator(np.column_stack([ys, xs]))

    crater_distance_m = extension_m
    station_distance_m = extension_m + base_length
    distance_from_crater_m = distance_m - crater_distance_m

    return (
        distance_m,
        distance_from_crater_m,
        elevation_m,
        np.column_stack([xs, ys]),
        crater_distance_m,
        station_distance_m,
    )


def resolve_marker_date(marker: dict[str, object], target_timezone: str) -> date:
    target_tz = ZoneInfo(target_timezone)

    if marker["kind"] == "timestamp":
        timestamp = marker["timestamp"]
        assert isinstance(timestamp, datetime)
        return timestamp.astimezone(target_tz).date()

    if marker["kind"] == "local_date":
        source_tz = ZoneInfo(str(marker["source_timezone"]))
        local_day = marker["date"]
        assert isinstance(local_day, date)
        local_noon = datetime.combine(local_day, datetime.min.time(), tzinfo=source_tz)
        local_noon += timedelta(hours=12)
        return local_noon.astimezone(target_tz).date()

    raise ValueError(f"Unsupported marker kind: {marker['kind']}")


def load_daily_rainfall(
    csv_file: Path,
    timezone_name: str,
    start_date: date,
    end_year: int,
    end_month: int,
    marker_dates: list[date],
) -> tuple[dict[date, float], datetime | None]:
    tz = ZoneInfo(timezone_name)
    daily_totals = defaultdict(float)
    latest_timestamp = None

    with csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp_utc = datetime.fromisoformat(
                row["Observation time UTC"].replace("Z", "+00:00")
            )
            timestamp = timestamp_utc.astimezone(tz)
            observation_date = timestamp.date()

            if observation_date < start_date:
                continue

            if (timestamp.year, timestamp.month) > (end_year, end_month):
                continue

            rain_text = (row.get("Rainfall [mm]") or "").strip()
            rainfall_mm = float(rain_text) if rain_text else 0.0
            daily_totals[observation_date] += rainfall_mm

            if latest_timestamp is None or timestamp > latest_timestamp:
                latest_timestamp = timestamp

    if latest_timestamp is None:
        return {}, None

    filled_totals = {}
    current_date = start_date
    end_date = max([latest_timestamp.date(), *marker_dates])

    while current_date <= end_date:
        filled_totals[current_date] = daily_totals.get(current_date, 0.0)
        current_date += timedelta(days=1)

    return filled_totals, latest_timestamp


def plot_rainfall_panel(
    ax,
    daily_totals: dict[date, float],
    timezone_name: str,
    latest_timestamp: datetime,
    event_markers: tuple[dict[str, object], ...],
):
    days = list(daily_totals.keys())
    rainfall = list(daily_totals.values())
    positions = list(range(len(days)))
    labels = [day.strftime("%d %b") for day in days]
    tick_positions = list(range(0, len(days), 7))
    tick_labels = [labels[index] for index in tick_positions]

    ax.bar(
        positions,
        rainfall,
        width=0.85,
        color="tab:blue",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_title("(c) Rainfall", fontsize=TITLE_SIZE)
    ax.set_ylabel("Rainfall (mm)", fontsize=AXIS_LABEL_SIZE)
    ax.set_xlabel("Date", fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.set_xlim(-0.6, len(days) - 0.4)
    ax.grid(axis="y", alpha=0.3)

    max_rainfall = max(rainfall, default=0.0)
    marker_y = max(0.35, max_rainfall * 0.04)
    label_y = marker_y + max(0.4, max_rainfall * 0.05)

    for marker in event_markers:
        marker_date = resolve_marker_date(marker, timezone_name)
        if marker_date not in daily_totals:
            continue

        x_position = days.index(marker_date)
        style = marker["style"]
        assert isinstance(style, dict)
        ax.plot(x_position, marker_y, linestyle="None", **style)
        ax.text(
            x_position,
            label_y,
            str(marker["label"]),
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylim(0, max(max_rainfall * 1.08, label_y + 0.4))


stations = load_stations(INVENTORY_FILE)
print(f"Loaded {len(stations)} stations")
infrasound_stations = load_stations(INFRASOUND_INVENTORY_FILE)
infrasound_station_codes = {station["station"] for station in infrasound_stations}

station_lookup = {station["station"]: station for station in stations}
if PROFILE_STATION not in station_lookup:
    raise ValueError(f"Station {PROFILE_STATION} not found in inventory")

profile_station = station_lookup[PROFILE_STATION]
for station in stations:
    station["has_infrasound"] = station["station"] in infrasound_station_codes

station_gdf = gpd.GeoDataFrame(
    stations,
    geometry=[Point(s["lon"], s["lat"]) for s in stations],
    crs="EPSG:4326",
).to_crs(epsg=2193)
station_gdf["marker_group"] = np.where(
    station_gdf["has_infrasound"],
    "co-located seismic + infrasound",
    "seismic only",
)

crater_gdf = gpd.GeoDataFrame(
    [{"label": "Ngauruhoe"}],
    geometry=[Point(CRATER_LON, CRATER_LAT)],
    crs="EPSG:4326",
).to_crs(epsg=2193)
crater_x = crater_gdf.geometry.x.iloc[0]
crater_y = crater_gdf.geometry.y.iloc[0]

profile_station_gdf = gpd.GeoDataFrame(
    [{"label": PROFILE_STATION}],
    geometry=[Point(profile_station["lon"], profile_station["lat"])],
    crs="EPSG:4326",
).to_crs(epsg=2193)

ews_gdf = gpd.GeoDataFrame(
    [{"label": "Tongariro Crossing EWS", "elev": EWS_ELEV_M}],
    geometry=[Point(EWS_LON, EWS_LAT)],
    crs="EPSG:4326",
).to_crs(epsg=2193)

dem_interpolator = prepare_dem_interpolator(DEM_FILE)
(
    profile_distance_m,
    profile_distance_from_crater_m,
    profile_elevation_m,
    profile_points_nztm,
    crater_distance_m,
    station_distance_m,
) = build_profile(
    CRATER_LON,
    CRATER_LAT,
    profile_station["lon"],
    profile_station["lat"],
    PROFILE_EXTENSION_M,
    dem_interpolator,
)

line_transformer = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)
profile_line_lon, profile_line_lat = line_transformer.transform(
    profile_points_nztm[:, 0],
    profile_points_nztm[:, 1],
)
profile_line_gdf = gpd.GeoDataFrame(
    [{"label": "Profile"}],
    geometry=[LineString(zip(profile_line_lon, profile_line_lat))],
    crs="EPSG:4326",
).to_crs(epsg=2193)

rain_marker_dates = [
    resolve_marker_date(marker, RAIN_TIMEZONE) for marker in RAIN_EVENT_MARKERS
]
rain_daily_totals, rain_latest_timestamp = load_daily_rainfall(
    csv_file=RAIN_CSV_FILE,
    timezone_name=RAIN_TIMEZONE,
    start_date=RAIN_START_DATE,
    end_year=RAIN_END_YEAR,
    end_month=RAIN_END_MONTH,
    marker_dates=rain_marker_dates,
)
if rain_latest_timestamp is None:
    raise RuntimeError("No rainfall data found for embedded rainfall panel")

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(
    2,
    2,
    width_ratios=[1.7, 1.0],
    height_ratios=[1.0, 1.0],
    wspace=0.18,
    hspace=0.28,
)
ax_map = fig.add_subplot(gs[:, 0])
ax_profile = fig.add_subplot(gs[0, 1])
ax_rain = fig.add_subplot(gs[1, 1])

station_gdf.loc[station_gdf["has_infrasound"]].plot(
    ax=ax_map,
    marker="s",
    color="yellow",
    edgecolors="black",
    linewidths=0.8,
    markersize=100,
    zorder=5,
)
station_gdf.loc[~station_gdf["has_infrasound"]].plot(
    ax=ax_map,
    marker="o",
    color="deepskyblue",
    edgecolors="black",
    linewidths=0.8,
    markersize=90,
    zorder=5,
)

outline = [pe.withStroke(linewidth=2.5, foreground="black")]
for _, row in station_gdf.iterrows():
    label = str(row["label"])
    if label == "TWVZ":
        xytext = (8, 8)
        ha = "center"
    else:
        xytext = (0, 8)
        ha = "center"
    ax_map.annotate(
        label,
        xy=(row.geometry.x, row.geometry.y),
        xytext=xytext,
        textcoords="offset points",
        fontsize=15,
        fontweight="bold",
        color="white",
        ha=ha,
        va="bottom",
        path_effects=outline,
        zorder=6,
    )

ctx.add_basemap(
    ax_map,
    source="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    zoom="auto",
    crs="EPSG:2193",
)

profile_line_gdf.plot(
    ax=ax_map,
    color="cyan",
    linewidth=2.5,
    linestyle="--",
    zorder=4,
)

ax_map.plot(
    crater_x,
    crater_y,
    marker="*",
    markersize=18,
    color="red",
    markeredgecolor="black",
    markeredgewidth=0.8,
    zorder=7,
)
ax_map.annotate(
    "Ngauruhoe",
    xy=(crater_x, crater_y),
    xytext=(0, 10),
    textcoords="offset points",
    fontsize=15,
    fontweight="bold",
    color="red",
    ha="center",
    va="bottom",
    path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
    zorder=8,
)

station_x = profile_station_gdf.geometry.x.iloc[0]
station_y = profile_station_gdf.geometry.y.iloc[0]


ews_x = ews_gdf.geometry.x.iloc[0]
ews_y = ews_gdf.geometry.y.iloc[0]
ax_map.plot(
    ews_x,
    ews_y,
    marker="s",
    markersize=11,
    color="magenta",
    markeredgecolor="black",
    markeredgewidth=0.9,
    zorder=8,
)
ax_map.annotate(
    "EWS",
    xy=(ews_x, ews_y),
    xytext=(10, 8),
    textcoords="offset points",
    fontsize=13,
    fontweight="bold",
    color="magenta",
    ha="left",
    va="bottom",
    path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
    zorder=9,
)

search_box_half_width = SEARCH_BOX_SIDE_M / 2.0
search_area_patch = mpatches.Rectangle(
    (crater_x - search_box_half_width, crater_y - search_box_half_width),
    SEARCH_BOX_SIDE_M,
    SEARCH_BOX_SIDE_M,
    fill=False,
    edgecolor="orange",
    linewidth=2.2,
    linestyle="-.",
    zorder=7,
)
ax_map.add_patch(search_area_patch)

add_north_arrow(ax_map)
add_scale_bar(ax_map)

legend_handles = [
    Line2D(
        [0],
        [0],
        marker="s",
        color="none",
        markerfacecolor="yellow",
        markeredgecolor="black",
        markeredgewidth=0.8,
        markersize=10,
        label="Co-located seismic + infrasound",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="deepskyblue",
        markeredgecolor="black",
        markeredgewidth=0.8,
        markersize=9,
        label="Seismic only",
    ),
    Line2D(
        [0],
        [0],
        marker="*",
        color="none",
        markerfacecolor="red",
        markeredgecolor="black",
        markeredgewidth=0.8,
        markersize=14,
        label="Ngauruhoe",
    ),
    Line2D(
        [0],
        [0],
        color="cyan",
        linestyle="--",
        linewidth=2.5,
        label="Profile trace",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        color="none",
        markerfacecolor="magenta",
        markeredgecolor="black",
        markeredgewidth=0.9,
        markersize=9,
        label="Tongariro Crossing EWS",
    ),
    mpatches.Patch(
        facecolor="none",
        edgecolor="orange",
        linewidth=2.2,
        linestyle="-.",
        label="Search area",
    ),
]
legend = ax_map.legend(
    handles=legend_handles,
    loc="upper left",
    bbox_to_anchor=(0.02, 0.90),
    fontsize=11,
    frameon=True,
    facecolor="white",
    edgecolor="black",
    framealpha=0.9,
)
legend.set_zorder(20)

ax_map.set_xlabel("Easting (m)", fontsize=AXIS_LABEL_SIZE)
ax_map.set_ylabel("Northing (m)", fontsize=AXIS_LABEL_SIZE)
ax_map.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
ax_map.ticklabel_format(axis="both", style="sci", scilimits=(6, 6), useMathText=False)
ax_map.xaxis.offsetText.set_fontsize(TICK_LABEL_SIZE)
ax_map.yaxis.offsetText.set_fontsize(TICK_LABEL_SIZE)
ax_map.set_title("(a) Station Map", fontsize=TITLE_SIZE)
ax_map.title.set_fontweight("bold")

profile_distance_km = profile_distance_from_crater_m / 1000.0
finite_mask = np.isfinite(profile_elevation_m)
profile_fill_base = 1300

ax_profile.plot(
    profile_distance_km[finite_mask],
    profile_elevation_m[finite_mask],
    color="black",
    linewidth=2,
)
ax_profile.fill_between(
    profile_distance_km[finite_mask],
    profile_elevation_m[finite_mask],
    profile_fill_base,
    color="0.85",
)
crater_elevation = np.interp(crater_distance_m, profile_distance_m, profile_elevation_m)
station_elevation = np.interp(station_distance_m, profile_distance_m, profile_elevation_m)
ax_profile.scatter(
    [0],
    [crater_elevation],
    color="red",
    marker="*",
    s=130,
    edgecolors="black",
    linewidths=0.6,
    zorder=3,
)
ax_profile.scatter(
    [(station_distance_m - crater_distance_m) / 1000.0],
    [station_elevation],
    color="yellow",
    marker="s",
    s=95,
    edgecolors="black",
    linewidths=0.6,
    zorder=3,
)
ax_profile.text(
    0,
    crater_elevation + 55,
    "Ngauruhoe",
    color="red",
    ha="left",
    va="bottom",
    fontsize=13,
    fontweight="bold",
)
ax_profile.text(
    (station_distance_m - crater_distance_m) / 1000.0,
    station_elevation + 25,
    PROFILE_STATION,
    color="yellow",
    ha="right",
    va="bottom",
    fontsize=13,
    fontweight="bold",
    path_effects=[pe.withStroke(linewidth=2, foreground="black")],
)
ax_profile.set_xlim(profile_distance_km[0], profile_distance_km[-1])
ax_profile.set_ylim(1300, 2500)
ax_profile.set_xlabel("Distance from Ngauruhoe (km)", fontsize=AXIS_LABEL_SIZE)
ax_profile.set_ylabel("Elevation (m)", fontsize=AXIS_LABEL_SIZE)
ax_profile.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
ax_profile.grid(alpha=0.3)
ax_profile.set_title("(b) Topographic Profile", fontsize=TITLE_SIZE)
ax_profile.title.set_fontweight("bold")

plot_rainfall_panel(
    ax_rain,
    rain_daily_totals,
    RAIN_TIMEZONE,
    rain_latest_timestamp,
    RAIN_EVENT_MARKERS,
)
ax_rain.title.set_fontweight("bold")

plt.tight_layout()
plt.savefig(OUTPUT_FIGURE, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved -> {OUTPUT_FIGURE.name}")
