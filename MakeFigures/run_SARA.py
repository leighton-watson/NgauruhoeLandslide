"""
Run station-amplitude-ratio analysis and save reusable outputs for later plotting.
"""

from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from AnalysisCodes.sara import run_method, save_output_bundle


TOPOGRAPHY_DIR = ROOT / "Topography"
MSEED_FILE = ROOT / "DATA" / "seismic_20260321_133600.mseed"
STATION_COORDS_FILE = ROOT / "DATA" / "seismic_20260321_133600.xml"

STATIONS = [
    "COVZ",
    "ETVZ",
    "FWVZ",
    "KRVZ",
    "NGZ",
    "NOVZ",
    "NTVZ",
    "OTVZ",
    "SNVZ",
    "TMVZ",
    "TUVZ",
    "TWVZ",
    "WHVZ",
    "WTVZ",
]

START_TIME = "2026-03-21T13:36:00"
END_TIME = "2026-03-21T13:38:10"
PLOT_START_TIME = "2026-03-21T13:36:05"
PLOT_END_TIME = "2026-03-21T13:38:05"

CRATER_LAT = -39.1566302543244
CRATER_LON = 175.63253480007924

FREQ_MIN = 2.0
FREQ_MAX = 10.0
WIN_LENGTH_S = 5.0
WIN_STEP_S = 0.5
ENVELOPE_SMOOTH_S = 1.0

SEARCH_HALF_WIDTH = 5000
SEARCH_BOX = None
SPATIAL_STEP = 100
DECAY_EXPONENT = 1.0
STORE_FULL_FIELDS = False

SAVE_OUTPUT = HERE / "RESULTS" / "SARA_output.npz"


if __name__ == "__main__":
    cfg = dict(
        TOPOGRAPHY_DIR=TOPOGRAPHY_DIR,
        MSEED_FILE=MSEED_FILE,
        STATION_COORDS_FILE=STATION_COORDS_FILE,
        STATIONS=STATIONS,
        START_TIME=START_TIME,
        END_TIME=END_TIME,
        PLOT_START_TIME=PLOT_START_TIME,
        PLOT_END_TIME=PLOT_END_TIME,
        CRATER_LAT=CRATER_LAT,
        CRATER_LON=CRATER_LON,
        FREQ_MIN=FREQ_MIN,
        FREQ_MAX=FREQ_MAX,
        WIN_LENGTH_S=WIN_LENGTH_S,
        WIN_STEP_S=WIN_STEP_S,
        ENVELOPE_SMOOTH_S=ENVELOPE_SMOOTH_S,
        SEARCH_AREA=SEARCH_HALF_WIDTH,
        SEARCH_BOX=SEARCH_BOX,
        SPATIAL_STEP=SPATIAL_STEP,
        DECAY_EXPONENT=DECAY_EXPONENT,
        STORE_FULL_FIELDS=STORE_FULL_FIELDS,
    )

    topo, data, result = run_method("amplitude_ratio", cfg)
    save_path = save_output_bundle(SAVE_OUTPUT, cfg, topo, data, result)
    print(f"Saved SARA outputs → {save_path}")
