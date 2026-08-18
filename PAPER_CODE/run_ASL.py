"""
Run amplitude source localisation and save reusable outputs for later plotting.
"""

from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from asl import run_analysis, save_output_bundle


TOPOGRAPHY_DIR = HERE / "DATA"
MSEED_FILE = HERE / "DATA" / "seismic_20260321_133600.mseed"
STATION_COORDS_FILE = HERE / "DATA" / "seismic_20260321_133600.xml"

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

FREQ_MIN = 1.0
FREQ_MAX = 10.0
WIN_LENGTH_S = 5.0
WIN_STEP_S = 0.5

# Amplitude corrections. Set either USE_* switch to False to disable that
# correction while retaining the measured values below for reproducibility.
USE_SITE_AMPLIFICATION = True
SITE_AMPLIFICATION_CSV = (
    HERE / "SiteAmplification" / "site_amplification_summary.csv"
)
SITE_AMPLIFICATION_COLUMN = "site_amplification"

USE_INTRINSIC_ATTENUATION = True
ATTENUATION_COEFFICIENT_PER_KM = 0.0219

SEARCH_HALF_WIDTH = 5000
SEARCH_BOX = None
SPATIAL_STEP = 100
TEMPORAL_STEP = 0.5
# Geometric-spreading exponent in r**(-B); this is distinct from the
# intrinsic attenuation coefficient used in exp(-attenuation_B * r).
B = 1.0
STORE_FULL_FIELDS = False

SAVE_OUTPUT = HERE / "Outputs" / "ASL_output.npz"


if __name__ == "__main__":
    active_site_amplification_csv = (
        SITE_AMPLIFICATION_CSV if USE_SITE_AMPLIFICATION else None
    )
    active_attenuation_coefficient_per_km = (
        ATTENUATION_COEFFICIENT_PER_KM if USE_INTRINSIC_ATTENUATION else 0.0
    )

    print("\n" + "=" * 70)
    print("ASL amplitude-decay configuration")
    print("=" * 70)
    if USE_SITE_AMPLIFICATION:
        print("Site amplification       : ENABLED")
        print(f"  factor table           : {SITE_AMPLIFICATION_CSV}")
        print(f"  factor column          : {SITE_AMPLIFICATION_COLUMN}")
    else:
        print("Site amplification       : DISABLED")
    if USE_INTRINSIC_ATTENUATION:
        print("Intrinsic attenuation    : ENABLED")
        print(
            "  attenuation B          : "
            f"{active_attenuation_coefficient_per_km:.8f} km^-1"
        )
    else:
        print("Intrinsic attenuation    : DISABLED")
    print(f"Geometric spreading      : b = {B:g} (amplitude proportional to r^-b)")
    print("=" * 70 + "\n")

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
        USE_SITE_AMPLIFICATION=USE_SITE_AMPLIFICATION,
        SITE_AMPLIFICATION_CSV=active_site_amplification_csv,
        SITE_AMPLIFICATION_COLUMN=SITE_AMPLIFICATION_COLUMN,
        USE_INTRINSIC_ATTENUATION=USE_INTRINSIC_ATTENUATION,
        ATTENUATION_COEFFICIENT_PER_KM=active_attenuation_coefficient_per_km,
        SEARCH_AREA=SEARCH_HALF_WIDTH,
        SEARCH_BOX=SEARCH_BOX,
        SPATIAL_STEP=SPATIAL_STEP,
        TEMPORAL_STEP=TEMPORAL_STEP,
        B=B,
        STORE_FULL_FIELDS=STORE_FULL_FIELDS,
    )

    topo, data, results = run_analysis(cfg)
    save_path = save_output_bundle(SAVE_OUTPUT, cfg, topo, data, results)
    print(f"Saved ASL outputs → {save_path}")
