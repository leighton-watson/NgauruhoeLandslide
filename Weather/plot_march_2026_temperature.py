from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
CSV_CANDIDATES = (
    HERE / "46771_Temperature_hourly.csv",
    HERE / "46771__Temperature__hourly.csv",
)
OUTPUT_FIGURE = HERE / "march_2026_temperature.png"

TEMPERATURE_COLUMN = "Mean Temperature [Deg C]"
PLOT_START = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
PLOT_END = datetime(2026, 4, 1, 0, 0, 0, tzinfo=timezone.utc)
TARGET_TIMESTAMP = datetime(2026, 3, 21, 13, 36, 15, tzinfo=timezone.utc)

GAMMA = 1.4
R_DRY_AIR = 287.05  # J kg^-1 K^-1


@dataclass(frozen=True)
class Observation:
    timestamp: datetime
    temperature_c: float


def resolve_csv_file() -> Path:
    for csv_file in CSV_CANDIDATES:
        if csv_file.exists():
            return csv_file

    candidates = ", ".join(path.name for path in CSV_CANDIDATES)
    raise FileNotFoundError(f"Could not find any of: {candidates}")


def load_temperature_observations(
    csv_file: Path,
    temperature_column: str,
) -> list[Observation]:
    observations: list[Observation] = []

    with csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp_text = (row.get("Observation time UTC") or "").strip()
            temperature_text = (row.get(temperature_column) or "").strip()
            if not timestamp_text or not temperature_text:
                continue

            timestamp = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
            observations.append(
                Observation(timestamp=timestamp, temperature_c=float(temperature_text))
            )

    observations.sort(key=lambda observation: observation.timestamp)
    return observations


def select_time_range(
    observations: list[Observation],
    start: datetime,
    end: datetime,
) -> list[Observation]:
    return [
        observation
        for observation in observations
        if start <= observation.timestamp < end
    ]


def find_closest_observation(
    observations: list[Observation],
    target_timestamp: datetime,
) -> Observation:
    if not observations:
        raise RuntimeError("No valid temperature observations were loaded.")

    return min(
        observations,
        key=lambda observation: (
            abs(observation.timestamp - target_timestamp),
            observation.timestamp,
        ),
    )


def format_timedelta(delta: timedelta) -> str:
    total_seconds = int(delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s"


def compute_sound_speed(
    temperature_c: float,
    gamma: float = GAMMA,
    gas_constant: float = R_DRY_AIR,
) -> float:
    temperature_k = temperature_c + 273.15
    if temperature_k <= 0.0:
        raise ValueError("Temperature must be above absolute zero.")

    return math.sqrt(gamma * gas_constant * temperature_k)


def plot_temperature_series(
    observations: list[Observation],
    target_timestamp: datetime,
    closest_observation: Observation,
    output_file: Path,
) -> None:
    times = [observation.timestamp for observation in observations]
    temperatures = [observation.temperature_c for observation in observations]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(
        times,
        temperatures,
        color="tab:red",
        linewidth=1.6,
        label="Hourly mean temperature",
    )
    ax.axvline(
        target_timestamp,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Target time ({target_timestamp:%d %b %Y %H:%M:%S UTC})",
    )
    ax.scatter(
        [closest_observation.timestamp],
        [closest_observation.temperature_c],
        color="navy",
        s=70,
        zorder=3,
        label=(
            "Closest observation "
            f"({closest_observation.timestamp:%d %b %Y %H:%M UTC})"
        ),
    )

    ax.set_title("Mean Temperature During March 2026 (UTC)")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Mean Temperature (deg C)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlim(PLOT_START, PLOT_END)
    ax.legend()

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.show()


def main() -> None:
    csv_file = resolve_csv_file()
    all_observations = load_temperature_observations(csv_file, TEMPERATURE_COLUMN)
    march_observations = select_time_range(all_observations, PLOT_START, PLOT_END)
    if not march_observations:
        raise RuntimeError("No March 2026 temperature observations were found.")

    closest_observation = find_closest_observation(all_observations, TARGET_TIMESTAMP)
    temperature_k = closest_observation.temperature_c + 273.15
    sound_speed = compute_sound_speed(closest_observation.temperature_c)
    time_offset = abs(closest_observation.timestamp - TARGET_TIMESTAMP)

    print(f"Loaded CSV file: {csv_file}")
    print(f"Using temperature column: {TEMPERATURE_COLUMN}")
    print(
        f"Plotting {len(march_observations)} observations from "
        f"{PLOT_START:%Y-%m-%d %H:%M:%S %Z} to {PLOT_END:%Y-%m-%d %H:%M:%S %Z}"
    )
    print(f"Target timestamp: {TARGET_TIMESTAMP:%Y-%m-%d %H:%M:%S %Z}")
    print(
        "Closest observation: "
        f"{closest_observation.timestamp:%Y-%m-%d %H:%M:%S %Z} "
        f"(offset {format_timedelta(time_offset)})"
    )
    print(f"Mean temperature: {closest_observation.temperature_c:.2f} deg C")
    print(f"Temperature used in c = sqrt(gamma * R * T): {temperature_k:.2f} K")
    print(f"gamma = {GAMMA:.3f}")
    print(f"R = {R_DRY_AIR:.2f} J kg^-1 K^-1")
    print(f"Sound speed: {sound_speed:.3f} m/s")

    plot_temperature_series(
        observations=march_observations,
        target_timestamp=TARGET_TIMESTAMP,
        closest_observation=closest_observation,
        output_file=OUTPUT_FIGURE,
    )
    print(f"Saved figure to: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
