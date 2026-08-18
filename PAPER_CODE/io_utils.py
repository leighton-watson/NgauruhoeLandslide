from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def save_results_bundle(save_path, bundle):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {}
    metadata = {}
    for key, value in bundle.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
        else:
            metadata[key] = _json_safe(value)

    arrays["_metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(save_path, **arrays)
    return save_path


def load_results_bundle(load_path):
    load_path = Path(load_path)
    with np.load(load_path, allow_pickle=False) as data:
        bundle = {key: data[key] for key in data.files if key != "_metadata_json"}
        metadata = json.loads(data["_metadata_json"].item())
    bundle.update(metadata)
    bundle["bundle_path"] = str(load_path)
    return bundle


def subset_topography(topo, east_limits, north_limits, pad_m=250.0):
    east_min, east_max = east_limits
    north_min, north_max = north_limits

    x = topo["X"][0, :]
    y = topo["Y"][:, 0]

    x_mask = (x >= east_min - pad_m) & (x <= east_max + pad_m)
    y_mask = (y >= north_min - pad_m) & (y <= north_max + pad_m)

    return {
        "X": topo["X"][y_mask][:, x_mask],
        "Y": topo["Y"][y_mask][:, x_mask],
        "C": topo["C"][y_mask][:, x_mask],
    }


def traces_to_matrix(traces):
    if not traces:
        return np.array([], dtype=float), np.empty((0, 0), dtype=float)

    sampling_rate = float(traces[0].stats.sampling_rate)
    npts = int(traces[0].stats.npts)
    time = np.arange(npts, dtype=float) / sampling_rate
    matrix = np.vstack([trace.data.astype(float) for trace in traces])
    return time, matrix


def _parse_time_string(value):
    return datetime.fromisoformat(str(value).replace("Z", ""))


def get_plot_window_seconds(cfg):
    data_start = _parse_time_string(cfg["START_TIME"])
    plot_start = _parse_time_string(cfg.get("PLOT_START_TIME", cfg["START_TIME"]))
    plot_end = _parse_time_string(cfg.get("PLOT_END_TIME", cfg["END_TIME"]))
    return (
        (plot_start - data_start).total_seconds(),
        (plot_end - data_start).total_seconds(),
    )
