# Ngauruhoe Landslide

Code for locating emergent seismic and infrasound signals focusing on a suspected landslide signal recorded at Ngauruhoe, New Zealand, on 21 March 2026. 

The repository includes code for:
- amplitude source location (ASL) based on absolute seismic amplitudes across the network,
- seismic amplitude ratio analysis (SARA) based on pairwise relative amplitudes,
- seismo-acoustic differential onset times between seismic and infrasound arrivals,

The code is organized as a research workflow rather than a packaged software library. Most scripts are intended to be run directly after editing the configuration values near the top of each file.

## Repository Structure

- `AnalysisCodes/`: reusable core implementations for ASL and SARA, plus I/O helpers
- `DATA/`: waveform files, station metadata, and code for downloading data
- `MakeFigures/`: scripts used to generate paper figures 
- `Topography/`: topographic grid used to constrain candidate source locations to the ground surface.
- `Weather/`: weather data products and plotting scripts

## Main Workflows

### 1. Seismic amplitude source location

Core implementation:

- `AnalysisCodes/asl.py`

Example entry point:

- `MakeFigures/run_ASL.py`

This workflow:

- loads topography from `Topography/topography_big.npz`,
- reads seismic waveform and station metadata from `DATA/`,
- computes moving RMS amplitudes,
- searches for the best-fitting source location on the topographic surface,
- saves a reusable output bundle to `MakeFigures/RESULTS/ASL_output.npz`.

### 2. Seismic amplitude ratio analysis

Core implementation:

- `AnalysisCodes/sara.py`

Example entry point:

- `MakeFigures/run_SARA.py`

This workflow uses pairwise station-amplitude ratios rather than absolute amplitudes and writes reusable outputs to `MakeFigures/RESULTS/SARA_output.npz`.

## Data Included in the Repository

The repository currently includes:

- MiniSEED waveform files for seismic and infrasound data for the Ngauruhoe landslide,
- StationXML metadata files,
- topography and weather files 

There are several tutorials at https://github.com/GeoNet/data-tutorials that provide instructions on how to download the data. See specifically the https://github.com/GeoNet/data-tutorials/tree/main/FDSN or https://github.com/GeoNet/data-tutorials/tree/main/AWS_Open_Data tutorials. 

## Python Dependencies

The scripts use standard scientific Python tools together with seismological/geospatial packages, including:

- `numpy`
- `pandas`
- `matplotlib`
- `obspy`
- `pyproj`

Some scripts may also rely on other packages depending on the workflow you run. There is currently no pinned environment or packaged installer in this repository, so dependencies are inferred from script imports.

## Who to contact

Leighton Watson, University of Canterbury
Leighton.watson@canterbury.ac.nz
