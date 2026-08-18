# Ngauruhoe Landslide

Code and supporting materials for analysing a suspected landslide signal recorded at Ngauruhoe, New Zealand, on 21 March 2026. The repository combines seismic amplitude-based source localisation, seismo-acoustic onset-time analysis, figure generation, and manuscript materials for the accompanying study.

## Overview

The main goal of this repository is to test whether an emergent mass-movement signal can be located using:

- amplitude source location (ASL) based on absolute seismic amplitudes across the network,
- seismic amplitude ratio analysis (SARA) based on pairwise relative amplitudes,
- seismo-acoustic differential onset times between seismic and infrasound arrivals,
- supporting comparisons, sensitivity tests, and manuscript figures.

The code is organized more like a research workflow than a packaged software library. Most scripts are intended to be run directly after editing the configuration values near the top of each file.

## Repository Structure

- `AnalysisCodes/`: reusable core implementations for ASL and SARA, plus I/O helpers.
- `ASL_Code/`: legacy ASL-related code retained for comparison.
- `DATA/`: waveform files, station metadata, archive search/download helpers, and related data utilities.
- `InfrasoundAnalysis/`: picking tools and seismo-acoustic / infrasound localisation workflows.
- `MakeFigures/`: scripts used to generate paper figures and reusable `.npz` result bundles.
- `PAPER/`: LaTeX manuscript, bibliography, and figure assets for the paper draft.
- `SourceLocalisation/`: alternative source-localisation methods and comparison workflows.
- `Topography/`: topographic grid used to constrain candidate source locations to the ground surface.
- `Weather/`: weather data products and plotting scripts used for event context.
- `References/`: reference documents used during the study.
- `find_enclosing_radius.py`: helper script for summarising the spatial spread of source-location ensembles.

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

### 3. Seismo-acoustic and infrasound analysis

Main scripts live in `InfrasoundAnalysis/` and include:

- `interactive_pick_arrivals.py`: manual picking of seismic and infrasound onsets,
- `assisted_pick_arrivals.py`: assisted picking with automatic onset suggestions,
- `run_seismoacoustic_location.py`: source localisation from seismic-infrasound differential arrival times,
- `run_infrasound_semblance.py`: sparse-network infrasound semblance localisation.

That folder also contains its own README with more detail on the picking convention and output files.

### 4. Alternative localisation methods

`SourceLocalisation/` contains additional approaches that are less sensitive to absolute amplitudes, including:

- amplitude-ratio localisation,
- envelope-based TDOA localisation,
- hybrid methods,
- jackknife and consensus analyses.

See `SourceLocalisation/README.md` for folder-specific details.

### 5. Figure generation and manuscript production

- `MakeFigures/` contains scripts used to generate publication figures and benchmarking plots.
- `PAPER/Ngauruhoe.tex` is the main manuscript source.

## Data Included in the Repository

The repository currently includes:

- example MiniSEED waveform files for seismic and infrasound data,
- StationXML metadata files,
- precomputed result files and figure outputs in some `RESULTS/` directories,
- topography and weather context files used by the analysis.

Because this is an active research repository, some outputs are intermediate products rather than final archival releases.

## Python Dependencies

The scripts use standard scientific Python tools together with seismological/geospatial packages, including:

- `numpy`
- `pandas`
- `matplotlib`
- `obspy`
- `pyproj`

Some scripts may also rely on other packages depending on the workflow you run. There is currently no pinned environment or packaged installer in this repository, so dependencies are inferred from script imports.

## Typical Usage Pattern

1. Choose the workflow you want to run.
2. Open the relevant script and edit the configuration block near the top if needed.
3. Run the script directly from the repository root or its working folder.
4. Inspect outputs written to the corresponding `RESULTS/` directory.

Examples of entry-point scripts:

- `MakeFigures/run_ASL.py`
- `MakeFigures/run_SARA.py`
- `InfrasoundAnalysis/run_seismoacoustic_location.py`
- `InfrasoundAnalysis/run_infrasound_semblance.py`
- `SourceLocalisation/run_alternative_methods.py`

## Notes

- Candidate source locations are generally constrained to the topographic surface rather than solved at arbitrary depth.
- The repository reflects an active research workflow, so some scripts are exploratory, comparison-focused, or manuscript-specific.
- Subdirectories such as `InfrasoundAnalysis/` and `SourceLocalisation/` include additional README files with more detailed workflow notes.

## Citation

If you use this repository, please cite the associated paper once it is finalized. Until then, cite the repository and describe the specific scripts, data products, and commit or release used in your analysis.

## Status

This repository is a research codebase for one event study, not a polished general-purpose software package. The emphasis is on reproducible scientific analysis and figure generation for the Ngauruhoe landslide case study.
