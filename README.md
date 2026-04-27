# Mesoscale Structure

This repository contains the code used to build and analyze datasets for a research study on stratocumulus mesoscale organization, with a focus on closed and open cell structures and their transitions.

The workflow is split into three main stages:

1. Classification dataset/model preparation (`judge_for_msi`, `judge_for_goes`)
2. Trajectory-based processing (`backtrajectories`)
3. Scientific analysis (`analysis`)

## Repository Structure (Relevant Folders)

Only the core code folders are documented here. Figure output folders (`figures_...`) are intentionally omitted.

- `judge_for_msi/`
	- Fine-tuning and application of CNN-based classification for MSI imagery (the original CNN is from Wu et al., 2025 (https://doi.org/10.5194/essd-17-3243-2025); the original CNN can be accessed [here](https://zenodo.org/records/15776241)).
	- Produces labeled MSI outputs used downstream.

- `judge_for_goes/`
	- Fine-tuning and application of CNN-based classification for GOES imagery (original model by Wu et al. 2025, see above).
	- Contains a function to label GOES imagery used downstream in `backtrajectories/`.

- `backtrajectories/`
	- Computes backward and forward trajectories from EarthCARE overpasses using ERA5 wind fields at 975 hPa. The computation is done using the [`backtrajectory-calculator` library](https://github.com/licongren/backtrajectory-calculator/) by J.D. Mantilla; last access: 04.02.2026.
	- Applies GOES-based classification along trajectories (see `judge_for_goes/`).
	- Adds GOES channel-14 brightness temperatures along trajectories to help the precise timing of transitions during nighttime.

- `analysis/`
	- `closed_open_cell_properties.ipynb`: statistical characterization of closed vs open cells across four stratocumulus regions using ACM-CAP data for microphysical properties and MSI classification outputs.
	- `transitions_EC_GOES.ipynb`: analysis of cloud properties around closed-to-open transitions identified from GOES classification plus trajectory context, using ACM-CAP data for microphysical properties.

## Typical End-to-End Workflow

1. Find EarthCARE orbits that are relevant for the study (all orbits within four stratocumulus regions during Aug. 2024 to Nov. 2025). 
2. Retrieve the ACM-CAP product (2d profiles) from EarthCARE for these orbits, and prepare a dataset containing the variables of interest (reduced to 1d).
3. Prepare labeled MSI and GOES datasets in `judge_for_msi/` and `judge_for_goes/`.
4. Run trajectory calculations and trajectory-linked classification in `backtrajectories/`.
5. Perform final science analysis in the two main notebooks in `analysis/`.

## Environment Setup

Two environment files are provided:

- `environment_ml.yaml`: machine-learning/classification tasks (MSI/GOES judging).
- `environment_bt.yaml`: trajectory and related geophysical processing tasks.

Activate whichever environment matches the stage you are running.

## Notes

This folder is organized as a workflow-oriented research codebase rather than a packaged Python library.

## Data Dependencies

This workflow depends on the following data sources/products:

- EarthCARE overpass information used as trajectory start points.
- ACM-CAP data used for cloud and rain properties.
- ERA5 wind fields used for backward/forward trajectory computation.
- GOES imagery used for classification and detection of transitions between cloud structures.
- MSI imagery used for complementary cell-type classification.

Input and intermediate datasets are produced and consumed across the stage folders (`judge_for_msi/`, `judge_for_goes/`, `backtrajectories/`, `analysis/`).

## How To Cite This Code

If you use this codebase in a publication, please cite the associated research paper/preprint (??).

Suggested minimum information:

- Authors: Johanna Mayer
- Year: 2026
- ??
 