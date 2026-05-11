# Mesoscale Structure

This repository contains the code used to build and analyze datasets for a research study on stratocumulus mesoscale organization, with a focus on closed and open cell structures and their transitions.

The workflow is split into five main stages:

1. Find EarthCARE orbits that are relevant for the study (all orbits within four stratocumulus regions during Aug. 2024 to Nov. 2025) and prepare EarthCARE ACM-CAP data (`step0_preprocess_EC_ACMCAP/`)
2. Fine-tune and apply the "judge" CNN to MSI in `step1_judge_for_msi/`. Fine-tune the "judge" CNN to GOES in `step2_judge_for_goes/`.
3. Run trajectory calculations and apply the CNN along trajectories on GOES imagery in `step3_backtrajectories/`.
4. Create datasets combining ACM-CAP data with CNN labels along EarthCARE tracks and along trajectories in `step4_create_datasets_combining_EC_with_judge_results/`.
5. Perform final science analysis in the two main notebooks in `step5_analysis_closed_open_cells/`.


## Repository Structure

Only the core code folders are documented here. Figure output folders (`figures_...`) are intentionally omitted.

- `step0_preprocess_EC_ACMCAP/`
	- Preprocessing of EarthCARE ACM-CAP data, including:
		- Finding relevant EarthCARE orbits (over four stratocumulus regions during Aug. 2024 to Nov. 2025).
		- Extraction of relevant variables from the original 2d profiles (e.g. liquid water content) and reduction to 1d variables (e.g. liquid water path) along the EarthCARE track.

- `step1_judge_for_msi/`
	- Fine-tuning of a CNN for classification of EarthCARE-MSI imagery (the original CNN is from Wu et al., 2025 (https://doi.org/10.5194/essd-17-3243-2025); the original CNN can be accessed [here](https://zenodo.org/records/15776241)).
	- Application of the fine-tuned CNN on EarthCARE-MSI imagery for the four stratocumulus regions to produce labeled outputs used downstream.

- `step2_judge_for_goes/`
	- Fine-tuning and application of CNN-based classification for GOES imagery (original model by Wu et al. 2025, see above).
	- Contains a function to label GOES imagery used downstream in `step3_backtrajectories/`.

- `step3_backtrajectories/`
	- Computes backward and forward trajectories from EarthCARE overpasses using ERA5 wind fields at 975 hPa. The computation is done using the [`backtrajectory-calculator` library](https://github.com/licongren/backtrajectory-calculator/) by J.D. Mantilla; last access: 04.02.2026.
	- Applies GOES-based classification along trajectories (see `step2_judge_for_goes/`).
	- Adds GOES channel-14 brightness temperatures along trajectories to help the precise timing of transitions during nighttime.

- `step4_create_datasets_combining_EC_with_judge_results/`
	- creates the two main datasets for the final analysis:
		- `dataset1_judge_msi_along_EC.ipynb`: combines ACM-CAP data along EarthCARE tracks with the CNN-based classification results for EarthCARE-MSI from `step1_judge_for_msi/`.
		- `dataset2_combine_judge_goes_along_trajectories_with_acmcap.ipynb`: combines ACM-CAP data along EarthCARE tracks with the CNN-based classification results for GOES from `step2_judge_for_goes/` applied along trajectories in `step3_backtrajectories/`.

- `step5_analysis_closed_open_cells/`
	- `closed_open_cell_properties.ipynb`: statistical characterization of closed vs open cells across four stratocumulus regions using ACM-CAP data for microphysical properties and MSI classification outputs.
	- `closed_to_open_cell_transitions.ipynb`: analysis of cloud properties around closed-to-open transitions identified from GOES classification plus trajectory context, using ACM-CAP data for microphysical properties.


## Environment Setup

Two environment files are provided:

- `environment_ml.yaml`: machine-learning/classification tasks (judge for MSI and GOES).
- `environment_bt.yaml`: trajectory and related geophysical processing tasks.

Activate whichever environment matches the stage you are running.

## Notes

This folder is organized as a workflow-oriented research codebase rather than a packaged Python library.

## Data Dependencies

This workflow depends on the following data sources/products:

- EarthCARE overpass information used as trajectory start points.
- ACM-CAP data used for cloud and rain properties.
- MSI imagery used for complementary cell-type classification.
- ERA5 wind fields used for backward/forward trajectory computation.
- GOES imagery used for classification and detection of transitions between cloud structures.

Input and intermediate datasets are produced and consumed across the stage folders.

## How To Cite This Code

If you use this codebase in a publication, please cite the associated research paper/preprint (??).

Suggested minimum information:

- Authors: Johanna Mayer
- Year: 2026
- ??
 