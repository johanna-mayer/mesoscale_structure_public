# Mesoscale Structure

This repository contains the code used to build and analyze datasets for a research study on stratocumulus mesoscale organization, with a focus on closed and open cell structures and their transitions. Please refer to the publication for details on the study: https://doi.org/10.5194/acp-26-14085-2026.

The workflow is split into five main stages:

1. Find EarthCARE orbits that are relevant for the study (all orbits within four stratocumulus regions during Aug. 2024 to Nov. 2025) and prepare EarthCARE ACM-CAP data (`step0_preprocess_EC_ACMCAP/`)
2. Fine-tune and apply the "judge" CNN to MSI in `step1_judge_for_msi/`. Fine-tune the "judge" CNN to GOES in `step2_judge_for_goes/`.
3. Run trajectory calculations and apply the CNN along trajectories on GOES imagery in `step3_backtrajectories/`.
4. Create datasets combining ACM-CAP data with CNN labels along EarthCARE tracks and along trajectories in `step4_create_datasets_combining_EC_with_judge_results/`.
5. Perform final science analysis in the two main notebooks in `step5_analysis_closed_open_cells/`.


## Repository Structure

The repository is organized into folders with the prefix `stepX_` corresponding to the order of the workflow. Within each folder, scripts with the prefix `stepXX_` contain the code to reproduce the datasets and the final analysis of our study. Scripts without the `stepXX_` prefix are either utility or preparatory scripts, or notebooks for testing and exploration.

- `step0_preprocess_EC_ACMCAP/`
	- Preprocessing of EarthCARE ACM-CAP data, including:
		- `step00_frames_in_sc_regions_via_maap.ipynb`: Finding relevant EarthCARE orbits (over four stratocumulus regions during Aug. 2024 to Nov. 2025).
		- `step01_create_dataset_of_EC_ACMCAP_in_sc_regions.py`: Extraction of relevant variables from the original ACM-CAP 2d profiles (e.g. liquid water content) and reduction to 1d variables (e.g. liquid water path) along the EarthCARE track for all relevant orbits.

- `step1_judge_for_msi/`
	- Preparatory work in `fine_tune_judge_for_msi.ipynb`: Fine-tuning of a CNN for classification of EarthCARE-MSI imagery (the original CNN is from Wu et al., 2025 (https://doi.org/10.5194/essd-17-3243-2025); the original CNN can be accessed [here](https://zenodo.org/records/15776241)). The output is a fine-tuned model for MSI classification that is used downstream.
	- Utility functions to apply the fine-tuned CNN to MSI imagery are contained in `judge_msi_utils.py` and `judge_msi.py`. 
	- `step10_run_judge_msi.ipynb`: Application of the fine-tuned CNN on EarthCARE-MSI imagery for the four stratocumulus regions to produce labeled outputs used downstream.

- `step2_judge_for_goes/`
	- Preparatory work in `fine_tune_judge_for_goes.ipynb`: Fine-tuning and application of CNN-based classification for GOES imagery (original model by Wu et al. 2025, see above). The output is a fine-tuned model for GOES classification that is used downstream.
	- Utility functions to apply the fine-tuned CNN to GOES imagery are contained in `functions_for_goes.py` and `judge_goes.py`. The CNN is applied to GOES data downstream in `step3_backtrajectories/`.

- `step3_backtrajectories/`
	- `step30_compute_trajectories.py`: Computes backward and forward trajectories starting from EarthCARE overpasses using ERA5 wind fields at 975 hPa. The computation is done using the [`backtrajectory-calculator` library](https://github.com/licongren/backtrajectory-calculator/) by J.D. Mantilla; last access: 04.02.2026.
	- `step31_judge_along_trajectories.py`: Applies CNN classification (see `step2_judge_for_goes/`) on GOES data along trajectories.
	- `step32_goes_BT14_along_trajectories.py`: Adds GOES channel-14 brightness temperatures along trajectories to help the precise timing of transitions during nighttime.

- `step4_create_datasets_combining_EC_with_judge_results/`
	- creates the two main datasets for the final analysis:
		- `step40_dataset1_judge_msi_along_EC.ipynb`: combines ACM-CAP data along EarthCARE tracks (output of `step01_create_dataset_of_EC_ACMCAP_in_sc_regions.py`) with the CNN-based classification results for EarthCARE-MSI (output of `step10_run_judge_msi.ipynb`).
		- `step41_dataset2_combine_judge_goes_along_trajectories_with_acmcap.ipynb`: combines ACM-CAP data along EarthCARE tracks (output of `step40_dataset1_judge_msi_along_EC.ipynb`) with the CNN-based classification results for GOES applied along trajectories (output of `step31_judge_along_trajectories.py` and `step32_goes_BT14_along_trajectories.py`).

- `step5_analysis_closed_open_cells/`
	- `step50_closed_open_cell_properties.ipynb`: statistical characterization of closed vs open cells across four stratocumulus regions using ACM-CAP data for microphysical properties and MSI classification outputs.
	- `step51_closed_to_open_cell_transitions.ipynb`: analysis of cloud properties around closed-to-open transitions identified from GOES classification plus trajectory context, using ACM-CAP data for microphysical properties.


## Environment Setup

Two environment files are provided:

- `environment_ml.yaml`: machine-learning/classification tasks (judge for MSI and GOES).
- `environment_bt.yaml`: trajectory and analysis tasks.

Activate whichever environment matches the stage you are running.

## Notes

This folder is organized as a workflow-oriented research codebase rather than a packaged Python library.

## Data Dependencies

This workflow depends on the following data sources/products:

- Access to the EarthCARE MAAP system to download EarthCARE data (orbit information, ACM-CAP, M-COP, M-RGR).
- ERA5 wind fields used for backward/forward trajectory computation and the associated `backtrajectory-calculator` library by J.D. Mantilla (see above).
- GOES imagery used for classification and detection of transitions between cloud structures.

Input and intermediate datasets are produced and consumed across the stage folders.

## How To Cite This Code

If you use this codebase in a publication, please cite the associated research paper:

J. Mayer et al., 2026. EarthCARE reveals details on the role of rain in closed-to-open cell transitions. Atmospheric Chemistry and Physics. doi: 10.5194/acp-26-14085-2026

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
 