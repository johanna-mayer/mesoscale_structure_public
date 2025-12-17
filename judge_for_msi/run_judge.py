from typing import Tuple
import sys
import json
import gc
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import judge
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir + '/../../../ectools/')
from ectools import ecio

# Specification of directories and files
OUTPUT_FILE = '/home/jmayer/ACI_research/Code/resources/judge_results_Aug.csv'
OUTPUT_PLOT = current_dir + '/../figures_judge/'
INPUT_FILE_files_in_sc_regions = '/home/jmayer/ACI_research/Code/resources/files_in_sc_regions.json'
FAILED_FILES_LOG = '/home/jmayer/ACI_research/Code/resources/judge_failed_files_Aug.txt'
MCOP_data_dir = "/home/jmayer/Object_Data/MSI_COP_2A/"
MRGR_data_dir = "/home/jmayer/Object_Data/MSI_RGR_1C/"

stratus_regions = {
    'SEA': {'lat': (-25, -2.5), 'lon': (-5, 15)},    # South East Atlantic
    'NEA': {'lat': (10, 30), 'lon': (-40, -20)},   # North East Atlantic
    'SEP': {'lat': (-25, -2.5), 'lon': (-95, -75)},  # South East Pacific
    'NEP': {'lat': (15, 35), 'lon': (-135, -115)}, # North East Pacific
}

def printProgressBar(iteration: int, total: int, prefix: str = '', suffix: str = '', 
                    decimals: int = 1, length: int = 50, fill: str = '█', printEnd: str = "\r") -> None:
    """Print a simple progress bar."""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    print()
    if iteration == total: 
        print()


def find_MCOP_and_MRGR_files_matching_ACMCAP_filename(ACMCAP_filename: str, dir_MCOP: str, dir_MRGR: str) -> Tuple[str, str]:
    """Find MCOP and MRGR files that match the ACMCAP filename."""
    import re 

    day = ACMCAP_filename.split('/')[-2]
    month = ACMCAP_filename.split('/')[-3]
    year = ACMCAP_filename.split('/')[-4]
    ACMCAP_filename = ACMCAP_filename.split('/')[-1]
    production_datetime = ACMCAP_filename.split('_')[-2]
    start_datetime = ACMCAP_filename.split('_')[-3]
    tmp = ACMCAP_filename.replace('ACM_CAP_2B', 'MSI_COP_2A').replace(production_datetime, '*').replace(start_datetime, start_datetime[0:9]+'*')
    tmp = re.sub(r'ECA_EX[A-Z]{2}', 'ECA_EX*', tmp)
    filenames_MCOP = glob.glob(f'{dir_MCOP}/{year}/{month}/{day}/{tmp}')

    tmp = ACMCAP_filename.replace('ACM_CAP_2B', 'MSI_RGR_1C').replace(production_datetime, '*').replace(start_datetime, start_datetime[0:9]+'*')
    tmp = re.sub(r'ECA_EX[A-Z]{2}', 'ECA_EX*', tmp)
    filenames_MRGR = glob.glob(f'{dir_MRGR}/{year}/{month}/{day}/{tmp}')

    if len(filenames_MCOP) == 0:
        raise FileNotFoundError('no_MCOP_file_found')
    elif len(filenames_MRGR) == 0:
        raise FileNotFoundError('no_MRGR_file_found')
    elif len(filenames_MCOP) > 1:
        raise FileExistsError('more_than_1_MCOP_file_found')
    elif len(filenames_MRGR) > 1:
        raise FileExistsError('more_than_1_MRGR_file_found')
    else:
        return filenames_MCOP[0], filenames_MRGR[0]

def trim_to_stcu_region(data, region, frame):
    lat_min, lat_max = stratus_regions[region]['lat']
    lon_min, lon_max = stratus_regions[region]['lon']

    data_trimmed = data.where(
        (data.latitude > lat_min) & (data.latitude < lat_max) & 
        (data.longitude > lon_min) & (data.longitude < lon_max), 
        drop=True
    )

    # make sure that there is no overlap between D and E frames
    if frame == 'D':
        data_trimmed = data_trimmed.where(
            data_trimmed.latitude >= 22.5
        )
    if frame == 'E':
        data_trimmed = data_trimmed.where(
            data_trimmed.latitude <= 22.5
        )
    return data_trimmed

def plot_judge_results(MCOP, MRGR, df, orbit, region, year, month, day, save_to=None):
    sns.set_context('talk', font_scale=0.8)
    fig, axs = plt.subplots(2, 1, figsize=(35,6), sharex=True)
    plt.suptitle(
        f'Orbit {orbit}, Region {region}, Date 2025-{month}-{day}, '
        f'lat: {np.nanmax(MCOP.latitude.values[0,:]):.2f} to {np.nanmax(MCOP.latitude.values[-1,:]):.2f}, '
        f'lon: {np.nanmax(MCOP.longitude.values[0,:]):.2f} to {np.nanmax(MCOP.longitude.values[-1,:]):.2f}',
        fontsize = 20
    )
    MCOP.cloud_optical_thickness.T.plot(ax=axs[0], vmin=1, vmax=50)
    cbar = axs[0].collections[0].colorbar
    cbar.set_label('COT', fontsize=20)

    MRGR.TIR2.T.plot(ax=axs[1], vmin=273, vmax=300)
    cbar = axs[1].collections[0].colorbar
    cbar.set_label('TIR2 (K)', fontsize=20)

    hue = np.where(df.ice_flag > 0.01, 6, df.cats)
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e',  '#d62728', '#e377c2', '#8c564b', 'white'] # last color is for ice
    palette = {0: colors[0], 1: colors[1], 2: colors[2], 3: colors[3], 4: colors[4], 5: colors[5], 6: colors[6]}
    # Marker size reflects certainty, ensure it's a numpy array
    marker_sizes = np.array(df.cert).astype(float).ravel() * 250  # scale factor, adjust as needed
    for i in [0,1]:
        legend = True if i == 1 else False
        scatter = sns.scatterplot(ax=axs[i], x = df.along_track_mid, y=MRGR.dims['across_track'] - 128, hue=hue, size=df.cert, size_norm=(0.3,1),
                        marker='o', palette=palette, legend=legend)
        if legend and scatter.legend_ is not None:
            scatter.legend_.set_title('')
    fig.tight_layout()
    if save_to is not None:
        plt.savefig(save_to, dpi=300)
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    logging.info('Starting judge processing script.')
    with open(INPUT_FILE_files_in_sc_regions, 'r') as file:
        files_in_stratus_regions = json.load(file)

    if os.path.exists(OUTPUT_FILE):
        df_results = pd.read_csv(OUTPUT_FILE)
        df_results['time_mid'] = pd.to_datetime(df_results['time_mid'])
        computed_orbits = df_results['orbit'].unique().tolist()
        logging.info(f"Loaded existing results from {OUTPUT_FILE}, found {len(computed_orbits)} computed orbits.")
    else:
        logging.info(f"No existing results found. Starting fresh.")
        df_results = pd.DataFrame()
        computed_orbits = []

    failed_files = []
    # Clear previous failed files log
    open(FAILED_FILES_LOG, 'w').close()
    for region, filenames in files_in_stratus_regions.items():
        j = 0
        logging.info(f"Processing region: {region}, {len(filenames)} files.")
        for i, filename in enumerate(filenames):
            orbit = filename[-9:-3]
            if orbit[-1] not in ['D','E']:
                continue
            if orbit in computed_orbits:
                logging.info(f"Skipping already computed orbit {orbit}.")
                continue
            # TODO: Remove temporal filter
            if ('202508' not in filename):
                continue

            logging.info(f'Processing file {i+1} of {len(filenames)} in {region}: {filename}')
            printProgressBar(i, len(filenames), prefix='Progress:', suffix='Complete', length=50)

            try:
                MCOP_file, MRGR_file = find_MCOP_and_MRGR_files_matching_ACMCAP_filename(filename, MCOP_data_dir, MRGR_data_dir)
            except (FileNotFoundError, FileExistsError) as e:
                logging.error(f"Error finding files for {filename}: {e}")
                failed_files.append(filename)
                with open(FAILED_FILES_LOG, 'a') as f:
                    f.write(filename + '\n')
                continue

            logging.info(f"Loading MCOP: {MCOP_file}")
            logging.info(f"Loading MRGR: {MRGR_file}")
            MCOP = ecio.load_MCOP(MCOP_file)
            MRGR = ecio.load_MRGR(MRGR_file)
            MCOP = trim_to_stcu_region(MCOP, region, orbit[-1])
            MRGR = trim_to_stcu_region(MRGR, region, orbit[-1])
            MRGR = MRGR.where(MRGR.TIR1 < 1e30)

            logging.info('Running judge...')
            try:
                cats, cert, scene_params = judge.judge_MSI(MCOP, MRGR, debugging=False)
            except Exception as e:
                logging.error(f"Error during judging for {filename}: {e}")
                failed_files.append(filename)
                with open(FAILED_FILES_LOG, 'a') as f:
                    f.write(filename + '\n')
                continue
            if len(cats) == 0:
                logging.warning(f"No valid scenes found for {filename}. Skipping.")
                continue
            logging.info('Judge completed.')

            df = pd.DataFrame(scene_params)
            df['cats'] = cats
            df['cert'] = cert
            df['orbit'] = orbit
            df_results = pd.concat([df_results, df], ignore_index=True)

            logging.info('Plotting results...\n')
            plot_judge_results(MCOP, MRGR, df, orbit, region, 
                               filename.split('/')[-4], filename.split('/')[-3], filename.split('/')[-2],
                               save_to=f'{OUTPUT_PLOT}/judge_{orbit}_{region}.png')
            # Save intermediate results
            j += 1
            if j % 5 == 0:
                df_results.to_csv(OUTPUT_FILE, index=False)
                logging.info(f"Intermediate results saved to {OUTPUT_FILE}\n")

            del MCOP, MRGR, df
            gc.collect()

    df_results.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"Results saved to {OUTPUT_FILE}")
    if failed_files:
        logging.warning("The following files failed during processing:")
        for f in failed_files:
            logging.warning(f)
        # Write all failed files at the end as well
        with open(FAILED_FILES_LOG, 'a') as f:
            for ff in failed_files:
                f.write(ff + '\n')
    logging.info('Judge processing script finished.')

if __name__ == "__main__":
    main()
