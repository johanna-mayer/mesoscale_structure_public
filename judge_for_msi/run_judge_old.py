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
import psutil
import judge_msi
sys.path.append('/home/jmayer/ACI_research/Code/mesoscale_structure/judge_for_msi/')
import access_ec_on_maap as maap
current_dir = os.path.dirname(os.path.abspath(__file__))


regions_of_interest = ['SEA'] #, 'NEA', 'SEP', 'NEP'
# Specification of directories and files
OUTPUT_FILE = f'/home/jmayer/ACI_research/Code/resources_mesoscale_structure/judge_msi_results_{regions_of_interest[0]}.csv'
OUTPUT_PLOT = current_dir + '/../figures_judge_msi/'
INPUT_FILE_orbits_in_sc_regions = '/home/jmayer/ACI_research/Code/resources_mesoscale_structure/acmcap_orbits_in_sc_regions.json'
FAILED_FILES_LOG = f'/home/jmayer/ACI_research/Code/resources_mesoscale_structure/judge_msi_failed_orbits_{regions_of_interest[0]}.txt'


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


def DEPRECATED_find_MCOP_and_MNOM_files_matching_ACMCAP_filename(ACMCAP_filename: str, dir_MCOP: str, dir_MNOM: str) -> Tuple[str, str]:
    """Find MCOP and MNOM files that match the ACMCAP filename."""
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
    filenames_MNOM = glob.glob(f'{dir_MNOM}/{year}/{month}/{day}/{tmp}')

    if len(filenames_MCOP) == 0:
        raise FileNotFoundError('no_MCOP_file_found')
    elif len(filenames_MNOM) == 0:
        raise FileNotFoundError('no_MNOM_file_found')
    elif len(filenames_MCOP) > 1:
        raise FileExistsError('more_than_1_MCOP_file_found')
    elif len(filenames_MNOM) > 1:
        raise FileExistsError('more_than_1_MNOM_file_found')
    else:
        return filenames_MCOP[0], filenames_MNOM[0]

def DEPRECATED_trim_to_stcu_region(data, region, frame):
    stratus_regions = {
        'SEA': {'lat': (-22.5, -2.5), 'lon': (-5, 15)},    # South East Atlantic
        'NEA': {'lat': (10, 30), 'lon': (-40, -20)},   # North East Atlantic
        'SEP': {'lat': (-22.5, -2.5), 'lon': (-95, -75)},  # South East Pacific
        'NEP': {'lat': (15, 35), 'lon': (-135, -115)}, # North East Pacific
    }
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


def load_msi_data_via_maap(region, frame, orbitNumber):
    print('.' * 85)
    print(f"Processing orbit {orbitNumber}{frame}...")
    
    print("Loading MCOP data...")
    MCOP_list = maap.get_mcop_via_maap(
                        region=region, 
                        frame=frame,
                        orbitNumber=orbitNumber,
                        mcop_variables=['time', 'latitude', 'longitude', 'cloud_optical_thickness', 'cloud_water_path'], 
                        max_items=1)

    print("\n\nLoading MNOM data...")
    MNOM_list = maap.get_mnom_via_maap(
                        region=region, 
                        frame=frame,
                        orbitNumber=orbitNumber,
                        mnom_variables=['time', 'latitude', 'longitude', 'TIR1', 'TIR2', 'TIR3','land_flag'],
                        max_items=1)

    # for simplicity we only use the first item in the list (most of the time only one overpass exists per day per region)
    return MCOP_list[0], MNOM_list[0]


def plot_judge_results(MCOP, MNOM, df, orbit, region, year, month, day, save_to=None):
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

    MNOM.TIR2.T.plot(ax=axs[1], vmin=273, vmax=300)
    cbar = axs[1].collections[0].colorbar
    cbar.set_label('TIR2 (K)', fontsize=20)

    hue = np.where(df.ice_flag > 0.01, 6, df.cats)
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#e377c2', '#8c564b', 'white'] # last color is for ice
    palette = {0: colors[0], 1: colors[1], 2: colors[2], 3: colors[3], 4: colors[4], 5: colors[5], 6: colors[6]}
    # Marker size reflects certainty, ensure it's a numpy array
    marker_sizes = np.array(df.cert).astype(float).ravel() * 250  # scale factor, adjust as needed
    for i in [0,1]:
        legend = True if i == 1 else False
        scatter = sns.scatterplot(ax=axs[i], x = df.along_track_mid, y=MNOM.dims['across_track'] - 128, hue=hue, size=df.cert, size_norm=(0.3,1),
                        marker='o', palette=palette, legend=legend)
        if legend and scatter.legend_ is not None:
            scatter.legend_.set_title('')
    fig.tight_layout()
    if save_to is not None:
        plt.savefig(save_to, dpi=150)
    else:
        plt.show()
    # Force close figure and clear all matplotlib memory
    plt.close(fig)
    plt.close('all')
    plt.cla()
    plt.clf()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    logging.info('Starting judge processing script.')

    # Monitor initial memory
    process = psutil.Process(os.getpid())
    logging.info(f"Initial memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    
    with open(INPUT_FILE_orbits_in_sc_regions, 'r') as file:
        orbits_in_stratus_regions = json.load(file)

    if os.path.exists(OUTPUT_FILE):
        df_results = pd.read_csv(OUTPUT_FILE)
        df_results['time_mid'] = pd.to_datetime(df_results['time_mid'])
        computed_orbits = df_results['orbit'].unique().tolist()
        logging.info(f"Loaded existing results from {OUTPUT_FILE}, found {len(computed_orbits)} computed orbits.")
    else:
        logging.info(f"No existing results found. Starting fresh.")
        df_results = pd.DataFrame()
        computed_orbits = []

    failed_orbits = []
    # Clear previous failed files log
    #open(FAILED_FILES_LOG, 'w').close()
    for region in regions_of_interest:
        number_of_orbits_to_process = len(orbits_in_stratus_regions[region])
        j = 0
        logging.info(f"Processing region: {region}, {number_of_orbits_to_process} orbits.")
        for i, item in enumerate(orbits_in_stratus_regions[region]):
            frame = item['frame']
            orbitNumber = item['orbitNumber']
            orbit = orbitNumber+frame
            filename = item['filename']
            date = filename.split('ACM_CAP_2B_')[1].split('_')[0][0:8]
            year = date[0:4]
            month = date[4:6]
            day = date[6:8]
            #local_file_path = f"{ACMCAP_data_dir}{year}/{month}/{day}/{filename}"
            
            if frame not in ['D','E']:
                continue
            if orbit in computed_orbits:
                logging.info(f"Skipping already computed orbit {orbit}.")
                continue
            # TODO: Remove temporal filter
            #if ('2025010' not in filename):
            #    continue

            logging.info(f'Processing orbit {i+1} of {number_of_orbits_to_process} in {region}: {orbit}')
            printProgressBar(i, number_of_orbits_to_process, prefix='Progress:', suffix='Complete', length=50)

            # Initialize variables to None for cleanup
            MCOP = None
            MNOM = None
            df = None
            cats = None
            cert = None
            scene_params = None

            try: 
                (MCOP, orbit), (MNOM, orbit) = load_msi_data_via_maap(region=region, frame=frame, orbitNumber=orbitNumber)
                MNOM = MNOM.where(MNOM.TIR1 < 1e30)
            except (FileNotFoundError, FileExistsError) as e:
                logging.error(f"Error finding files for {orbit}: {e}")
                failed_orbits.append(orbit)
                with open(FAILED_FILES_LOG, 'a') as f:
                    f.write(orbit + '\n')
                continue

            logging.info('Running judge...')
            try:
                cats, cert, scene_params = judge_msi.judge_MSI(MCOP, MNOM, debugging=False)
            
            except Exception as e:
                logging.error(f"Error during judging for {orbit}: {e}")
                failed_orbits.append(orbit)
                with open(FAILED_FILES_LOG, 'a') as f:
                    f.write(orbit + '\n')
                # Clean up and continue
                try:
                    MCOP.close()
                    MNOM.close()
                except:
                    pass
                del MCOP, MNOM, cats, cert, scene_params
                gc.collect()
                continue

            if len(cats) == 0:
                logging.warning(f"No valid scenes found for {orbit}. Skipping.")
                # Clean up and continue
                try:
                    MCOP.close()
                    MNOM.close()
                except:
                    pass
                del MCOP, MNOM, cats, cert, scene_params
                gc.collect()
                continue

            logging.info('Judge completed.')

            df = pd.DataFrame(scene_params)
            df['cats'] = cats
            df['cert'] = cert
            df['orbit'] = orbit
            df_results = pd.concat([df_results, df], ignore_index=True)

            logging.info('Plotting results...\n')
            plot_judge_results(MCOP, MNOM, df, orbit, region, 
                               year, month, day,
                               save_to=f'{OUTPUT_PLOT}/judge_{orbit}_{region}.png')
            
            # cleanup
            try:
                MCOP.close()
                MNOM.close()
            except:
                pass
            del MCOP, MNOM, df, cats, cert, scene_params
            gc.collect()

            # Save intermediate results
            j += 1
            if j % 5 == 0:
                df_results.to_csv(OUTPUT_FILE, index=False)
                logging.info(f"Intermediate results saved to {OUTPUT_FILE}\n")

            # Monitor memory usage
            if j % 1 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                logging.info(f"Current memory usage: {current_memory:.1f} MB")
                
                # Force garbage collection if memory is high
                if current_memory > 8000:  # 8GB threshold
                    logging.warning("High memory usage detected, forcing garbage collection...")
                    gc.collect()
                    
                # Emergency break if memory is too high
                if current_memory > 12000:  # 12GB threshold
                    logging.error("Memory usage too high, stopping processing")
                    break

    # Final save of results
    df_results.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"Results saved to {OUTPUT_FILE}")
    if failed_orbits:
        logging.warning("The following files failed during processing:")
        for f in failed_orbits:
            logging.warning(f)
        # Write all failed files at the end as well
        with open(FAILED_FILES_LOG, 'a') as f:
            for ff in failed_orbits:
                f.write(ff + '\n')
    logging.info('Judge processing script finished.')

if __name__ == "__main__":
    main()
