from typing import Tuple
import sys
import json
import gc
import logging
import pandas as pd
import numpy as np
import os
import multiprocessing
import traceback
sys.path.append('/home/jmayer/ACI_research/Code/mesoscale_structure/')
import access_ec_on_maap as maap
current_dir = os.path.dirname(os.path.abspath(__file__))


regions_of_interest = ['SEA'] #, 'NEA', 'SEP', 'NEP'
# Specification of directories and files
OUTPUT_FILE = f'/home/jmayer/ACI_research/Code/resources_mesoscale_structure/judge_msi_results_{regions_of_interest[0]}.csv'
OUTPUT_PLOT = current_dir + '/../figures_judge_msi/'
INPUT_FILE_orbits_in_sc_regions = '/home/jmayer/ACI_research/Code/resources_mesoscale_structure/acmcap_orbits_in_sc_regions.json'
FAILED_FILES_LOG = f'/home/jmayer/ACI_research/Code/resources_mesoscale_structure/judge_msi_failed_orbits_{regions_of_interest[0]}.txt'


def process_single_orbit(args):
    """Process a single orbit in isolation to prevent memory leaks."""
    region, frame, orbitNumber, orbit, year, month, day, computed_orbits = args
    
    # Import modules inside the function to ensure fresh imports
    import judge_msi
    import access_ec_on_maap as maap
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    import gc
    import psutil
    
    print(f'Processing orbit {orbit}')
    
    try:
        # Load data
        print("Loading MCOP data...")
        MCOP_list = maap.get_mcop_via_maap(
            region=region, 
            frame=frame,
            orbitNumber=orbitNumber,
            mcop_variables=['time', 'latitude', 'longitude', 'cloud_optical_thickness', 'cloud_water_path'], 
            max_items=1)
        
        print("Loading MNOM data...")
        MNOM_list = maap.get_mnom_via_maap(
            region=region, 
            frame=frame,
            orbitNumber=orbitNumber,
            mnom_variables=['time', 'latitude', 'longitude', 'TIR1', 'TIR2', 'TIR3','land_flag'],
            max_items=1)
        
        MCOP, orbit_mcop = MCOP_list[0]
        MNOM, orbit_mnom = MNOM_list[0]
        MNOM = MNOM.where(MNOM.TIR1 < 1e30)
        
        # Run judge
        print('Running judge...')
        cats, cert, scene_params = judge_msi.judge_MSI(MCOP, MNOM, debugging=False)
        
        if len(cats) == 0:
            return {'status': 'no_scenes', 'orbit': orbit, 'reason': 'no_valid_scenes'}
        
        # Create results dataframe
        df = pd.DataFrame(scene_params)
        df['cats'] = cats
        df['cert'] = cert
        df['orbit'] = orbit
        
        # Create plot
        plot_data = create_plot_data(MCOP, MNOM, df, orbit, region, year, month, day)
        
        # Clean up large objects before returning
        try:
            MCOP.close()
            MNOM.close()
        except:
            pass
        
        del MCOP, MNOM, cats, cert, scene_params
        gc.collect()
        
        return {
            'status': 'success', 
            'orbit': orbit, 
            'df_data': df.to_dict('records'),
            'plot_data': plot_data
        }
        
    except Exception as e:
        return {'status': 'error', 'orbit': orbit, 'error': str(e), 'traceback': traceback.format_exc()}

def create_plot_data(MCOP, MNOM, df, orbit, region, year, month, day):
    """Extract minimal data needed for plotting to reduce memory usage."""
    try:
        plot_data = {
            'orbit': orbit,
            'region': region,
            'year': year,
            'month': month, 
            'day': day,
            'lat_range': (float(np.nanmin(MCOP.latitude.values).round(2)), float(np.nanmax(MCOP.latitude.values).round(2))),
            'lon_range': (float(np.nanmin(MCOP.longitude.values).round(2)), float(np.nanmax(MCOP.longitude.values).round(2))),
            'cot_data': MCOP.cloud_optical_thickness.values,
            'tir2_data': MNOM.TIR2.values,
            'along_track_coords': np.arange(MCOP.dims['along_track']),
            'across_track_coords': np.arange(MCOP.dims['across_track']),
            'df_plot': df[['along_track_mid', 'cats', 'cert', 'ice_flag']].to_dict('records'),
            'across_track_dim': MNOM.dims['across_track']
        }
        return plot_data
    except:
        return None

def create_plot_from_data(plot_data, save_to):
    """Create plot from extracted data."""
    if plot_data is None:
        return
        
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    sns.set_context('talk', font_scale=0.6)
    fig, axs = plt.subplots(2, 1, figsize=(18,4), sharex=True)
    
    # Plot background data using imshow instead of xarray plot
    along_track = plot_data['along_track_coords']
    across_track = plot_data['across_track_coords']
    
    # Plot COT data
    cot_plot = axs[0].imshow(plot_data['cot_data'].T, aspect='auto', 
                            extent=[along_track[0], along_track[-1], 
                                   across_track[0], across_track[-1]],
                            vmin=1, vmax=50, cmap='viridis')
    cbar0 = plt.colorbar(cot_plot, ax=axs[0])
    cbar0.set_label('COT', fontsize=12)
    axs[0].set_ylabel('across track', fontsize=12)
    
    # Plot TIR2 data
    tir2_plot = axs[1].imshow(plot_data['tir2_data'].T, aspect='auto',
                             extent=[along_track[0], along_track[-1], 
                                    across_track[0], across_track[-1]],
                             vmin=273, vmax=300, cmap='viridis')
    cbar1 = plt.colorbar(tir2_plot, ax=axs[1])
    cbar1.set_label('TIR2 (K)', fontsize=12)
    axs[1].set_ylabel('across track', fontsize=12)
    axs[1].set_xlabel('along track', fontsize=12)
    
    # Add scatter plot on top
    df_plot = plot_data['df_plot']
    hue = [6 if row['ice_flag'] > 0.01 else row['cats'] for row in df_plot]
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#e377c2', '#8c564b', 'white']
    palette = {i: colors[i] for i in range(7)}
    
    for i in [0,1]:
        for j, row in enumerate(df_plot):
            color = palette[hue[j]]
            size = row['cert'] * 60
            axs[i].scatter(row['along_track_mid'], plot_data['across_track_dim'] - 128, 
                          c=color, s=size, edgecolors='white', linewidth=0.5)
    
    plt.suptitle(f'Orbit {plot_data["orbit"]}, Region {plot_data["region"]}, Date {plot_data["year"]}-{plot_data["month"]}-{plot_data["day"]}, lat: {plot_data["lat_range"]}, lon: {plot_data["lon_range"]}', 
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(save_to, dpi=100, bbox_inches='tight')
    plt.close('all')
    plt.clf()
    plt.cla()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    logging.info('Starting judge processing script.')
    
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
    
    for region in regions_of_interest:
        logging.info(f"Processing region: {region}")
        
        # Prepare arguments for processing
        orbit_args = []
        for item in orbits_in_stratus_regions[region]:
            frame = item['frame']
            orbitNumber = item['orbitNumber']
            orbit = orbitNumber + frame
            filename = item['filename']
            date = filename.split('ACM_CAP_2B_')[1].split('_')[0][0:8]
            year = date[0:4]
            month = date[4:6]
            day = date[6:8]
            
            if frame in ['D','E']:
                orbit_args.append((region, frame, orbitNumber, orbit, year, month, day, computed_orbits))
        
        # Process orbits one by one with fresh processes
        for i, args in enumerate(orbit_args):
            orbit = args[3]
            logging.info(f'Processing orbit {i+1} of {len(orbit_args)}: {orbit}')
            if orbit in computed_orbits:
                logging.info(f"Orbit {orbit} already computed, skipping.")
                continue
            
            # Use multiprocessing to isolate memory
            with multiprocessing.Pool(1) as pool:
                result = pool.apply(process_single_orbit, [args])
            
            if result['status'] == 'success':
                # Add results to dataframe
                df_new = pd.DataFrame(result['df_data'])
                df_results = pd.concat([df_results, df_new], ignore_index=True)
                
                # Create plot
                if result['plot_data']:
                    create_plot_from_data(result['plot_data'], 
                                        f'{OUTPUT_PLOT}/judge_{orbit}_{region}.png')
                
                logging.info(f"Successfully processed orbit {orbit}")
                
            elif result['status'] == 'error':
                logging.error(f"Error processing orbit {orbit}: {result['error']}")
                failed_orbits.append(orbit)
                with open(FAILED_FILES_LOG, 'a') as f:
                    f.write(f"{orbit}: {result['error']}\n")
            
            # Save intermediate results every 5 orbits
            if (i + 1) % 5 == 0:
                df_results.to_csv(OUTPUT_FILE, index=False)
                logging.info(f"Intermediate results saved to {OUTPUT_FILE}")
                
            # Force garbage collection
            gc.collect()

    # Final save
    df_results.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"Results saved to {OUTPUT_FILE}")
    
    if failed_orbits:
        logging.warning(f"Failed orbits: {failed_orbits}")
    
    logging.info('Judge processing script finished.')

if __name__ == "__main__":
    main()
