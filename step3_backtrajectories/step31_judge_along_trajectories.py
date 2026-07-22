"""
This script uses the trajectories (computed from ERA5 wind data) as input, and runs the judge CNN on GOES data along these trajectories, to classify mesoscale structures.
"""

import xarray as xr
import numpy as np
import pandas as pd
import sys
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import timm
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('../step2_judge_for_goes/')
from functions_for_goes import crop_goes_to_latlon_box, convert_radiance_to_brightness_temp
import judge_goes

OUTPUT_file_partial = '/home/jmayer/ACI_research/Code/resources_mesoscale_structure/bt_with_goes_partial.nc'
OUTPUT_file_final = '/home/jmayer/ACI_research/Code/resources_mesoscale_structure/bt_with_goes.nc'
INPUT_file_trajectories = '/home/jmayer/ACI_research/Code/resources_mesoscale_structure/bt_SEP.nc'

####### TEMPORARY!! overwrite global veriables in step31_judge_along_trajectories.py ########
OUTPUT_file_partial = '/home/jmayer/ACI_research/Code/mesoscale_structure/mesoscale_structure_day_vs_night/bt_with_goes_night_partial.nc'
OUTPUT_file_final = '/home/jmayer/ACI_research/Code/mesoscale_structure/mesoscale_structure_day_vs_night/bt_with_goes_night.nc'
INPUT_file_trajectories = '/home/jmayer/ACI_research/Code/mesoscale_structure/mesoscale_structure_day_vs_night/bt_SEP_night_partial.nc'
################################################################################

judge_for_goes_path = '../step2_judge_for_goes/finetuned_goes_model.pth'

palette_goes_cats = {0: 'purple', 1: 'steelblue', 2: 'green', 3: 'orange', 4: 'red', 5: 'hotpink', np.nan: 'grey'}


def interpolate_trajectories_hourly(bt_SEP):
    """
    Interpolate backtrajectory data from 6-hourly to hourly timesteps.
    
    Parameters:
    -----------
    bt_SEP : xarray.Dataset
        Backtrajectory data with 6-hourly timesteps
    
    Returns:
    --------
    xarray.Dataset
        Interpolated trajectory data with hourly timesteps
    """
    # Create new hourly timestep coordinate
    # Find the range of timesteps
    min_timestep = bt_SEP.timestep.min().values
    max_timestep = bt_SEP.timestep.max().values
    
    # Create hourly timesteps
    hourly_timesteps = np.arange(min_timestep, max_timestep + 1, 2)  # Every 2 hours
    
    # Interpolate each variable
    bt_hourly = bt_SEP.interp(timestep=hourly_timesteps, method='linear')
    
    # For datetime_traj, we need to handle it specially since it's datetime data
    # Convert to numeric (nanoseconds since epoch), interpolate, then convert back
    datetime_numeric = bt_SEP.datetime_traj.astype('datetime64[ns]').astype(np.int64)
    datetime_interp = datetime_numeric.interp(timestep=hourly_timesteps, method='linear')
    bt_hourly['datetime_traj'] = datetime_interp.astype('datetime64[ns]')
    
    return bt_hourly


def set_up_results_dataset(bt_SEP_hourly):
    results = bt_SEP_hourly.copy()
    results['goes_cats'] = (('time_startpoint', 'timestep'), np.full(bt_SEP_hourly['lat_traj'].shape, np.nan))
    results['goes_cert'] = (('time_startpoint', 'timestep'), np.full(bt_SEP_hourly['lat_traj'].shape, np.nan))
    results['goes_ice_flag'] = (('time_startpoint', 'timestep'), np.full(bt_SEP_hourly['lat_traj'].shape, np.nan))
    results['goes_cloud_cover'] = (('time_startpoint', 'timestep'), np.full(bt_SEP_hourly['lat_traj'].shape, np.nan))
    results['goes_cot_gt_1'] = (('time_startpoint', 'timestep'), np.full(bt_SEP_hourly['lat_traj'].shape, np.nan))
    results['goes_cot_gt_5'] = (('time_startpoint', 'timestep'), np.full(bt_SEP_hourly['lat_traj'].shape, np.nan))
    results['goes_cot'] = (('time_startpoint', 'timestep'), np.full(bt_SEP_hourly['lat_traj'].shape, np.nan))
    results['goes_bt14'] = (('time_startpoint', 'timestep'), np.full(bt_SEP_hourly['lat_traj'].shape, np.nan))
    results['goes_y_idx'] = (('time_startpoint', 'timestep'), np.full(bt_SEP_hourly['lat_traj'].shape, np.nan))
    results['goes_x_idx'] = (('time_startpoint', 'timestep'), np.full(bt_SEP_hourly['lat_traj'].shape, np.nan))
    results['goes_lon_mid'] = (('time_startpoint', 'timestep'), np.full(bt_SEP_hourly['lat_traj'].shape, np.nan))
    results['goes_lat_mid'] = (('time_startpoint', 'timestep'), np.full(bt_SEP_hourly['lat_traj'].shape, np.nan))
    results['day_flag'] = (('time_startpoint', 'timestep'), np.full(bt_SEP_hourly['lat_traj'].shape, np.nan))

    # Load existing partial results if available
    if os.path.isfile(OUTPUT_file_partial):
        with xr.open_dataset(OUTPUT_file_partial) as results_existing:
            print("Loading existing results from disk...")
            results_existing.load()

        # Update results where coordinates match with results_existing
        results = results.combine_first(results_existing)
    return results


def load_and_prepare_goes_data(date, time):
    ts = pd.to_datetime(date + time)

    lat_bounds, lon_bounds, times = judge_goes.setup_region_parameters(region='SEP')
    year = date[0:4]
    month = date[4:6]
    if date < '20250407':
        satellite = 'GOES16'
        satellite_small = 'goes16'
    else:
        satellite = 'GOES19'
        satellite_small = 'goes19'
    data_dir=f'/home/jmayer/Object_Data/{satellite}/{year}/{month}/'

    print('.' * 85)
    print(f"Processing date {date} time {time}...")

    if not os.path.exists(f'{data_dir}/{satellite_small}_C11_RadF_{date}_{time}.nc'):
        print(f"GOES C11 data for {date} {time} not found. Skipping.")
        return None, None, None, None
    if not os.path.exists(f'{data_dir}/{satellite_small}_C14_RadF_{date}_{time}.nc'):
        print(f"GOES C14 data for {date} {time} not found. Skipping.")
        return None, None, None, None
    if not os.path.exists(f'{data_dir}/{satellite_small}_C15_RadF_{date}_{time}.nc'):
        print(f"GOES C15 data for {date} {time} not found. Skipping.")
        return None, None, None, None
    if not os.path.exists(f'{data_dir}/{satellite_small}_COD2KMF_{date}_{time}.nc'):
        print(f"GOES COD data for {date} {time} not found. Skipping.")
        return None, None, None, None

    # Load and crop GOES data
    print("Loading C11 data...")
    with xr.open_dataset(f'{data_dir}/{satellite_small}_C11_RadF_{date}_{time}.nc') as ds:
        ds = convert_radiance_to_brightness_temp(ds)
        ds_11_cropped = crop_goes_to_latlon_box(ds, lat_bounds, lon_bounds)

    print("Loading C14 data...")
    with xr.open_dataset(f'{data_dir}/{satellite_small}_C14_RadF_{date}_{time}.nc') as ds:
        ds = convert_radiance_to_brightness_temp(ds)
        ds_14_cropped = crop_goes_to_latlon_box(ds, lat_bounds, lon_bounds)

    print("Loading C15 data...")
    with xr.open_dataset(f'{data_dir}/{satellite_small}_C15_RadF_{date}_{time}.nc') as ds:
        ds = convert_radiance_to_brightness_temp(ds)
        ds_15_cropped = crop_goes_to_latlon_box(ds, lat_bounds, lon_bounds)

    print("Loading COD data...")
    with xr.open_dataset(f'{data_dir}/{satellite_small}_COD2KMF_{date}_{time}.nc') as ds:
        ds_COD_cropped = crop_goes_to_latlon_box(ds, lat_bounds, lon_bounds)

    return ds_11_cropped, ds_14_cropped, ds_15_cropped, ds_COD_cropped


def find_closest_goes_grid_points(goes_latitudes, goes_longitudes, traj_lats, traj_lons):
    """
    Find the closest grid points in the latitude and longitude arrays for each trajectory point.
    """

    # Find closest grid points for each trajectory point using two-step approach
    traj_goes_indices_x = []
    traj_goes_indices_y = []
    for traj_lat, traj_lon in zip(traj_lats, traj_lons):
        # Step 1: Filter coordinates to within 0.1 degree of trajectory point
        lat_diff = np.abs(goes_latitudes - traj_lat)
        lon_diff = np.abs(goes_longitudes - traj_lon)
        within_bounds = (lat_diff <= 0.1) & (lon_diff <= 0.1)
        
        if not np.any(within_bounds):
            print(f"Warning: No grid points within 0.1 degree of trajectory point ({traj_lat:.2f}, {traj_lon:.2f})")
            continue
        
        # Step 2: Calculate distance only for points within bounds
        distance_sq = np.full_like(goes_latitudes, np.inf)  # Initialize with infinity
        distance_sq[within_bounds] = (lat_diff[within_bounds])**2 + (lon_diff[within_bounds])**2
        
        # Find indices of minimum distance
        min_idx = np.unravel_index(np.argmin(distance_sq), distance_sq.shape)
        traj_goes_indices_x.append(min_idx[1])
        traj_goes_indices_y.append(min_idx[0])

    return np.array(traj_goes_indices_x), np.array(traj_goes_indices_y)


def load_model(model_path):
    """Load the fine-tuned model from disk"""
    device_num = '0'  # Change this to your GPU number or 'cpu'
    device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() and device_num != 'cpu' else 'cpu')
    
    model = timm.create_model('resnet50d', pretrained=False)
    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(2048, 6)
    model = model.to(device)
    
    weights = torch.load(model_path, map_location=device)
    model.load_state_dict(weights)
    model.eval()
    
    print(f"Loaded fine-tuned model from {model_path}")
    return model

# apply the fine tuned model to the preprocessed data
class GOESDataset(Dataset):
    def __init__(self, data):
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def run_judge_on_goes_data(all_weirgb, fine_tuned_model):
    """
    Run inference using the fine-tuned model on the preprocessed GOES data.
    
    Parameters:
    -----------
    all_weirgb : np.ndarray
        Preprocessed GOES data of shape (N, C, H, W)
    fine_tuned_model : torch.nn.Module
        The fine-tuned model for inference
    
    Returns:
    --------
    predictions : np.ndarray
        Predicted class labels for each input sample
    probabilities : np.ndarray
        Class probabilities for each input sample
    """

    # Set up device
    device_num = '0'
    device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() and device_num != 'cpu' else 'cpu')

    # Create dataset and dataloader
    dataset = GOESDataset(all_weirgb)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Run inference
    predictions = []
    probabilities = []

    print(f"Running inference on {len(all_weirgb)} samples...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            
            # Forward pass
            outputs = fine_tuned_model(batch)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            # Store results
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            
            # Progress update
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                print(f"Processed {min((batch_idx + 1) * 32, len(all_weirgb))}/{len(all_weirgb)} samples")

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)

    return predictions, probabilities


def summary_plot_results(df_traj_points_at_datetime, COD):
    from matplotlib.colors import ListedColormap
    print("Results summary:")
    print(f"\nClass distribution:")
    for class_id in range(6):
        count = np.sum(df_traj_points_at_datetime['goes_cats'].values == class_id)
        percentage = 100 * count / len(df_traj_points_at_datetime['goes_cats'].values)
        print(f"Class {class_id}: {count} samples ({percentage:.1f}%)")

    # Plot spatial distribution of predictions
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    scale_factor = 2  # 2x upscaling for COD2KMF; 4x upsampling is needed if CODF is used
    COD_interpolated = zoom(COD.values, scale_factor, order=1)
    # Create new xarray DataArray
    COD_interpolated = xr.DataArray(
        COD_interpolated,
        dims=['y', 'x']
    )
    ax.imshow(COD_interpolated, vmax=40, vmin=0, cmap='Greys_r')
    colors = [palette_goes_cats[i] for i in range(6)]
    custom_cmap = ListedColormap(colors, name='goes_classes')
    scatter = ax.scatter(df_traj_points_at_datetime['goes_x_idx'], df_traj_points_at_datetime['goes_y_idx'], 
                        c=df_traj_points_at_datetime['goes_cats'], cmap=custom_cmap, vmin=0, vmax=5, alpha=0.7)
    plt.colorbar(scatter, ax=ax, label='Class')
    plt.show()
    plt.close(fig)

    print(f"\nMean prediction confidence: {df_traj_points_at_datetime['goes_cert'].mean():.3f}")
    print(f"Min prediction confidence: {df_traj_points_at_datetime['goes_cert'].min():.3f}")
    print(f"Max prediction confidence: {df_traj_points_at_datetime['goes_cert'].max():.3f}")


def process_goes_for_trajectory_timestep(date, time, bt_SEP_hourly, fine_tuned_model, plot_results=False):
    # get trajectory points for the current date and time
    traj_points_at_datetime = bt_SEP_hourly.where(bt_SEP_hourly.datetime_traj.dt.strftime('%Y%m%d%H%M') == date + time, drop=True)
    #sns.scatterplot(x=traj_points_at_datetime['lon_traj'].values.flatten(), y=traj_points_at_datetime['lat_traj'].values.flatten())
    # transform to dataframe for easier handling
    df_traj_points_at_datetime = traj_points_at_datetime.to_dataframe().reset_index()
    df_traj_points_at_datetime.dropna(subset=['lat_traj', 'lon_traj'], inplace=True)
    if len(df_traj_points_at_datetime) == 0:
        raise ValueError(f"No trajectory points found for datetime {date} {time}.")

    # Load and prepare GOES data
    ds_11_cropped, ds_14_cropped, ds_15_cropped, ds_COD_cropped = load_and_prepare_goes_data(date, time)
    if ds_11_cropped is None:
        return None
    BT_11, BT_14, BT_15, COD = ds_11_cropped['BT'], ds_14_cropped['BT'], ds_15_cropped['BT'], ds_COD_cropped['COD']

    # get GOES latitude and longitude values (zoomed by the scale factor that is also used for the CNN)
    scale_factor = 2
    latitudes = zoom(BT_11.latitude.values, scale_factor, order=1)
    longitudes = zoom(BT_11.longitude.values, scale_factor, order=1)

    traj_goes_indices_x, traj_goes_indices_y = find_closest_goes_grid_points(latitudes, longitudes, 
                                                                            df_traj_points_at_datetime['lat_traj'].values, 
                                                                            df_traj_points_at_datetime['lon_traj'].values)

    # Preprocess data for model input
    xy_indices = list(zip(traj_goes_indices_x, traj_goes_indices_y))
    all_weirgb, all_scenes_params = judge_goes.preprocess_data(BT_11, BT_14, BT_15, COD, window_size=128, xy_indices=xy_indices, debugging=False)

    # Run inference
    predictions, probabilities = run_judge_on_goes_data(all_weirgb, fine_tuned_model)
    print(f"Inference complete!")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Class distribution: {np.bincount(predictions)}")

    # save judge results to dataframe
    df_traj_points_at_datetime['goes_cats'] = predictions
    df_traj_points_at_datetime['goes_cert'] = np.max(probabilities, axis=1)
    df_traj_points_at_datetime['goes_cats'] = np.where(np.isnan(df_traj_points_at_datetime['goes_cert']), np.nan, df_traj_points_at_datetime['goes_cats'])
    df_traj_points_at_datetime['goes_ice_flag'] = all_scenes_params['ice_flag']
    df_traj_points_at_datetime['goes_cloud_cover'] = all_scenes_params['cloud_cover']
    df_traj_points_at_datetime['goes_cot_gt_1'] = all_scenes_params['cot_gt_1']
    df_traj_points_at_datetime['goes_cot_gt_5'] = all_scenes_params['cot_gt_5']
    df_traj_points_at_datetime['goes_cot'] = all_scenes_params['cot']
    df_traj_points_at_datetime['goes_bt14'] = all_scenes_params['bt14']
    df_traj_points_at_datetime['goes_y_idx'] = all_scenes_params['y_idx']
    df_traj_points_at_datetime['goes_x_idx'] = all_scenes_params['x_idx']
    df_traj_points_at_datetime['goes_lat_mid'] = all_scenes_params['lat_mid']
    df_traj_points_at_datetime['goes_lon_mid'] = all_scenes_params['lon_mid']
    df_traj_points_at_datetime.set_index(['time_startpoint', 'timestep'], inplace=True)
    df_traj_points_at_datetime.drop(columns=['orbit'], inplace=True) # remove orbit, otherwise it gets messy to bring it to a coordinate in the resulting xarray (and not a dimension or variable)

    if plot_results:
        summary_plot_results(df_traj_points_at_datetime, COD)
    
    results_at_datetime = df_traj_points_at_datetime.to_xarray()
    #results_at_datetime = results_at_datetime.assign_coords(orbit=('time_startpoint', results_at_datetime['orbit'].data))
        
    return results_at_datetime


def main():
    with xr.open_dataset(INPUT_file_trajectories) as ds:
        bt_SEP = ds.load()

    # Apply the interpolation
    bt_SEP_hourly = interpolate_trajectories_hourly(bt_SEP)
    print(f"Original timesteps: {len(bt_SEP.timestep)} (every 6h)")
    print(f"Interpolated timesteps: {len(bt_SEP_hourly.timestep)} (every 2h)")
    print(f"Original timestep range: {bt_SEP.timestep.min().values} to {bt_SEP.timestep.max().values}")
    print(f"Hourly timestep range: {bt_SEP_hourly.timestep.min().values} to {bt_SEP_hourly.timestep.max().values}")

    # set up a dataset to store the results; load existing results if available
    results = set_up_results_dataset(bt_SEP_hourly)

    # Load judge CNN model,fine-tuned for GOES
    fine_tuned_model = load_model(judge_for_goes_path)

    # available GOES times
    time_list = ['0000', '0200', '0400', '0600', '0800', '1000', '1200', '1400', '1600', '1800', '2000', '2200']

    for year in ['2024','2025']:
        if year == '2024':
            month_list = ['08','09', '10', '11', '12']
        else:
            month_list = ['01', '02', '03', '04', '05', '06', '07', '08','09', '10', '11']
        for month in month_list:
            if month == '02':
                day_list = [f'{i:02d}' for i in range(1, 29)]
            elif month in ['04', '06', '09', '11']:
                day_list = [f'{i:02d}' for i in range(1, 31)]
            else:
                day_list = [f'{i:02d}' for i in range(1, 32)]

            for day in day_list:
                date=year + month + day
                for time in time_list:
                    # check if GOES data has already been processed for this date and time
                    target_day = np.datetime64(f"{year}-{month}-{day}", "D")
                    target_hour = int(time[:2])
                    goes_cats_at_datetime = results["goes_cats"].where(
                            (results.datetime_traj.dt.floor("D") == target_day) &
                            (results.datetime_traj.dt.hour == target_hour),
                            drop=True
                        )
                    if len(goes_cats_at_datetime) == 0:
                        print(f"Datetime does not exist in trajectory data: {date} {time}. Something went wrong.")
                        break
                    #if not np.isnan(goes_cats_at_datetime).all():
                    if (not goes_cats_at_datetime.isnull().all(dim='time_startpoint').any()) or (time not in ['1400', '1600', '1800', '2000', '2200']): # look into each time_startpoint, if there is at least one non-nan value for goes_cats
                        print(f"GOES data already processed for {date} {time}. Skipping.")
                        continue
                    
                    try:
                        results_at_datetime = process_goes_for_trajectory_timestep(date, time, bt_SEP_hourly, fine_tuned_model, plot_results=False)
                    except ValueError as e:
                        print(e)
                        continue

                    if results_at_datetime is not None:
                        # add a day-night flag
                        day_flag = 1 if time in ['1400', '1600', '1800', '2000', '2200'] else 0
                        results_at_datetime['day_flag'] = (('time_startpoint', 'timestep'), np.full(results_at_datetime['goes_cats'].shape, day_flag))
                        results_at_datetime['day_flag'] = results_at_datetime['day_flag'].where(~results_at_datetime.datetime_traj.isnull())
                        
                        results = xr.merge([results, results_at_datetime])
                        print(f"Updated results dataset for {date} {time}\n")
                
                    results.to_netcdf(OUTPUT_file_partial, 
                            engine='h5netcdf')
            
    results.to_netcdf(OUTPUT_file_final, 
                    engine='h5netcdf')
    
if __name__ == "__main__":
    main()
