""" 
This script uses the trajectories (computed from ERA5 data), with already computed judge results along trajectories. 
The goal is to add the average BT14 from GOES per cloud scene along the trajectories.
"""

import xarray as xr
import numpy as np
import pandas as pd

from judge_along_trajectories import *

bt_with_goes_file = '/home/jmayer/ACI_research/Code/resources_mesoscale_structure/bt_with_goes.nc'
OUTPUT_file_partial = '/home/jmayer/ACI_research/Code/resources_mesoscale_structure/bt_with_goes_BT14_partial.nc'

def load_and_prepare_goes_BT14_data(date, time):
    ts = pd.to_datetime(date + time)

    lat_bounds, lon_bounds, times = judge_goes19.setup_region_parameters(region='SEP')
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

    if not os.path.exists(f'{data_dir}/{satellite_small}_C14_RadF_{date}_{time}.nc'):
        print(f"GOES C14 data for {date} {time} not found. Skipping.")
        return None

    print("Loading C14 data...")
    with xr.open_dataset(f'{data_dir}/{satellite_small}_C14_RadF_{date}_{time}.nc') as ds:
        ds = convert_radiance_to_brightness_temp(ds)
        ds_14_cropped = crop_goes_to_latlon_box(ds, lat_bounds, lon_bounds)

    return ds_14_cropped


def process_goes_for_trajectory_timestep(date, time, bt_with_goes):
    ds_14_cropped= load_and_prepare_goes_BT14_data(date, time)
    if ds_14_cropped is None:
        return None
    BT_14 = ds_14_cropped['BT']


    # get trajectory points for the current date and time
    traj_points_at_datetime = bt_with_goes.where(bt_with_goes.datetime_traj.dt.strftime('%Y%m%d%H%M') == date + time, drop=True)
    #sns.scatterplot(x=traj_points_at_datetime['lon_traj'].values.flatten(), y=traj_points_at_datetime['lat_traj'].values.flatten())
    # transform to dataframe for easier handling
    df_traj_points_at_datetime = traj_points_at_datetime.to_dataframe().reset_index()
    df_traj_points_at_datetime.dropna(subset=['lat_traj', 'lon_traj'], inplace=True)

    # get GOES latitude and longitude values
    latitudes = BT_14.latitude.values
    longitudes = BT_14.longitude.values

    traj_goes_indices_x, traj_goes_indices_y = find_closest_goes_grid_points(latitudes, longitudes, 
                                                                            df_traj_points_at_datetime['lat_traj'].values, 
                                                                            df_traj_points_at_datetime['lon_traj'].values)

    # use judge preprocessing function to get BT14 cloud scene windows
    xy_indices = list(zip(traj_goes_indices_x, traj_goes_indices_y))
    num_scenes = len(xy_indices)
    window_size = 64 # size of the window to extract around each trajectory point (64 instead of 128 because GOES has a resolution of 2km)
    goes_scenes_params = {'lon_mid_2': [], 'lat_mid_2': [], 'BT14': [], 'ice_flag_2': []}

    for i,j in xy_indices:
        lat_mid = latitudes[j, i]
        lon_mid = longitudes[j, i]

        # get window data
        data22 = BT_14[j - window_size // 2 : j + window_size // 2,
                        i - window_size // 2 : i + window_size // 2].values
        
        ice_pixel = (data22 < 273)
        ice_flag = np.sum(ice_pixel) / (window_size * window_size)
        if ice_flag < 0.1:
            BT_14_mean = np.nanmean(data22)
        else:
            BT_14_mean = np.nan

        goes_scenes_params['lon_mid_2'].append(lon_mid)
        goes_scenes_params['lat_mid_2'].append(lat_mid)
        goes_scenes_params['BT14'].append(BT_14_mean)
        goes_scenes_params['ice_flag_2'].append(ice_flag)


    # save judge results to dataframe
    df_traj_points_at_datetime['goes_BT14'] = goes_scenes_params['BT14']
    df_traj_points_at_datetime['goes_ice_flag_2'] = goes_scenes_params['ice_flag_2']
    df_traj_points_at_datetime['goes_lat_mid_2'] = goes_scenes_params['lat_mid_2']
    df_traj_points_at_datetime['goes_lon_mid_2'] = goes_scenes_params['lon_mid_2']
    df_traj_points_at_datetime.set_index(['time_startpoint', 'timestep'], inplace=True)
    df_traj_points_at_datetime.drop(columns=['orbit'], inplace=True) # remove orbit, otherwise it gets messy to bring it to a coordinate in the resulting xarray (and not a dimension or variable)

    results_at_datetime = df_traj_points_at_datetime.to_xarray()
    return results_at_datetime


if __name__ == "__main__":
    if os.path.exists(OUTPUT_file_partial):
        print(f"Output file {OUTPUT_file_partial} already exists. Loading existing data.")
        with xr.open_dataset(OUTPUT_file_partial) as ds:
            bt_with_goes = ds.load()
    else:
        with xr.open_dataset(bt_with_goes_file) as bt_with_goes:
            bt_with_goes = bt_with_goes.load()

    # available GOES times
    time_list = ['0000', '0200', '0400', '0600', '0800', '1000', '1200', '1400', '1600', '1800', '2000', '2200']

    for year in ['2024','2025']:
        if year == '2024':
            month_list = ['08', '09', '10', '11', '12']
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
                    goes_BT14_at_datetime = bt_with_goes.where(bt_with_goes.datetime_traj.dt.strftime('%Y%m%d%H%M') == date + time, drop=True).goes_BT14
                    if len(goes_BT14_at_datetime) == 0:
                        print(f"Datetime does not exist in trajectory data: {date} {time}. Something went wrong.")
                        break
                    if ~np.isnan(goes_BT14_at_datetime).all():
                        print(f"GOES data already processed for {date} {time}. Skipping.")
                        continue


                    results_at_datetime = process_goes_for_trajectory_timestep(date, time, bt_with_goes)

                    if results_at_datetime is not None:
                        bt_with_goes = xr.merge([bt_with_goes, results_at_datetime])
                        print(f"Updated results dataset for {date} {time}\n")

                
                bt_with_goes.to_netcdf(OUTPUT_file_partial, 
                        engine='h5netcdf')
    
