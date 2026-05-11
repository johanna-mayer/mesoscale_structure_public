"""
this script computes trajectories using the backtrajectory-calculator package (from ERA5 wind data). 
Starting point for each trajectory are EarthCARE overpass locations, at the midpoints of the judge CNN windows applied to MSI (i.e. always 64km spaced apart along EarthCARE tracks).
-------------------------------------------------------------------------------------------------
Environment: bt_env
"""

import xarray as xr
import pandas as pd
import sys
sys.path.append('/home/jmayer/ACI_research/backtrajectory-calculator/')
from BT import compute_bt

def load_judge_msi_results():
    # load judge results for EarthCARE, to get positions and times of overpasses, specifically the mid-points used for judge
    judge_results = pd.read_csv('/home/jmayer/ACI_research/Code/resources_mesoscale_structure/judge_msi_results_SEP.csv')
    judge_SEP = judge_results[(judge_results['lat_mid'] >= -25) & (judge_results['lat_mid'] <= -2.5) &
                                        (judge_results['lon_mid'] >= -95) & (judge_results['lon_mid'] <= -75)]
    
    # TODO remove filter for time
    judge_SEP = judge_SEP[pd.to_datetime(judge_SEP['time_mid']) >= pd.to_datetime('2025-08-31')]

    return judge_SEP


def bt_dict_to_xr(bt, backwards, orbit, time_i):
    traj_ds = xr.Dataset()
    
    traj_ds['lat_traj'] = ('timestep', bt['lat_traj'][0])
    traj_ds['lon_traj'] = ('timestep', bt['lon_traj'][0])
    traj_ds['plev_traj'] = ('timestep', bt['plev_traj'][0])
    traj_ds['datetime_traj'] = ('timestep', bt['datetime_traj'][0])
    timestep = [i*6 for i in range(len(bt['lat_traj'][0]))]
    if backwards:
        timestep = [-t for t in timestep]
    traj_ds = traj_ds.assign_coords({'timestep': timestep})
    traj_ds = traj_ds.sortby('timestep')
    
    # Expand dimensions to include time_startpoint as a dimension
    traj_ds = traj_ds.expand_dims('time_startpoint')
    traj_ds = traj_ds.assign_coords({'time_startpoint': [pd.to_datetime(time_i)]})
    traj_ds = traj_ds.assign_coords({'orbit': ('time_startpoint', [orbit])})
    return traj_ds

def main():
    bt_SEP = None
    judge_SEP = load_judge_msi_results()
        
    failed_indices = []
    for i in range(len(judge_SEP)):
        #for i in range(3):
        print(i)
        orbit = judge_SEP['orbit'].values[i]
        loni = judge_SEP['lon_mid'].values[i]
        lati = judge_SEP['lat_mid'].values[i]
        datei = pd.to_datetime(judge_SEP['time_mid'].values[i]).date().strftime('%Y-%m-%d')
        timei = pd.to_datetime(judge_SEP['time_mid'].values[i])

        try:
            bt_forward = compute_bt(lati=lati, 
                        loni=loni, 
                        datei=datei, 
                        start_hour=18, # TODO: not exact time of overpass - might this introduce errors in trajectory calculation?
                        ndays=2,
                        backwards=False,
                        overwrite_files=True,)

            bt_backward = compute_bt(lati=lati, 
                        loni=loni, 
                        datei=datei, 
                        start_hour=18, 
                        ndays=2,
                        backwards=True,
                        overwrite_files=True,)

            bt_forward_ds = bt_dict_to_xr(bt_forward, backwards=False, orbit=orbit, time_i=timei)
            bt_backward_ds = bt_dict_to_xr(bt_backward, backwards=True, orbit=orbit, time_i=timei)
            bt = xr.concat([bt_backward_ds, bt_forward_ds], dim='timestep')
            bt = bt.drop_duplicates('timestep')
        except Exception as e:
            print(f'Error computing bt for index {i}, time {timei}, lat {lati}, lon {loni}: {e}')
            failed_indices.append(i)
            continue

        if bt_SEP is None:
            bt_SEP = bt
        else:
            bt_SEP = xr.concat([bt_SEP, bt], dim='time_startpoint')

        bt_SEP.to_netcdf(f'/home/jmayer/ACI_research/Code/resources_mesoscale_structure/bt_SEP_new_partial.nc')
        #if i > 5:
        #    break  # for testing
        
    bt_SEP.to_netcdf('/home/jmayer/ACI_research/Code/resources_mesoscale_structure/bt_SEP_new.nc')
    print(f'Failed indices: {failed_indices}')

if __name__ == "__main__":
    main()
