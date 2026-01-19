import xarray as xr
import pandas as pd
import gcsfs
import numpy as np
import os
import pyproj
from pyproj import Transformer

def download_goes19_simple(start_date, end_date, level='L2', satellite_id='19', product='COD2KMF', 
                          save_dir='/home/jmayer/Object_Data/GOES19/'):
    """
    Simple approach: Download full files first, then process
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fs = gcsfs.GCSFileSystem()
    bucket = f"gcp-public-data-goes-{satellite_id}"
    
    date_range = pd.date_range(start_date, end_date, freq='2h')
    
    for dt in date_range:
        year = dt.year
        day_of_year = dt.timetuple().tm_yday
        hour = dt.hour
        
        path = f"{bucket}/ABI-{level}-{product}/{year}/{day_of_year:03d}/{hour:02d}/"
        
        try:
            files = fs.ls(path)
            
            if files:
                # Download first file of the hour
                remote_file = files[0]
                local_file = os.path.join(save_dir, 
                                        f"goes{satellite_id}_{product}_{dt.strftime('%Y%m%d_%H%M')}.nc")
                
                print(f"Downloading: {remote_file}")
                fs.get(remote_file, local_file)
                print(f"Saved: {local_file}")
                
        except Exception as e:
            print(f"Error downloading {dt}: {e}")



def download_goes19_ir_channels(start_date, end_date, satellite_id='19', channels=['07', '14', '15'],
                                rad_type='RadF', save_dir='/home/jmayer/Object_Data/GOES19/'):
    """
    Download specific GOES-19 IR channels (corrected for actual filename format)
    
    Parameters:
    - channels: List of channel numbers as strings ['07', '14', '15']
    - rad_type: 'RadC' (CONUS), 'RadF' (Full Disk), or 'RadM' (Mesoscale)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fs = gcsfs.GCSFileSystem()
    bucket = f"gcp-public-data-goes-{satellite_id}"
    
    date_range = pd.date_range(start_date, end_date, freq='2H')
    
    for dt in date_range:
        year = dt.year
        day_of_year = dt.timetuple().tm_yday
        hour = dt.hour
        
        print(f"\n=== {dt.strftime('%Y-%m-%d %H:%M')} ===")
        
        # List all files for this hour to find the channels
        path = f"{bucket}/ABI-L1b-{rad_type}/{year}/{day_of_year:03d}/{hour:02d}/"
        
        try:
            files = fs.ls(path)
            
            # Show available channels first
            available_channels = set()
            for f in files:
                filename = f.split('/')[-1]
                if '-M6C' in filename:
                    channel = filename.split('-M6C')[1][:2]
                    available_channels.add(channel)
            
            print(f"  Available channels: {sorted(available_channels)}")
            
            # Filter files for each requested channel
            for channel in channels:
                # Look for pattern like -M6C07_ in filename
                channel_files = [f for f in files if f'-M6C{channel}_' in f]
                
                if channel_files:
                    # Download first file for this channel
                    remote_file = channel_files[0]
                    local_file = os.path.join(save_dir, 
                                            f"goes{satellite_id}_C{channel}_{rad_type}_{dt.strftime('%Y%m%d_%H%M')}.nc")
                    
                    print(f"  Channel {channel}: {remote_file.split('/')[-1]}")
                    fs.get(remote_file, local_file)
                    print(f"  Saved: {os.path.basename(local_file)}")
                else:
                    print(f"  Channel {channel}: No files found")
                    
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    satellite_id = '16'  # GOES-16 or GOES-19
    year = '2024'

    for month in ['08','09','10','11','12']:
        days_of_month = 31 if month in ['01','03','05','07','08','10','12'] else 30
        if month == '02':
            days_of_month = 28  # Not considering leap years for simplicity

        download_goes19_simple(
            start_date=f'{year}-{month}-01 00:00',
            end_date=f'{year}-{month}-{days_of_month} 23:59', 
            satellite_id=satellite_id,
            product='COD2KMF',
            level='L2',
            save_dir=f'/home/jmayer/Object_Data/GOES{satellite_id}/{year}/{month}/'
        )
        # Download IR window channels 
        download_goes19_ir_channels(
            start_date=f'{year}-{month}-01 00:00',
            end_date=f'{year}-{month}-{days_of_month} 23:59', 
            satellite_id=satellite_id,
            channels=['11', '14', '15'],  # IR window channels
            rad_type='RadF',  # Full Disk coverage
            save_dir=f'/home/jmayer/Object_Data/GOES{satellite_id}/{year}/{month}/'
        )

