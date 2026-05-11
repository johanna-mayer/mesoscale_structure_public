import xarray as xr
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.ndimage import zoom

def convert_radiance_to_brightness_temp(ds):
    """
    Convert GOES ABI radiance to brightness temperature
    Uses the planck function parameters in the file attributes
    """
    # Get calibration parameters from file attributes
    fk1 = ds['planck_fk1']
    fk2 = ds['planck_fk2'] 
    bc1 = ds['planck_bc1']
    bc2 = ds['planck_bc2']
    
    if all(param is not None for param in [fk1, fk2, bc1, bc2]):
        # Convert radiance to brightness temperature
        bt = (fk2 / (np.log((fk1 / ds.Rad) + 1)) - bc1) / bc2
        
        # Add as new variable
        ds['BT'] = (ds.Rad.dims, bt.values)
        ds['BT'].attrs = {
            'long_name': 'Brightness Temperature',
            'units': 'K',
            'standard_name': 'brightness_temperature'
        }
        
    return ds


def add_latlon_to_goes(ds):
    """
    Simpler approach using pyproj with proper GOES coordinate handling
    """
    # Get satellite projection info
    if 'goes_imager_projection' in ds.variables:
        proj_info = ds['goes_imager_projection']
        sat_height = proj_info.attrs.get('perspective_point_height', 35786023.0)
        sat_lon = proj_info.attrs.get('longitude_of_projection_origin', -137.0)
        sat_sweep = proj_info.attrs.get('sweep_angle_axis', 'x')
    else:
        sat_height = 35786023.0
        sat_lon = -137.0
        sat_sweep = 'x'
    
    # GOES coordinates are in radians, need to convert to meters for pyproj
    # The scale factor is the satellite height
    x_rad = ds.x.values
    y_rad = ds.y.values
    
    # Convert radians to meters
    x_m = x_rad * sat_height
    y_m = y_rad * sat_height
    
    # Create the projection
    proj_string = f"+proj=geos +h={sat_height} +lon_0={sat_lon} +sweep={sat_sweep} +ellps=GRS80 +units=m +no_defs"
    transformer = Transformer.from_crs(proj_string, "EPSG:4326", always_xy=True)
    
    # Create meshgrid
    X_m, Y_m = np.meshgrid(x_m, y_m)
    
    # Transform to lat/lon
    lon, lat = transformer.transform(X_m, Y_m)
    
    # Add as coordinates
    ds = ds.assign_coords({
        'longitude': (('y', 'x'), lon),
        'latitude': (('y', 'x'), lat)
    })
    
    return ds

def crop_goes_to_latlon_box(ds, lat_bounds, lon_bounds):
    """
    Crop GOES data to a lat/lon bounding box
    
    Parameters:
    - ds: xarray Dataset with GOES data
    - lat_bounds: tuple (lat_min, lat_max)
    - lon_bounds: tuple (lon_min, lon_max)
    """
    
    # Add lat/lon coordinates if not present
    if 'latitude' not in ds.coords or 'longitude' not in ds.coords:
        ds = add_latlon_to_goes(ds)
    
    # Get bounds
    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds
    
    # Create mask for the region
    lat_mask = (ds.latitude >= lat_min) & (ds.latitude <= lat_max)
    lon_mask = (ds.longitude >= lon_min) & (ds.longitude <= lon_max)
    combined_mask = lat_mask & lon_mask
    
    # Find indices where mask is True
    y_indices, x_indices = np.where(combined_mask)
    
    if len(y_indices) == 0:
        print("No data found in specified region")
        return None
    
    # Get bounding box in array coordinates
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    
    # Add some padding if desired
    padding = 10  # pixels
    y_min = max(0, y_min - padding)
    y_max = min(ds.dims['y'] - 1, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(ds.dims['x'] - 1, x_max + padding)
    
    # Crop the dataset
    ds_cropped = ds.isel(y=slice(y_min, y_max+1), x=slice(x_min, x_max+1))
    
    print(f"Original shape: {ds.dims}")
    print(f"Cropped shape: {ds_cropped.dims}")
    print(f"Lat range: {ds_cropped.latitude.min().values:.2f} to {ds_cropped.latitude.max().values:.2f}")
    print(f"Lon range: {ds_cropped.longitude.min().values:.2f} to {ds_cropped.longitude.max().values:.2f}")
    
    return ds_cropped
