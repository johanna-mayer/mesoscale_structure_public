import glob
import sys
import json
import os
from typing import Tuple, List, Dict, Optional, Union

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import label
import cv2
import scipy.stats

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir + '/../../')
sys.path.append(current_dir + '/../../../ectools/')
sys.path.append('/home/jmayer/ACI_research/Code/mesoscale_structure/')
import access_ec_on_maap as maap

from ectools import ecio
from ectools import ecplot as ecplt

regions_of_interest = ['NEA'] #, 'NEA', 'SEP', 'NEP'
# Constants and Configuration
OUTPUT_FILE = f'/home/jmayer/ACI_research/Code/resources_mesoscale_structure/ACMCAP_StCu_{regions_of_interest[0]}.h5'
OUTPUT_PLOT = current_dir + '/../figures/'
#INPUT_FILE_files_in_sc_regions = '/home/jmayer/ACI_research/Code/resources/files_in_sc_regions.json'
INPUT_FILE_orbits_in_sc_regions = '/home/jmayer/ACI_research/Code/resources_mesoscale_structure/acmcap_orbits_in_sc_regions.json'

# Configuration flags
add_msi_vars = False
add_xmet_vars = False
add_xmet_cloud_vars = False
add_lwc_profiles = False
plot_frames = True

# Data directories
ACMCAP_data_dir = "/home/jmayer/Object_Data/ACM_CAP_2B/"
XMET_data_dir = "/home/jmayer/Object_Data/AUX_MET_1D/"
MCOP_data_dir = "/home/jmayer/Object_Data/MSI_COP_2A/"

# Stratus regions - expanded from Klein and Hartmann (1993)
stratus_regions = {
    'SEA': {'lat': (-25, -2.5), 'lon': (-5, 15)},    # South East Atlantic
    'NEA': {'lat': (10, 30), 'lon': (-40, -20)},   # North East Atlantic
    'SEP': {'lat': (-25, -2.5), 'lon': (-95, -75)},  # South East Pacific
    'NEP': {'lat': (15, 35), 'lon': (-135, -115)}, # North East Pacific
}

stratus_region_index = {'SEA': 0, 'NEA': 1, 'SEP': 2, 'NEP': 3}


# ======================== ACMCAP Data Processing ========================
vertical_pixelsize = 102.89 # (approximate) vertical pixelsize in troposphere

def compute_lwp(ds):
    lwp = ds.liquid_water_content.where(ds.liquid_classification != 0 
                                            ).sum(dim="JSG_height") * vertical_pixelsize # multiply with vertical pixelsize in m to get the right units

    lwp.attrs['long_name'] = 'Liquid water path'
    lwp.attrs['units'] = r'kg/m$^2$'
    lwp.attrs['plot_range'] = np.array([1.e-4, 1.])
    lwp.attrs['plot_scale'] = 'logarithmic'
    lwp.name = 'liquid_water_path'
    
    return lwp

def compute_Nd(ds):
    Nd = ds.liquid_number_concentration.where(ds.liquid_classification != 0
                                                      ).mean(dim="JSG_height")
    Nd.attrs['long_name'] = 'Mean liquid number concentration'
    Nd.attrs['units'] = r'm$^{-3}$'
    Nd.attrs['plot_range'] = np.array([1.e+5, 1.e+9])
    Nd.attrs['plot_scale'] = 'logarithmic'
    Nd.name = 'liquid_number_concentration'
    return Nd

def compute_ER(ds):
    ER = ds.liquid_effective_radius.where(ds.liquid_classification != 0
                                                      ).mean(dim="JSG_height")
    ER.attrs['long_name'] = 'Mean liquid effective radius'
    ER.attrs['units'] = 'm'
    ER.attrs['plot_range'] = np.array([0.,0.0002])
    ER.attrs['plot_scale'] = 'linear'
    ER.name = 'liquid_effective_radius'
    return ER

def compute_ER_passive_perspective(ds):
    density_liquid = 1000 # kg/m3
    LWP = compute_lwp(ds) # kg/m2
    ER = (3/2) * LWP / (density_liquid * ds.liquid_optical_depth)
    ER.attrs['long_name'] = 'Liquid effective radius computed as (3/2) * LWP / (density_liquid * optical_depth)'
    ER.attrs['units'] = 'm'
    ER.attrs['plot_range'] = np.array([0.,0.0002])
    ER.attrs['plot_scale'] = 'linear'
    ER.name = 'liquid_effective_radius_passive_perspective'
    return ER


def get_cloud_mask(ds):
        # create a cloud mask
        ds_liq_classification_nan_mask = (ds.liquid_classification == -127) | np.isnan(ds.liquid_classification) # -127 is the fill value
        ds_cld = xr.where((ds.liquid_classification.fillna(0) != 0) | (ds.ice_water_content > 0), 1, 0)
        ds_cld = ds_cld.where(~ds_liq_classification_nan_mask, np.nan)
        ds_cld.name = 'cloud_mask'

        # add along_track and JSG_height as coordinates if they are not already
        if 'along_track' not in ds_cld.coords or 'JSG_height' not in ds_cld.coords:
            ds_cld = ds_cld.assign_coords(
                {
                'along_track': (
                    ["along_track"],
                    ds_cld['along_track'].data,
                    ds_cld['along_track'].attrs
                    ),
                'JSG_height': (
                    ['JSG_height'],
                    ds_cld['JSG_height'].data,
                    ds_cld['JSG_height'].attrs
                    )
                })
        
        return ds_cld

def get_low_cloud_mask(ds_cld, 
                       filter_by='ctt', 
                       ds_temperature=None, 
                       ds_height=None,
                       ds_boundary_layer_height=None,
                       filter_by_avg_cloud_object_values=False
                       ):
    """ 
    Create a mask for low clouds by labeling connected components in the cloud mask and filtering out clouds with maximum CTH below 200 JSG_height units.
    INPUTS:
        ds_cld (xr.DataArray): cloud mask with values 0 and 1. The dataset must have the coordinates "along_track" and "JSG_height".
        ds_temperature (xr.DataArray): temperature dataset with the same coordinates as ds_cld. Required if filter_by is "ctt".
        filter_by (str): filter low clouds by cloud top temperature ("ctt") or cloud top height ("cth").
        filter_by_avg_cloud_object_values (bool): if True, the average CTT of a cloud object is used for filtering. Otherwise, the CTT is evaluated pixelwise.
    RETURNS:
        low_cloud_mask (xr.DataArray): mask for low clouds with values True and False.
    """
    from scipy.ndimage import label

    if filter_by == 'ctt':
        if ds_temperature is None:
            raise ValueError('ds_temperature must be provided if filter_by is "ctt"')
        threshold = 273
        # make sure that ds_temperature has the right coordinates
        if 'along_track' not in ds_temperature.coords or 'JSG_height' not in ds_temperature.coords:
            ds_temperature = ds_temperature.assign_coords(
                {
                'along_track': (
                    ["along_track"],
                    ds_temperature['along_track'].data,
                    ds_temperature['along_track'].attrs
                    ),
                'JSG_height': (
                    ['JSG_height'],
                    ds_temperature['JSG_height'].data,
                    ds_temperature['JSG_height'].attrs
                    )
        })
            
    elif filter_by == 'cth':
        if ds_height is None:
            raise ValueError('ds_height must be provided if filter_by is "cth"')
        if ds_boundary_layer_height is None:
            raise ValueError('ds_boundary_layer_height must be provided if filter_by is "cth"')


    ds_cld_nan_mask = np.isnan(ds_cld)
    # interpolate to fill single NaN columns
    ds_cld_interpolated = ds_cld.interpolate_na(dim='along_track', method='linear', limit=1)
    # round everything below 1 to 0
    ds_cld_interpolated = np.floor(ds_cld_interpolated)

    # Label connected components
    structure = np.ones((3, 3), dtype=int)
    labeled_array, num_features = label(ds_cld_interpolated.values == 1, structure=structure)

    # Convert labeled_array back to xarray DataArray
    labeled_ds_cld = xr.DataArray(labeled_array, dims=ds_cld.dims, coords=ds_cld.coords)

    #print(f'Number of cloud objects: {num_features}')
    #labeled_ds_cld.T.plot(figsize=(30, 5))

    labeled_ds_cld_filtered = labeled_ds_cld.copy()
    for i in range(1,num_features+1): #TODO make this more efficient?
        cloud_object = labeled_ds_cld.where(labeled_ds_cld == i, drop=True)
        cth_cloud_object = cloud_object['JSG_height'].where(~np.isnan(cloud_object)).groupby('along_track').min(dim='JSG_height')

        # Align the dimensions of cth_cloud_object with labeled_ds_cld_filtered
        cth_cloud_object_aligned = cth_cloud_object.reindex_like(labeled_ds_cld_filtered, method=None)

        if filter_by == 'cth':
            cth_wiggle_room = 0 # in meters

            #print('WARNING: Temporary filtering method: Filtering out clouds with maximum CTH below 200 JSG_height units')
            # set labeled_ds_cld to 0 where cth_cloud_object is too low (too high altitude)
            #labeled_ds_cld_filtered = xr.where((labeled_ds_cld == i) & (cth_cloud_object_aligned < threshold), 0, labeled_ds_cld_filtered)

            cth_nan_mask = np.isnan(cth_cloud_object_aligned)
            cth_cloud_object_aligned = cth_cloud_object_aligned.fillna(0).astype(int) # fill NaNs in the cth s.t. the CTH in meters can be retrieved
            cth_m = ds_height.sel(JSG_height = cth_cloud_object_aligned) # CTH in meters

            if filter_by_avg_cloud_object_values:
                cth_minus_blh = (cth_m - (ds_boundary_layer_height+cth_wiggle_room)).where(~cth_nan_mask).mean(dim='along_track')
                if cth_minus_blh > 0: # set labeled_ds_cld to 0 where mean CTH is too high (above the boundary layer + wiggle room)
                    labeled_ds_cld_filtered = xr.where((labeled_ds_cld == i), 0, labeled_ds_cld_filtered)
            else:
                # set labeled_ds_cld to 0 where CTH is too high (above the boundary layer + wiggle room)
                labeled_ds_cld_filtered = xr.where((labeled_ds_cld == i) & (cth_m > ds_boundary_layer_height+cth_wiggle_room), 0, labeled_ds_cld_filtered)

        if filter_by == 'ctt':
            # get the CTT of the cloud object
            cth_nan_mask = np.isnan(cth_cloud_object_aligned)
            cth_cloud_object_aligned = cth_cloud_object_aligned.fillna(250) # fill NaNs in the cth s.t. the CTT can be retrieved
            ctt = ds_temperature.sel(JSG_height = cth_cloud_object_aligned)

            if filter_by_avg_cloud_object_values:
                ctt_mean = ctt.where(~cth_nan_mask).mean(dim='along_track')
                if ctt_mean < threshold:
                    labeled_ds_cld_filtered = xr.where((labeled_ds_cld == i), 0, labeled_ds_cld_filtered)
            else:
                # set labeled_ds_cld to 0 where CTT is too low (below 273 K)
                labeled_ds_cld_filtered = xr.where((labeled_ds_cld == i) & (ctt < threshold), 0, labeled_ds_cld_filtered)

        #fig, ax = plt.subplots(1,2,figsize=(15,4))
        #cloud_object.T.plot(ax=ax[0])
        #cth_cloud_object_aligned.plot(ax=ax[0])
        #labeled_ds_cld_filtered.T.plot(ax=ax[1])

        # ----------- Deprecated approach -------------
        # delete whole cloud object if maximum CTH is below 200 JSG_height units
        #cloud_object = labeled_ds_cld.where(labeled_ds_cld == i, drop=True)
        ## get minimum cth of cloud_object
        #max_cth_of_cloud_object = cloud_object.coords['JSG_height'].min()
    #
        #if max_cth_of_cloud_object < 200: # TODO: need a better method depending on temperature
        #    # set labeled_ds_cld to 0 where cloud_object is
        #    labeled_ds_cld_filtered = labeled_ds_cld_filtered.where(labeled_ds_cld_filtered != i, 0)
        # ----------- Deprecated approach -------------

    low_cloud_mask = (labeled_ds_cld_filtered != 0)
    low_cloud_mask = low_cloud_mask.where(~ds_cld_nan_mask, np.nan)

    # add attributes
    low_cloud_mask.name = 'low_cloud_mask'
    low_cloud_mask.attrs['long_name'] = 'Low cloud mask'
    low_cloud_mask.attrs['units'] = '1'
    low_cloud_mask.attrs['plot_range'] = [0,1]
    low_cloud_mask.attrs['plot_scale'] = 'linear'

    return low_cloud_mask

def get_cth_and_cbh(ds_cld):
        """ 
        retrieves the overall CTH and CBH from a cloud mask ds_cld.
        INPUTS:
            ds_cld (xr.DataArray): cloud mask with values 0 and 1. ds_cld must have the coordinates "along_track" and "JSG_height". It can be computed using the get_cloud_mask funtion.

        RETURNS:
                ds_cloudtop (xr.Dataset): dataset containing the two variables cth and cbh
        """
        ds_cld = ds_cld.where(ds_cld == 1)

        # get the maximum and minimum JSG_height of each along_track step (by first converting to a pandas dataframe and grouping by "along_track" values)
        df_cth = ds_cld.to_dataframe().dropna().reset_index().groupby(["along_track"])["JSG_height"].min()
        df_cbh = ds_cld.to_dataframe().dropna().reset_index().groupby(["along_track"])["JSG_height"].max()

        # create DataArrays for CTH, CBH
        cth_arr = xr.DataArray(
        data=df_cth,
        dims=["along_track"],
        coords={'along_track': (["along_track"],
                                df_cth.index,
                                ds_cld['along_track'].attrs)
                },
        name="cth",
        attrs={'long_name': 'cloud top JSG_height',
                'plot_title': 'cloud top JSG_height'}
        )
        cbh_arr = xr.DataArray(
        data=df_cbh,
        dims=["along_track"],
        coords={'along_track': (["along_track"],
                                df_cth.index,
                                ds_cld['along_track'].attrs)
                },
        name="cbh",
        attrs={'long_name': 'cloud bottom JSG_height',
                'plot_title': 'cloud bottom JSG_height'}
        )

        # the DataArrays cth_arr and cbh_arr do not have values for along_tracks where there are no clouds.
        # We therefore need to merge (and left-join) with ds_cld, which has the correct along_track steps.
        ds_tmp = xr.merge([ds_cld, cth_arr, cbh_arr], join='left', fill_value=np.nan)

        ds_cloudtop = ds_tmp[['cth','cbh']].drop_attrs(deep=False)

        return ds_cloudtop

def get_cth_second_cloud_layer(ds_cld, cth, cbh, minimum_non_cloudy_distance=10):
    """ 
    retrieves the CTH of second cloud layer (as seen from above) from a cloud mask ds_cld.
    the dataset ds_cld must have the coordinates "along_track" and "JSG_height".

    RETURNS:
        da_cth_second_layer (xr.DataArray): DataArray containing the CTH_sedcond_layer data
    """
    # select pixels between CBH and CTH. Only differentiate between cloud (1) and no cloud (0). Save the results in a pd.dataframe
    da = ds_cld.where((ds_cld.JSG_height <= cbh) & (ds_cld.JSG_height >= cth))
    da.name = 'cloud_mask'
    df = da.to_dataframe().dropna()

    # For each timestep, find all pixels that have the same value (cloud/ no cloud) consecutively for at least 2 km vertically
    df['id_consecutive_same_values'] = df.cloud_mask.diff().ne(0).cumsum()

    # For each timestep find the maximal cloud gap (there are often multiple layers)
    df = df[df.cloud_mask == 0]
    df['size_consecutive_same_values'] = df.cloud_mask.groupby(
        ['along_track', df['id_consecutive_same_values']]).transform('size')

    # we only consider a scene "multilayered" if the cloud-free gap between layers is at least 2 km large
    df_multilayer = df[df['size_consecutive_same_values'] >= minimum_non_cloudy_distance]
    df_multilayer = df_multilayer.reset_index().set_index(['along_track', 'id_consecutive_same_values'])

    # compute the mean height of the cloud-free gap
    df_multilayer['mean_gap_height'] = df_multilayer.groupby(['along_track', 'id_consecutive_same_values'])[
        'JSG_height'].transform('mean')
    # choose the cloud-free gap with the highest altitude
    df_multilayer['highest_gap'] = df_multilayer.groupby("along_track")["mean_gap_height"].transform('max')

    # The CTH of the second cloud layer is the maximum JSG_height value of the cloud-free pixels in the highest cloud-free gap (plus 1 JSG_height unit)
    d = df_multilayer.dropna()[['JSG_height']]
    cth_second_layer = d.JSG_height.groupby(['along_track', 'id_consecutive_same_values']).max().groupby('along_track').min() + 1
    cth_second_layer = cth_second_layer.rename('cth_second_layer')

    # append the new variable cth_second_layer to the dataset
    da_cth_second_layer = cth_second_layer.to_xarray()
    da_cth_second_layer.attrs = {'name': 'CTH second layer',
                                    'long_name': 'CTH of the second cloud layer seen from TOA',
                                    'comments': 'NaN if there is no second cloud layer;\na scene is considered "multilayered" if the cloud-free gap between layers is at least 5 JSG_height units large',
                                    'units': 'JSG_height units'}
    
    return da_cth_second_layer


def prepare_ACMCAP_data(ds: xr.Dataset, filter_by_quality: bool = True, xmet_status: int = -1) -> xr.Dataset:
    """
    Prepare ACMCAP data by filtering quality, computing cloud properties, and deriving variables.
    
    Args:
        ds: Raw ACMCAP dataset
        filter_by_quality: Whether to filter data by quality status
        
    Returns:
        Processed ACMCAP dataset with computed cloud properties
    """
    if filter_by_quality:
        print('Filtering data by quality: only consider data with quality_status <= 1')
        ds = ds.where(ds.quality_status <= 1, drop=True)

    # Add coordinates if not already present
    if 'along_track' not in ds.coords or 'JSG_height' not in ds.coords:
        ds = ds.assign_coords({
            'along_track': (["along_track"], ds['along_track'].data, ds['along_track'].attrs),
            'JSG_height': (['JSG_height'], ds['JSG_height'].data, ds['JSG_height'].attrs)
        })
            
    # Compute cloud masks and properties
    ds_cld = get_cloud_mask(ds)
    tmp_cth_cbh = get_cth_and_cbh(ds_cld)
    ds['cth_all_clouds'] = tmp_cth_cbh['cth']
    ds['low_cloud_mask'] = get_low_cloud_mask(ds_cld, filter_by='cth', ds_boundary_layer_height=ds['latitude']*0 + 3000, ds_height=ds['height'])

    tmp_cth_cbh = get_cth_and_cbh(ds['low_cloud_mask'])
    ds['cth'] = tmp_cth_cbh['cth']
    ds['cbh'] = tmp_cth_cbh['cbh']
    ds['cth_m'] = ds.height.sel(JSG_height = ds.cth.fillna(0)).where(ds.cth.notnull())
    ds['cbh_m'] = ds.height.sel(JSG_height = ds.cbh.fillna(0)).where(ds.cbh.notnull())

    # Multi-layer detection
    cth_second_layer = get_cth_second_cloud_layer(
        ds.low_cloud_mask, ds.cth, ds.cbh, minimum_non_cloudy_distance=2
    )
    ds = xr.merge([ds, cth_second_layer], compat='override')
    ds['multilayer'] = xr.where(np.isnan(ds.cth_second_layer), 0, 1)

    ds['N_cloudy_pixels'] = ds.low_cloud_mask.sum(dim='JSG_height')
    ds['N_cloudy_pixels'] = ds.N_cloudy_pixels.where(ds.N_cloudy_pixels > 0)  # set to NaN if there are no cloudy pixels 

    # vertical distribution of cloud fraction
    # computed following the third criterion of Cesana et al. (2019) for the CASCCAD algorithm
    
    if len(ds['low_cloud_mask'].along_track) >= 80:
        cc_rolling_per_height = ds['low_cloud_mask'].groupby('JSG_height').apply(lambda x: x.rolling(along_track=80, center=True, min_periods=1).mean())
        ds['N_high_cc'] = (cc_rolling_per_height > 0.3).sum(dim='JSG_height')
    else:
        print(f'WARNING: Skipping computation of N_high_cc, since Number of along_track pixels is {len(ds["low_cloud_mask"].along_track)}, which is less than 80.')
        ds['N_high_cc'] = ds.latitude*0 + (-1) # set to -1 if error occurs

    # auxiliary variables: surface height and 1d cloud mask
    #ds['surface_height_JSG'] = np.abs(ds.height).idxmin(dim='JSG_height') # not sure if the minimum height is really at the surface or below?
    if add_xmet_vars and xmet_status == 0:
        ds['surface_height_JSG'] = ds.pressure.idxmax(dim='JSG_height')
    else:
        print('ATTENTION: No XMET data added, assuming surface height at 235 (approx. 1000 hPa)')
        ds['surface_height_JSG'] = 235
    ds['cloud_mask_1d'] = ds.low_cloud_mask.max(dim='JSG_height')

    # Compute microphysical properties
    ds_low_clouds = ds[['liquid_water_content', 'liquid_number_concentration', 'liquid_effective_radius', 
                        'liquid_water_content_error', 'liquid_number_concentration_error', 'liquid_classification']].where(ds.low_cloud_mask)
    ds_low_clouds['liquid_optical_depth'] = ds['liquid_optical_depth'] # needed for the computation of ER_passive_perspective
    ds['lwp'] = compute_lwp(ds_low_clouds)
    ds['Nd'] = compute_Nd(ds_low_clouds)
    ds['ER'] = compute_ER(ds_low_clouds)
    ds['ER_passive_perspective'] = compute_ER_passive_perspective(ds_low_clouds)
    ds['lwp_error'] = ds_low_clouds.liquid_water_content_error.mean(dim='JSG_height')
    ds['Nd_error'] = ds_low_clouds.liquid_number_concentration_error.mean(dim='JSG_height')
    ds['liquid_classification_mean'] = ds_low_clouds.liquid_classification.where(ds_low_clouds.liquid_classification != 0).mean(dim='JSG_height')

    return ds


# ======================== XMET Data Processing ========================

def find_XMET_file_matching_ACMCAP_filename(ACMCAP_filename: str, dir: str) -> Tuple[str, int]:
    """Find XMET files that match the ACMCAP filename."""
    import re 

    ACMCAP_filename = ACMCAP_filename.split('/')[-1]
    production_datetime = ACMCAP_filename.split('_')[-2]
    start_datetime = ACMCAP_filename.split('_')[-3]
    tmp = ACMCAP_filename.replace('ACM_CAP_2B', 'AUX_MET_1D').replace(production_datetime, '*').replace(start_datetime, start_datetime[0:9]+'*')
    tmp = re.sub(r'ECA_EX[A-Z]{2}', 'ECA_EX*', tmp)
    
    filenames_XMET = glob.glob(f'{dir}/{tmp}')
    
    if len(filenames_XMET) == 0:
        return 'no_XMET_file_found', -1
    elif len(filenames_XMET) > 1:
        return 'more_than_1_XMET_file_found', -1
    else:
        return filenames_XMET[0], 0


def add_XMET_temperature_to_ACMCAP(ACMCAP: xr.Dataset, ACMCAP_filename: str, 
                                   XMET_2D_variables: List[str], XMET_1D_variables: List[str]) -> Tuple[xr.Dataset, int]:
    """Add XMET temperature data to ACMCAP dataset."""
    filename_XMET, status = find_XMET_file_matching_ACMCAP_filename(ACMCAP_filename, f'{XMET_data_dir}/*/*/*/')

    if status == 0:
        for attempt in range(3):
            try:
                XMET = ecio.load_XMET(filename_XMET)
                ACMCAP = ecio.get_XMET(XMET, ACMCAP, XMET_2D_variables=XMET_2D_variables, XMET_1D_variables=XMET_1D_variables)
                break
            except Exception as e:
                if attempt == 2:
                    print(f'Error loading XMET data after 3 attempts: {e}')
                    status = -1
                    return ACMCAP, status
                else:
                    print(f'Error loading XMET data (attempt 1 or 2), retrying: {e}')
    else:
        print(f'No XMET data found for ACMCAP filename {ACMCAP_filename}')
    
    return ACMCAP, status


# ======================== Cloud Property Calculations ========================

def get_d_lwc_dz(ACMCAP: xr.Dataset) -> xr.DataArray:
    """Compute vertical gradient of liquid water content using second order central differences."""
    d_lwc_dz = (ACMCAP.liquid_water_content.where(ACMCAP.liquid_water_content > 0).shift(JSG_height=1) - 
                ACMCAP.liquid_water_content.where(ACMCAP.liquid_water_content > 0).shift(JSG_height=-1)) / 2
    return d_lwc_dz


def get_ER_cldtop(ds: xr.Dataset) -> xr.DataArray:
    """Select the mean liquid effective radius of the 2 uppermost cloud layers."""
    valid_cth = ~np.isnan(ds.cth)
    ER_cld_top_0 = ds.liquid_effective_radius.sel(JSG_height=ds.cth.fillna(0))
    ER_cld_top_1 = ds.liquid_effective_radius.sel(JSG_height=ds.cth.fillna(0)+1)
    ER_cld_top = ((ER_cld_top_0 + ER_cld_top_1)/2).where(valid_cth, drop=True)
    return ER_cld_top


def get_cloud_cover(cloud_mask_1d: xr.DataArray, window_size: int = 100) -> xr.DataArray:
    """Compute cloud cover over a rolling window."""
    if cloud_mask_1d.shape[0] < window_size:
        print('Not enough data for cloud cover calculation')
        return cloud_mask_1d * np.nan

    return cloud_mask_1d.rolling(along_track=window_size, center=True).construct('window').reduce(np.nanmean, dim='window')


def get_skewness_and_inhomogeneity(lwp: xr.DataArray, windowsize: int = 100) -> Tuple[xr.DataArray, xr.DataArray]:
    """Compute skewness and inhomogeneity of LWP, considering only pixels with lwp > 0."""
    def skew_only_positive(x, axis):
        x = np.where(x > 0, x, np.nan)
        return scipy.stats.skew(x, axis=axis, nan_policy='omit')

    def inhomog_only_positive(x, axis):
        x = np.where(x > 0, x, np.nan)
        mean = np.nanmean(x, axis=axis)
        std = np.nanstd(x, axis=axis)
        return std / mean

    lwp_skew = lwp.rolling(along_track=windowsize, center=True).reduce(skew_only_positive)
    lwp_inhomog = lwp.rolling(along_track=windowsize, center=True).reduce(inhomog_only_positive)

    return lwp_skew, lwp_inhomog


def get_surface_rain_rate(ds: xr.Dataset) -> xr.DataArray:
    """Compute surface rain rate between cloud base and surface."""
    surface_rain_rate = ds.rain_rate.sel(JSG_height = ds.surface_height_JSG.fillna(1) - 1)
    return surface_rain_rate.where(ds.cloud_mask_1d == 1)

def get_rain_rate_at_cbh(ds: xr.Dataset) -> xr.DataArray:
    """Compute rain rate at cloud base height."""
    return ds.rain_rate.sel(JSG_height = ds.cbh.fillna(0) + 1).where(ds.cloud_mask_1d == 1)

def get_max_rain_rate(ds: xr.Dataset) -> xr.DataArray:
    """Compute maximum rain rate."""
    rain_rate_below_cth = ds.rain_rate.where(
        (ds.JSG_height > ds.cth) & (ds.lwp > 0)
    )
    return rain_rate_below_cth.max(dim='JSG_height').where(ds.cloud_mask_1d == 1)

def get_rain_water_content_at_cbh(ds: xr.Dataset) -> xr.DataArray:
    """Compute rain water content at cloud base height."""
    return ds.rain_water_content.sel(JSG_height = ds.cbh.fillna(0) + 1).where(ds.cloud_mask_1d == 1)

# ======================== Meteorological Calculations ========================

def potential_temperature(T: xr.DataArray, P: xr.DataArray) -> xr.DataArray:
    """Calculate potential temperature."""
    return T * (1000 / P) ** 0.286


def get_LTS(ds: xr.Dataset) -> xr.DataArray:
    """Calculate Lower Tropospheric Stability."""
    T_surface = ds.temperature_at_2_metres
    JSG_700hPa = np.abs(ds.pressure - 70000).idxmin(dim='JSG_height')
    T_700 = ds.temperature.sel(JSG_height=JSG_700hPa.fillna(0))
    T_700 = T_700.where(~np.isnan(JSG_700hPa))

    theta_surface = potential_temperature(T_surface, 1000)
    theta_700 = potential_temperature(T_700, 700)

    LTS = theta_700 - theta_surface
    return LTS


def get_RH_at_BLH(ds: xr.Dataset) -> xr.DataArray:
    """Get relative humidity at boundary layer height."""
    BLH_JSG = np.abs(ds.height - ds.boundary_layer_height).idxmin(dim='JSG_height')
    return ds.relative_humidity.sel(JSG_height=BLH_JSG)


def get_RH_below_BLH(ds: xr.Dataset) -> xr.DataArray:
    """Get mean relative humidity below boundary layer height."""
    BLH_JSG = np.abs(ds.height - ds.boundary_layer_height).idxmin(dim='JSG_height')
    relative_humidity_below_bl = ds.relative_humidity.where(ds.JSG_height > BLH_JSG)
    return relative_humidity_below_bl.mean(dim='JSG_height')

def get_RH_above_BLH(ds: xr.Dataset) -> xr.DataArray:
    """Get mean relative humidity in a 1km thick layer above boundary layer height."""
    BLH_JSG = np.abs(ds.height - ds.boundary_layer_height).idxmin(dim='JSG_height')
    relative_humidity_1km_above_bl = ds.relative_humidity.where((ds.JSG_height < BLH_JSG) & (ds.JSG_height > BLH_JSG - 10))
    return relative_humidity_1km_above_bl.mean(dim='JSG_height')

def get_RH_below_CTH(ds: xr.Dataset) -> xr.DataArray:
    """Get mean relative humidity below cloud top height."""
    relative_humidity_below_cth = ds.relative_humidity.where(ds.JSG_height > ds.cth)
    return relative_humidity_below_cth.mean(dim='JSG_height')

def get_RH_above_CTH(ds: xr.Dataset) -> xr.DataArray:
    """Get mean relative humidity above cloud top height."""
    return ds.relative_humidity.where(
        (ds.JSG_height <= ds.cth - 5) & (ds.JSG_height > ds.cth - 10)
    ).mean(dim='JSG_height')


def get_EIS(ds: xr.Dataset) -> xr.DataArray:
    """Calculate Estimated Inversion Strength."""
    theta = potential_temperature(ds.temperature, ds.pressure)
    dtheta_dz = (theta.shift(JSG_height=1) - theta.shift(JSG_height=-1)) / 2

    max_JSG_height = ds.JSG_height.max()
    dtheta_dz_slice = dtheta_dz.sel(JSG_height=slice(180, max_JSG_height))
    
    if dtheta_dz_slice.size == 0:
        EIS = xr.full_like(dtheta_dz.isel(JSG_height=0), np.nan)
        EIS = EIS.drop_vars('JSG_height')
    else:
        EIS = dtheta_dz_slice.max(dim='JSG_height')
    
    return EIS


def dew_point_temperature(T: xr.DataArray, RH: xr.DataArray) -> xr.DataArray:
    """Calculate dew point temperature from temperature and relative humidity."""
    a, b = 17.27, 237.7
    T_celsius = T - 273.15
    alpha = ((a * T_celsius) / (b + T_celsius)) + np.log(RH)
    dew_point = (b * alpha) / (a - alpha)
    return dew_point + 273.15


def get_LCL(ds: xr.Dataset) -> xr.DataArray:
    """Calculate Lifting Condensation Level."""
    JSG_100m = np.abs(ds.height - 100).idxmin(dim='JSG_height')
    T_surf = ds.temperature.sel(JSG_height=JSG_100m)
    RH_surf = ds.relative_humidity.sel(JSG_height=JSG_100m)
    Td = dew_point_temperature(T_surf, RH_surf)
    return 125 * (T_surf - Td)


def get_CBH_minus_LCL(ds: xr.Dataset) -> xr.DataArray:
    """Calculate difference between cloud base height and LCL."""
    valid_cbh = ~np.isnan(ds.cth)
    cbh_height = ds.height.sel(JSG_height=ds.cbh.fillna(0))
    cbh_height = cbh_height.where(valid_cbh)
    return cbh_height - get_LCL(ds)

def add_XMET_cloud_and_rain_variables(dsp: xr.Dataset) -> xr.Dataset:
    def water_content_per_volume(specific_water_content, P, T, RH):
        # derivation of formula:
        # specific water content q = mass_liq / (mass_air + mass_water_vapor + mass_liq)
        # mass_liq / V = q/(1-q) (mass_air + mass_water_vapor) / V
        # using p V = m R T  --> m/V = p / (R T)
        # --> mass_liq / V = q/(1-q) (P/ (R_air T) + RH e_sat / (R_vapor T))
        q = specific_water_content
        R_d = 287.05  # J/(kg·K)
        R_v = 461.5   # J/(kg·K)
        def e_s(T):
            # Tetens formula for saturation vapor pressure over liquid water
            return 6.112 * np.exp((17.67 * (T - 273.15)) / (T - 29.65)) * 100  # Convert hPa to Pa
        
        return q /(1-q) * (P / (R_d * T) + RH * e_s(T) / (R_v * T))

    xmet_cloud_mask = (dsp['cloud_cover'] > 0.2) | (dsp['specific_cloud_liquid_water_content'] > 1e-5)
    xmet_cloud_mask.name = 'xmet_cloud_mask'
    tmp_cth_cbh = get_cth_and_cbh(xmet_cloud_mask)
    all_clouds_are_low_clouds = np.all(tmp_cth_cbh.cth.fillna(250) > 210)
    if all_clouds_are_low_clouds:
        xmet_low_cloud_mask = xmet_cloud_mask
    else:
        xmet_low_cloud_mask = get_low_cloud_mask(xmet_cloud_mask, filter_by='cth', ds_boundary_layer_height=dsp['latitude']*0 + 3000, ds_height=dsp['height'])
        tmp_cth_cbh = get_cth_and_cbh(xmet_low_cloud_mask)

    dsp['xmet_cth'] = tmp_cth_cbh['cth']
    dsp['xmet_cbh'] = tmp_cth_cbh['cbh']
    dsp['xmet_cth_m'] = dsp.height.sel(JSG_height = dsp.xmet_cth.fillna(0)).where(dsp.xmet_cth.notnull())
    dsp['xmet_cbh_m'] = dsp.height.sel(JSG_height = dsp.xmet_cbh.fillna(0)).where(dsp.xmet_cbh.notnull())
    dsp['xmet_low_cloud_cover_binary'] = xmet_low_cloud_mask.max(dim='JSG_height')
    dsp['xmet_low_cloud_cover_original'] = dsp['low_cloud_cover']

    dsp['xmet_cloud_liquid_water_content'] = water_content_per_volume(dsp['specific_cloud_liquid_water_content'], dsp['pressure'], dsp['temperature'], dsp['relative_humidity'])
    dsp['xmet_rain_water_content'] = water_content_per_volume(dsp['specific_rain_water_content'], dsp['pressure'], dsp['temperature'], dsp['relative_humidity'])
    dsp['xmet_lwp'] = (dsp.where(xmet_low_cloud_mask)['xmet_cloud_liquid_water_content']).sum(dim='JSG_height') * 100
    dsp['xmet_lwp'].attrs['units'] = 'kg/m^2'
    dsp['xmet_rwp'] = (dsp.where(dsp.JSG_height >= dsp.xmet_cth)['xmet_rain_water_content']).sum(dim='JSG_height') * 100
    dsp['xmet_rwp'].attrs['units'] = 'kg/m^2'
    dsp['xmet_rwp_subcloud'] = (dsp.where(dsp.JSG_height >= dsp.xmet_cbh)['xmet_rain_water_content']).sum(dim='JSG_height') * 100
    dsp['xmet_rwp_subcloud'].attrs['units'] = 'kg/m^2'
    dsp['xmet_rwc_cbh'] = dsp['xmet_rain_water_content'].sel(JSG_height = dsp.xmet_cbh.fillna(0) + 1).where(dsp.xmet_low_cloud_cover_binary == 1)
    dsp['xmet_rwc_cbh'].attrs['units'] = 'kg/m^3'

    return dsp


# ======================== MSI Data Processing ========================

def find_MCOP_file_matching_ACMCAP_filename(ACMCAP_filename: str, dir: str) -> Tuple[str, int]:
    """Find MCOP files that match the ACMCAP filename."""
    import re 

    ACMCAP_filename = ACMCAP_filename.split('/')[-1]
    production_datetime = ACMCAP_filename.split('_')[-2]
    start_datetime = ACMCAP_filename.split('_')[-3]
    tmp = ACMCAP_filename.replace('ACM_CAP_2B', 'MSI_COP_2A').replace(production_datetime, '*').replace(start_datetime, start_datetime[0:9]+'*')
    tmp = re.sub(r'ECA_EX[A-Z]{2}', 'ECA_EX*', tmp)
    
    filenames_MCOP = glob.glob(f'{dir}/{tmp}')
    
    if len(filenames_MCOP) == 0:
        return 'no_MCOP_file_found', -1
    elif len(filenames_MCOP) > 1:
        return 'more_than_1_MCOP_file_found', -1
    else:
        return filenames_MCOP[0], 0


def load_matching_MCOP(ACMCAP_filename: str, lon_min: float, lon_max: float, 
                      lat_min: float, lat_max: float) -> Optional[xr.Dataset]:
    """Load the MCOP file that matches the ACMCAP filename."""
    filename_MCOP, status = find_MCOP_file_matching_ACMCAP_filename(ACMCAP_filename, f'{MCOP_data_dir}/*/*/*/')
    
    if status != 0:
        print(f'No MCOP data found for ACMCAP filename {ACMCAP_filename}')
        return None
    # try three times to load MCOP before throwing an error
    for attempt in range(3):
        print(f'\t --> attempt {attempt + 1}')
        try:
            MCOP = ecio.load_MCOP(filename_MCOP)
            break
        except Exception as e:
            if attempt == 2:
                raise ImportError(f'Error loading MCOP data after 3 attempts: {e}')
            else:
                print(f'Error loading MCOP data, retrying. Error message was: {e}')
    
    MCOP = MCOP.assign_coords({
        'along_track': (["along_track"], MCOP['along_track'].data, MCOP['along_track'].attrs),
        'across_track': (['across_track'], MCOP['across_track'].data, MCOP['across_track'].attrs)
    })
    
    MCOP2 = MCOP.where(
        (MCOP.latitude > lat_min) & (MCOP.latitude < lat_max) & 
        (MCOP.longitude > lon_min) & (MCOP.longitude < lon_max), 
        drop=True
    )
    MCOP2 = MCOP2[['latitude','longitude','cloud_water_path','cloud_optical_thickness','cloud_effective_radius','cloud_water_path']]
    MCOP2['time'] = MCOP['time'].sel(along_track=MCOP2.along_track)

    return MCOP2


def LvL(cloud_mask: xr.DataArray) -> Tuple[float, float]:
    """Calculate Length variation Lumpiness metric."""
    if np.all(cloud_mask == 0) or np.all(cloud_mask == 1):
        # If cloud fraction is 0 or 1, the LvL is not defined. Return NaN
        return np.nan, np.nan
    
    # Convert cloud_mask to double
    cloud_mask = cloud_mask.astype(float)
    sz = cloud_mask.shape
    mnsz = min(sz)  # minimum scale of the field
    #sz_cloud_mask = sz[0] * sz[1]
    sz_cloud_mask = cloud_mask.count().values
    #p = np.sum(cloud_mask) / sz_cloud_mask  # cloud fraction
    p = cloud_mask.mean().values

    # estimating the ideal length to capture almost 100% of the histogram
    # we want that the error < exp(-12) for the perfect rand case
    mxln = int(np.floor(abs(12 / np.log(p))) + 1)
    mxln = min(mxln, mnsz)

    # The cloud part
    # Flatenning along the two directions
    # then we need to divide c1 and c2 by two

    B = cloud_mask.values.flatten()  # rows
    C = cloud_mask.values.flatten(order='F') #columns 
    B = np.concatenate((C, B))

    L, num_labels = label(B)
    # Label connected components in the binary image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(L.astype(np.uint8))

    # Extract area of each connected component
    areas = stats[1:, cv2.CC_STAT_AREA]
    # Adding one to the max area
    mx_ar = np.max(areas) + 1   

    # To avoid single histogram
    if mx_ar < mxln:
        mx_ar = mxln
    else:
        mxln = mx_ar

    # Get the cloud chord length counts - c1    
    c1, _ = np.histogram(areas, bins=np.arange(1, mx_ar + 2))
    # Correct for flattening in two directions
    c1 = c1 / 2
    s1 = np.sum(c1)

    # Now, the theortical calculations ct, nt for a given cloud fraction (p)
    nt1 = np.arange(1, mxln + 1)
    ct1 = (sz_cloud_mask * (1 - p) ** 2) * p ** nt1
    st1 = np.sum(ct1)
    nt1 = nt1.astype(int)


    # Get the KS score for the cloud part
    adf1 = np.abs(np.cumsum(ct1 / st1) - np.cumsum(c1 / s1))
    KS1 = np.max(adf1)


    # The void part
    q = 1 - p # void fraction
    mxln = int(np.floor(abs(12 / np.log(q))) + 1)
    mxln = min(mxln, mnsz)
    B = -B + 1

    L, num_labels = label(B)
    # Label connected components in the binary image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(L.astype(np.uint8))

    # Extract area of each connected component
    areas = stats[1:, cv2.CC_STAT_AREA]
    # Adding one to the max area
    mx_ar = np.max(areas) + 1   

    if mx_ar < mxln:
        mx_ar = mxln
    else:
        mxln = mx_ar

    c2, _ = np.histogram(areas, bins=np.arange(1, mx_ar + 2))
    c2 = c2 / 2
    s2 = np.sum(c2)

    nt2 = np.arange(1, mxln + 1)
    ct2 = (sz_cloud_mask * (1 - q) ** 2) * q ** nt2
    st2 = np.sum(ct2)
    nt2 = nt2.astype(int)

    # Get the KS score for the void part
    adf2 = np.abs(np.cumsum(ct2 / st2) - np.cumsum(c2 / s2))
    KS2 = np.max(adf2)

    return KS1, KS2


def get_cloud_params_from_msi(MCOP: xr.Dataset, ACMCAP: xr.Dataset, 
                             window_size: int = 200) -> Tuple[List, List, List, List, List, List, List]:
    """Compute cloud parameters from MSI data in a rolling window along nadir path."""
    idx_across_track_nadir = 284
    half_window_size = int(window_size / 2)

    results = {
        'cwp_skew': [], 'cwp_inhomogeneity': [], 'cloud_cover': [],
        'ks1': [], 'ks2': [], 'ks1_median': [], 'ks2_median': [], 'ks1_gt01': [], 'ks2_gt01': [],
        'effective_radius': [], 'optical_thickness': [], 'cloud_water_path': []
    }

    for i in range(ACMCAP.along_track.size):
        if ACMCAP.cloud_mask_1d[i] == 0:
            for key in results:
                results[key].append(np.nan)
            continue
        
        along_track_idx = np.abs(
            MCOP.sel(across_track=idx_across_track_nadir).latitude - ACMCAP.latitude[i]
        ).argmin().values

        CWP_window = MCOP.cloud_water_path.isel(
            along_track=slice(along_track_idx-half_window_size, along_track_idx+half_window_size)
        ).sel(across_track=slice(
            max(idx_across_track_nadir - half_window_size, 0), 
            idx_across_track_nadir + half_window_size
        ))

        CWP_window_clean = CWP_window.where(np.isfinite(CWP_window))
        
        # Calculate statistics
        results['cwp_skew'].append(scipy.stats.skew(CWP_window_clean.values.flatten(), bias=False, nan_policy='omit'))
        results['cwp_inhomogeneity'].append(np.nanstd(CWP_window_clean.values.flatten()) / np.nanmean(CWP_window_clean.values.flatten()))
        results['cloud_cover'].append(float(xr.where(CWP_window_clean > 0, 1, 0).mean().values))
        
        cld_mask = xr.where(CWP_window_clean > 0, 1, 0)
        cld_mask_median = xr.where(CWP_window_clean > CWP_window_clean.median(), 1, 0)
        cld_mask_gt01 = xr.where(CWP_window_clean > 0.1, 1, 0)
        ks1, ks2 = LvL(cld_mask)
        ks1_median, ks2_median = LvL(cld_mask_median)
        ks1_gt01, ks2_gt01 = LvL(cld_mask_gt01)

        results['ks1'].append(ks1)
        results['ks2'].append(ks2)
        results['ks1_median'].append(ks1_median)
        results['ks2_median'].append(ks2_median)
        results['ks1_gt01'].append(ks1_gt01)
        results['ks2_gt01'].append(ks2_gt01)

        MCOP_at_ACMCAP_latlon = MCOP.isel(along_track=along_track_idx).sel(across_track=idx_across_track_nadir)
        results['effective_radius'].append(float(MCOP_at_ACMCAP_latlon.cloud_effective_radius.values))
        results['optical_thickness'].append(float(MCOP_at_ACMCAP_latlon.cloud_optical_thickness.values))
        results['cloud_water_path'].append(float(MCOP_at_ACMCAP_latlon.cloud_water_path.values))

    return tuple(results.values())


def add_msi_cloud_params_to_acmcap(ACMCAP_filename: str, ACMCAP: xr.Dataset, 
                                  lon_min: float, lon_max: float, lat_min: float, lat_max: float, 
                                  window_size: int = 200) -> Tuple[xr.Dataset, Optional[xr.Dataset]]:
    """Add cloud parameters from MSI to ACMCAP."""
    no_valid_msi_data_flag = False
    if ACMCAP.daytime.sum() == 0:  # No MSI data at night
        no_valid_msi_data_flag = True
        MCOP = None
    else:
        MCOP = load_matching_MCOP(ACMCAP_filename, lon_min, lon_max, lat_min, lat_max)
        if MCOP is not None:
            (MCOP_cwp_skew, MCOP_cwp_inhomogeneity, MCOP_cloud_cover, 
             MCOP_ks1, MCOP_ks2, MCOP_ks1_median, MCOP_ks2_median, MCOP_ks1_gt01, MCOP_ks2_gt01,
             MCOP_effective_radius, MCOP_optical_thickness, MCOP_cloud_water_path) = get_cloud_params_from_msi(MCOP, ACMCAP, window_size)
        else:
            no_valid_msi_data_flag = True

    if no_valid_msi_data_flag:
        nan_array = np.nan * np.ones(ACMCAP.along_track.size)
        MCOP_cwp_skew, MCOP_cwp_inhomogeneity, MCOP_cloud_cover = nan_array, nan_array, nan_array
        MCOP_ks1, MCOP_ks2, MCOP_ks1_median, MCOP_ks2_median, MCOP_ks1_gt01, MCOP_ks2_gt01 = nan_array, nan_array, nan_array, nan_array, nan_array, nan_array
        MCOP_effective_radius, MCOP_optical_thickness, MCOP_cloud_water_path = nan_array, nan_array, nan_array

    # Add parameters to ACMCAP dataset
    suffix = str(window_size)
    ACMCAP[f'MCOP_cwp_skew_{suffix}'] = (('along_track'), MCOP_cwp_skew)
    ACMCAP[f'MCOP_cwp_inhomogeneity_{suffix}'] = (('along_track'), MCOP_cwp_inhomogeneity)
    ACMCAP[f'MCOP_cloud_cover_{suffix}'] = (('along_track'), MCOP_cloud_cover)
    ACMCAP[f'MCOP_ks1_{suffix}'] = (('along_track'), MCOP_ks1)
    ACMCAP[f'MCOP_ks2_{suffix}'] = (('along_track'), MCOP_ks2)
    ACMCAP[f'MCOP_ks1_median_{suffix}'] = (('along_track'), MCOP_ks1_median)
    ACMCAP[f'MCOP_ks2_median_{suffix}'] = (('along_track'), MCOP_ks2_median)
    ACMCAP[f'MCOP_ks1_gt01_{suffix}'] = (('along_track'), MCOP_ks1_gt01)
    ACMCAP[f'MCOP_ks2_gt01_{suffix}'] = (('along_track'), MCOP_ks2_gt01)
    ACMCAP['MCOP_effective_radius'] = (('along_track'), MCOP_effective_radius)
    ACMCAP['MCOP_optical_thickness'] = (('along_track'), MCOP_optical_thickness)
    ACMCAP['MCOP_cloud_water_path'] = (('along_track'), MCOP_cloud_water_path)

    return ACMCAP, MCOP


# ======================== Aerosol Calculations ========================

def get_aerosol_extinction_at_cth(dsp: xr.Dataset, n_levels_above_cth: int = 3) -> xr.DataArray:
    """Get aerosol extinction at cloud top height."""
    valid_cth = ~np.isnan(dsp.cth)
    cth_fillna = dsp.cth.fillna(10)
    
    aerosol_ext_list = []
    for i in range(n_levels_above_cth):
        aerosol_ext_at_i_levels_above_cth = dsp.aerosol_extinction.sel(JSG_height=(cth_fillna - i))
        aerosol_ext_list.append(aerosol_ext_at_i_levels_above_cth)
    
    aerosol_ext = xr.concat(aerosol_ext_list, dim='tmp')
    aerosol_ext_at_cth = aerosol_ext.mean(dim='tmp')
    aerosol_ext_at_cth = aerosol_ext_at_cth.where(valid_cth, drop=True)
    aerosol_ext_at_cth.name = 'aerosol_extinction_at_cth'

    return aerosol_ext_at_cth


def get_mean_aerosol_extinction_below_cth(dsp: xr.Dataset) -> Tuple[xr.DataArray, xr.DataArray]:
    """Compute mean aerosol extinction below the mean cloud top height of the frame."""
    mean_cth = dsp.cth.mean(dim='along_track')
    
    if mean_cth.isnull():
        return np.nan, 0
    
    aer_ext_below_cth_profiles = dsp.aerosol_extinction.where(
        (dsp.aerosol_extinction > 0) | np.isnan(dsp.cth)
    ).where(dsp.JSG_height >= mean_cth)
    
    mean_aerosol_extinction_below_cth = aer_ext_below_cth_profiles.mean()
    counts_aerosol_extinction_below_cth = aer_ext_below_cth_profiles.count()
    
    da_mean = xr.DataArray(
        dims='along_track', 
        data=[mean_aerosol_extinction_below_cth] * dsp.along_track.size, 
        coords={'along_track': dsp.along_track}
    )
    da_counts = xr.DataArray(
        dims='along_track', 
        data=[counts_aerosol_extinction_below_cth] * dsp.along_track.size, 
        coords={'along_track': dsp.along_track}
    )
    
    return da_mean, da_counts


# ======================== Plotting Functions ========================

def make_overview_plot_v2(dsp: xr.Dataset, MCOP: Optional[xr.Dataset], save_as: str) -> None:
    """Create overview plot version 2."""
    fig, axs = plt.subplots(6, 1, figsize=(30, 24), gridspec_kw={'height_ratios': [1, 1, 0.6, 0.6, 0.6, 0.6]})
    
    # Plot 1: Rain rate and liquid water content
    ax = axs[0]
    if add_msi_vars:
        ecplt.plot_EC_2D(ax, dsp, 'rain_rate', "$R$", scale_factor=3600., units="mmh$^{-1}$", 
                        plot_scale='log', plot_range=[1e-3, 1e1], hmax=3000, cmap='Blues')
        ax.collections[0].colorbar.remove()
    
    ecplt.plot_EC_2D(ax, dsp, 'liquid_water_content', "$L$", units="kg m$^{-3}$", 
                     plot_scale='log', plot_range=[1e-6, 1e-3], cmap='chiljet2', hmax=3000)
    if add_xmet_vars:
        lcl = get_LCL(dsp)
        sns.lineplot(ax=ax, x=dsp.time, y=lcl, color='black', lw=1)

    # Plot 2: MCOP cloud water path
    ax = axs[1]
    if add_msi_vars:
        if MCOP is not None:
            ax.sharex(axs[0])
            idx_across_track_nadir = 284
            _cm = ax.imshow(
                MCOP.cloud_water_path.T, aspect='auto', origin='lower',
                extent=[MCOP.time.min().values, MCOP.time.max().values, 
                    MCOP.across_track.min().values, MCOP.across_track.max().values],
                norm=ecplt.LogNorm(1e-2, 1), cmap='viridis', interpolation='nearest'
            )
            ax.axhline(idx_across_track_nadir, color='k', linestyle='--', lw=2)
            ax.set_title('')
            ecplt.add_colorbar(ax, _cm, 'LWP [kg m$^{-2}$]', horz_buffer=0.01, width_ratio='1%')
    else:
        ecplt.plot_EC_2D(ax, dsp, 'rain_rate', "$R$", scale_factor=3600., units="mmh$^{-1}$", 
                        plot_scale='log', plot_range=[1e-3, 1e1], hmax=3000, cmap='Blues')

    ax=axs[2]
    ax.sharex(axs[0])
    sns.scatterplot(ax=ax, x=dsp.time, y=dsp.liquid_optical_depth, color='red', lw=0, s=8).set(ylim=(0,18))
    ax.set_ylabel('OT', fontsize=20, color='red')
    if add_msi_vars:
        ax2 = ax.twinx()
        sns.scatterplot(ax=ax2, x=dsp.time, y=dsp.MCOP_optical_thickness, color='orange', lw=0, s=8).set(ylim=(0,18))
        ax2.set_ylabel('MCOP OT', fontsize=20, color='orange')

    ax = axs[3]
    sns.scatterplot(ax=ax, x=dsp.time, y=dsp.lwp, color='green', lw=0, s=8).set(yscale='log', ylim=(1e-5,1e0))
    ax.set_ylabel('LWP /kg m$^{-2}$', fontsize=20, color='green')
    if add_msi_vars:
        ax2 = ax.twinx()
        sns.scatterplot(ax=ax2, x=dsp.time, y=dsp.MCOP_cloud_water_path, color='orange', lw=0, s=8).set(yscale='log', ylim=(1e-5,1e0))
        ax2.set_ylabel('MCOP LWP', fontsize=20, color='orange')

    ax = axs[4]
    ax.sharex(axs[0])
    sns.scatterplot(ax=ax, x=dsp.time, y=dsp.rain_rate_cbh * 3600, color='blue', lw=0, s=8).set(yscale='log', ylim=(1e-3, 1e1))
    ax.set_ylabel('$R$ at CBH /mm h$^{-1}$', fontsize=20, color='blue')
    if add_msi_vars:
        ax2 = ax.twinx()
        sns.scatterplot(ax=ax2, x=dsp.time, y=dsp.MCOP_cloud_cover_256, color='black', lw=0, s=8).set(ylim=(0.4, 1.05))
        ax2.set_ylabel('MCOP CC', fontsize=20, color='black')
    
    ax=axs[5]
    ax.sharex(axs[0])
    sns.scatterplot(ax=ax, x=dsp.time, y=dsp.Nd, color='darkviolet', lw=0, s=8).set(yscale='log', ylim=(1e7,5e8))
    ax.set_ylabel('Nd / m$^{-3}$', fontsize=20, color='darkviolet')
    ax2 = ax.twinx()
    sns.scatterplot(ax=ax2, x=dsp.time, y=dsp.MSI_BT_108, color='orange', lw=0, s=8).set(ylim=(270, 300))
    ax2.set_ylabel('BT_108', fontsize=20, color='orange')
    if add_msi_vars:
        ax2 = ax.twinx()
        sns.scatterplot(ax=ax2, x=dsp.time, y=dsp.MCOP_cwp_skew_256, color='orange', lw=0, s=8).set(ylim=(0,10))
        ax2.set_ylabel('MCOP skew', fontsize=20, color='orange')
    
    for ax in axs[1:6]:
        ax.tick_params(axis='x', labelsize=15)
    
    fig.tight_layout()
    plt.savefig(save_as, dpi=300)
    plt.close(fig)


# ======================== Utility Functions ========================

def printProgressBar(iteration: int, total: int, prefix: str = '', suffix: str = '', 
                    decimals: int = 1, length: int = 50, fill: str = '█', printEnd: str = "\r") -> None:
    """Print a simple progress bar."""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    if iteration == total: 
        print()


def compute_localtime(ds, timevar='time', lonvar='longitude'):
    localtime = ds[timevar] + [np.timedelta64(int(l), 's') for l in np.round(ds[lonvar].values[:]/15*60*60)]
    return localtime


def get_selected_variables() -> List[str]:
    """Return list of variables to select for output dataset."""
    selected_variables_ACMCAP = [
        'time', 
        'latitude', 
        'longitude',
        'stratus_region_index',
        'daytime',
        'lwp', 
        'ER', 
        'ER_passive_perspective',
        'Nd',
        'lwp_error', 
        'Nd_error', 
        'liquid_classification_mean', 
        'liquid_optical_depth',
        'cth', 
        'cbh', 
        'cth_m', 
        'cbh_m', 
        'cth_all_clouds', 
        'multilayer', 
        'N_cloudy_pixels',
        'N_high_cc',
        #'aerosol_optical_depth', 'aerosol_extinction_below_cth_mean', 'aerosol_extinction_below_cth_counts',
        'rain_rate_mean',
        'rain_rate_surface',
        'rain_rate_cbh',
        'rain_rate_max',
        #'rain_rate_incloud', 
        'rain_rate_subcloud', 
        'rwc_cbh',
        'rwp',
        'rwp_subcloud',
        'rain_median_volume_diameter', 
        'rain_normalized_number_concentration',
        'doppler_velocity_mean',
        #'cloud_cover_100km', 'cloud_cover_200km', 
        #'cbh_roughness_100km', 'cbh_roughness_200km',
        #'cbh_min_max_128km',
        #'lwp_skew_100km', 'lwp_skew_200km',
        #'lwp_inhomogeneity_100km', 'lwp_inhomogeneity_200km', 
        #'cbh_minus_lcl', 
        #'coupled_fraction',
        'liquid_optical_depth',
        'CPR_path_integrated_attenuation',
        'MSI_BT_108'
    ]
    selected_variables_XMET = [
        'boundary_layer_height',
        'lcl',
        'sea_surface_temperature', 'total_column_water_vapour', 
        'lower_tropospheric_stability', 'inversion_strength', 
        'relative_humidity_below_bl', 'relative_humidity_above_bl', 
        'relative_humidity_below_cth', 'relative_humidity_above_cth', 'relative_humidity_at_3km', 
        'upward_air_velocity_at_3km', 'eastward_wind_at_10_metres', 'northward_wind_at_10_metres',
    ]
    selected_cloud_variables_XMET = [
        'xmet_cth',
        'xmet_cbh',
        'xmet_cth_m',
        'xmet_cbh_m',
        'xmet_low_cloud_cover_binary',
        'xmet_low_cloud_cover_original',
        'xmet_lwp',
        'xmet_rwp',
        'xmet_rwp_subcloud',
        'xmet_rwc_cbh'
    ]
    if add_xmet_vars:
        if add_xmet_cloud_vars:
            return selected_variables_ACMCAP + selected_variables_XMET + selected_cloud_variables_XMET
        return selected_variables_ACMCAP + selected_variables_XMET
    else:
        return selected_variables_ACMCAP


# ======================== Main Processing Function ========================

def main():
    """Main processing function."""
    with open(INPUT_FILE_orbits_in_sc_regions, 'r') as file:
        orbits_in_stratus_regions = json.load(file)

    df_stratus = None
    failed_files = []

    for region in regions_of_interest:
        number_of_orbits_to_process = len(orbits_in_stratus_regions[region])
        for i, item in enumerate(orbits_in_stratus_regions[region]):
            frame = item['frame']
            orbitNumber = item['orbitNumber']
            orbit = orbitNumber+frame
            filename = item['filename']
            date = filename.split('ACM_CAP_2B_')[1].split('_')[0][0:8]
            year = date[0:4]
            month = date[4:6]
            day = date[6:8]
            local_file_path = f"{ACMCAP_data_dir}{year}/{month}/{day}/{filename}"

            print(f'\nProcessing file {i+1} of {number_of_orbits_to_process} in {region}: {orbit}')
            printProgressBar(i, number_of_orbits_to_process, prefix='Progress:', suffix='Complete', length=50)
            print()

            selected_variables = get_selected_variables()

            try:
                if os.path.isfile(local_file_path):
                    # Load ACMCAP data from local machine
                    print('Loading ACM-CAP data from local file...')
                    ds = ecio.load_ACMCAP(local_file_path)
                    encoding = ds.encoding
        
                    # Select region
                    # TODO do same cropping as for MAAP data
                    lat_min, lat_max = stratus_regions[region]['lat']
                    lon_min, lon_max = stratus_regions[region]['lon']
                    ds = ds.where((ds.latitude > lat_min) & (ds.latitude < lat_max) & 
                                (ds.longitude > lon_min) & (ds.longitude < lon_max), drop=True)
                    ds.encoding = encoding.copy()
                else:
                    # Load ACMCAP data via MAAP
                    print('Loading ACM-CAP data via MAAP...')
                    ACMCAP_list = maap.get_acmcap_via_maap(
                                        region=region, 
                                        frame=frame,
                                        orbitNumber=orbitNumber,
                                        max_items=1,
                                        )
                    ds = ACMCAP_list[0][0]
                    encoding = ds.encoding


                ds_length = len(ds.along_track)
                if ds_length == 0:
                    print(f'No data in stratus region {region} for file {orbit}. Skipping to next file...')
                    continue

                # Add XMET data
                if add_xmet_vars:
                    print('ACM-CAP data loaded. Adding X-MET temperature data')
                    XMET_2D_variables = ['temperature', 'pressure', 'upward_air_velocity', 'relative_humidity',
                                         'cloud_cover', 'specific_cloud_liquid_water_content', 'specific_rain_water_content']
                    XMET_1D_variables = ['temperature_at_2_metres', 'boundary_layer_height', 'skin_temperature', 
                                         'eastward_wind_at_10_metres', 'northward_wind_at_10_metres',
                                       'sea_surface_temperature', 'total_column_water_vapour', 'low_cloud_cover']

                    ds, xmet_status = add_XMET_temperature_to_ACMCAP(ds, filename, XMET_2D_variables, XMET_1D_variables)
                    if xmet_status != 0:
                        print(f'Error in adding XMET data. Continuing without XMET...')
                else:
                    xmet_status = -1
                    print('Continuing without XMET data...')

                # Process ACMCAP data
                dsp = prepare_ACMCAP_data(ds, xmet_status=xmet_status)

                # Add derived variables
                local_time = compute_localtime(dsp)
                dsp['daytime'] = (local_time.dt.hour > 6) & (local_time.dt.hour < 18)
                    
                # Compute precipitation properties
                if 'rain_rate_mean' in selected_variables:
                    dsp['rain_rate_mean'] = dsp.rain_rate.where(dsp.JSG_height > dsp.cth).mean(dim='JSG_height')

                if 'rain_rate' in selected_variables or 'rain_rate_surface' in selected_variables or \
                   'rain_rate_cbh' in selected_variables or 'rain_rate_max' in selected_variables or \
                   'rain_rate_subcloud' in selected_variables:
                    dsp['rain_rate_surface'] = get_surface_rain_rate(dsp)
                    dsp['rain_rate_cbh'] = get_rain_rate_at_cbh(dsp)
                    dsp['rain_rate_max'] = get_max_rain_rate(dsp)
                    dsp['rain_rate_subcloud'] = dsp.rain_rate.where((dsp.rain_rate > 0) & (dsp.JSG_height > dsp.cbh)).sum(dim='JSG_height') * 100

                if 'rain_median_volume_diameter' in selected_variables:
                    dsp['rain_median_volume_diameter'] = dsp.rain_median_volume_diameter.where((dsp.rain_water_content > 0) & (dsp.JSG_height > dsp.cth)).mean(dim='JSG_height')

                if 'rain_normalized_number_concentration' in selected_variables:
                    dsp['rain_normalized_number_concentration'] = dsp.rain_normalized_number_concentration.where((dsp.rain_water_content > 0) & (dsp.JSG_height > dsp.cth)).mean(dim='JSG_height')

                if 'rwc_cbh' in selected_variables:
                    dsp['rwc_cbh'] = get_rain_water_content_at_cbh(dsp)
                    dsp['rwp'] = dsp.rain_water_content.where((dsp.rain_water_content > 0) & (dsp.JSG_height > dsp.cth)).sum(dim='JSG_height') * 100
                    dsp['rwp_subcloud'] = dsp.rain_water_content.where((dsp.rain_water_content > 0) & (dsp.JSG_height > dsp.cbh)).sum(dim='JSG_height') * 100

                if 'doppler_velocity_mean' in selected_variables:
                    dsp['doppler_velocity_mean'] = dsp.CPR_doppler_velocity.where((dsp.JSG_height > dsp.cth)).mean(dim='JSG_height')
                
                # Cloud cover and roughness metrics
                if 'cloud_cover_100km' in selected_variables:
                    if ds_length >= 100:
                        dsp['cloud_cover_100km'] = get_cloud_cover(dsp.cloud_mask_1d, window_size=100)
                    else:
                        print(f'WARNING: skipping calculation of cloud cover for window size 100 km, since the length of the dataset is only {ds_length} points')
                        dsp['cloud_cover_100km'] = np.nan * dsp.cbh
                if 'cloud_cover_200km' in selected_variables:
                    if ds_length >= 200:
                        dsp['cloud_cover_200km'] = get_cloud_cover(dsp.cloud_mask_1d, window_size=200)
                    else:
                        print(f'WARNING: skipping calculation of cloud cover for window size 200 km, since the length of the dataset is only {ds_length} points')
                        dsp['cloud_cover_200km'] = np.nan * dsp.cbh

                if 'cbh_min_max_128km' in selected_variables:
                    if ds_length >= 128:
                        cbh_min_max_128km = (dsp.cbh.rolling(along_track=128, center=True).construct('window').reduce(np.nanmax, dim='window') - 
                                            dsp.cbh.rolling(along_track=128, center=True).construct('window').reduce(np.nanmin, dim='window'))
                        dsp['cbh_min_max_128km'] = cbh_min_max_128km.where(dsp.cloud_mask_1d == 1)
                    else:
                        print(f'WARNING: skipping calculation of cbh_min_max for window size 128 km, since the length of the dataset is only {ds_length} points')
                        dsp['cbh_min_max_128km'] = np.nan * dsp.cbh
                if 'cbh_roughness_100km' in selected_variables:
                    for window_size in [100, 200]:
                        if ds_length >= window_size:
                            cbh_roughness = dsp.cbh.rolling(along_track=window_size, center=True).construct('window').reduce(np.nanstd, dim='window')
                            dsp[f'cbh_roughness_{window_size}km'] = cbh_roughness.where(dsp.cloud_mask_1d == 1)
                            
                            lwp_skew, lwp_inhomogeneity = get_skewness_and_inhomogeneity(dsp.lwp, windowsize=window_size)
                            dsp[f'lwp_skew_{window_size}km'] = lwp_skew
                            dsp[f'lwp_inhomogeneity_{window_size}km'] = lwp_inhomogeneity
                        else:
                            print(f'WARNING: skipping calculation of cloud cover, roughness, skewness and inhomogeneity metrics for window size {window_size} km, since the length of the dataset is only {ds_length} points')
                            dsp[f'cbh_roughness_{window_size}km'] = np.nan * dsp.cbh
                            dsp[f'lwp_skew_{window_size}km'] = np.nan * dsp.lwp
                            dsp[f'lwp_inhomogeneity_{window_size}km'] = np.nan * dsp.lwp

                # Aerosol and meteorological variables
                if 'aerosol_extinction_below_cth_mean' in selected_variables and 'aerosol_extinction_below_cth_counts' in selected_variables:
                    da_mean_aerosol_extinction_below_cth, da_counts_aerosol_extinction_below_cth = get_mean_aerosol_extinction_below_cth(dsp)
                    dsp['aerosol_extinction_below_cth_mean'] = da_mean_aerosol_extinction_below_cth
                    dsp['aerosol_extinction_below_cth_counts'] = da_counts_aerosol_extinction_below_cth

                if add_xmet_vars and xmet_status == 0 and 'cbh_minus_lcl' in selected_variables:
                    dsp['cbh_minus_lcl'] = get_CBH_minus_LCL(dsp)
                else:
                    print('ATTENTION: No XMET data added, setting cbh_minus_lcl to NaN')
                    dsp['cbh_minus_lcl'] = np.nan * dsp.latitude
                dsp['coupling_state'] = xr.where(dsp.cbh_minus_lcl < 250, 1, 0)
                if 'coupled_fraction' in selected_variables and ds_length >= 100:
                    dsp['coupled_fraction'] = dsp.coupling_state.rolling(along_track=100, center=True).mean()

                if 'MSI_BT_108' in selected_variables:
                    dsp['MSI_BT_108'] = dsp.MSI_longwave_brightness_temperature.isel({'MSI_longwave_channel':1})

                dsp['stratus_region_index'] = stratus_region_index[region]

                # meteo variables
                if add_xmet_vars and xmet_status == 0:
                    dsp['lower_tropospheric_stability'] = get_LTS(dsp)
                    dsp['relative_humidity_below_bl'] = get_RH_below_BLH(dsp)
                    dsp['relative_humidity_above_bl'] = get_RH_above_BLH(dsp)
                    dsp['relative_humidity_below_cth'] = get_RH_below_CTH(dsp)
                    dsp['relative_humidity_above_cth'] = get_RH_above_CTH(dsp)
                    dsp['inversion_strength'] = get_EIS(dsp)
                    dsp['relative_humidity_at_3km'] = dsp.relative_humidity.sel(JSG_height=208)
                    dsp['upward_air_velocity_at_3km'] = dsp.upward_air_velocity.sel(JSG_height=208)
                    dsp['lcl'] = get_LCL(dsp)
                if add_xmet_cloud_vars and xmet_status == 0:
                    dsp = add_XMET_cloud_and_rain_variables(dsp)


                if add_lwc_profiles:
                    lwc = dsp.liquid_water_content.sel(JSG_height=slice(200,240))
                    # Convert to pandas DataFrame (along_track as index, columns are height_level)
                    lwc_df = lwc.to_pandas()
                    # Each row is a pandas Series; convert each row to a numpy array (or list)
                    lwc_as_arrays = lwc_df.apply(lambda row: row.values, axis=1)

                    rain_rate = dsp.rain_rate.where(dsp.JSG_height >= dsp.cth).sel(JSG_height=slice(200,240))
                    rain_rate_df = rain_rate.to_pandas()
                    rain_rate_as_arrays = rain_rate_df.apply(lambda row: row.values, axis=1)

                    rain_rate_below_cloud = dsp.rain_rate.where(dsp.JSG_height >= dsp.cbh).sel(JSG_height=slice(200,240))
                    rain_rate_below_cloud_df = rain_rate_below_cloud.to_pandas()
                    rain_rate_below_cloud_as_arrays = rain_rate_below_cloud_df.apply(lambda row: row.values, axis=1)

                if add_msi_vars:
                    # TODO rewrite for fetching MSI from MAAP
                    print('Adding MSI variables')
                    try:
                        selected_variables_MCOP = ['MCOP_effective_radius', 'MCOP_optical_thickness', 'MCOP_cloud_water_path']
                        for window_size in [256]:
                            dsp, MCOP = add_msi_cloud_params_to_acmcap(filename, dsp, lon_min, lon_max, lat_min, lat_max, window_size)
                            selected_variables_MCOP += [f'MCOP_cwp_skew_{window_size}', f'MCOP_cwp_inhomogeneity_{window_size}', f'MCOP_cloud_cover_{window_size}', 
                                                   f'MCOP_ks1_{window_size}', f'MCOP_ks2_{window_size}', f'MCOP_ks1_median_{window_size}', 
                                                   f'MCOP_ks2_median_{window_size}', f'MCOP_ks1_gt01_{window_size}', f'MCOP_ks2_gt01_{window_size}'
                                                   ]
                        selected_variables += selected_variables_MCOP

                    except Exception as e:
                        MCOP = None
                        print(f'Error adding MSI cloud parameters to ACMCAP: {e}')

                # Convert to dataframe
                df = dsp[selected_variables].to_dataframe()
                if add_lwc_profiles:
                    df['lwc_profile'] = lwc_as_arrays
                    df['rain_rate_profile'] = rain_rate_as_arrays
                    df['rain_rate_below_cloud_profile'] = rain_rate_below_cloud_as_arrays
                df['orbit'] = orbit

                if df_stratus is None:
                    df_stratus = df
                else:
                    df_stratus = pd.concat([df_stratus, df])

                # Create plots
                if plot_frames:
                    plot_path = f'{OUTPUT_PLOT}/fig_{orbit}_{region}.png'
                    #if not os.path.exists(plot_path):
                    dsp.encoding = encoding.copy()
                    if not add_msi_vars:
                        MCOP = None
                    make_overview_plot_v2(dsp=dsp, MCOP=MCOP, save_as=plot_path)

                print(f'{i + 1} of {number_of_orbits_to_process} files processed in region {region}\n')
                
                # Save periodically
                if (i % 10) < 1:
                    print('Saving dataframe to h5 file')
                    #df_stratus.to_csv(OUTPUT_FILE)
                    df_stratus.to_hdf(OUTPUT_FILE, key='StCu', mode='w')


            except Exception as e:
                print(f'Error processing file {filename}: {e}')
                # print line where error occurred
                import traceback
                print(traceback.print_exc())
                failed_files.append((filename, e))
                continue

        #df_stratus.to_csv(OUTPUT_FILE)
        if df_stratus is not None:
            df_stratus.to_hdf(OUTPUT_FILE, key='StCu', mode='w')

    df_stratus = df_stratus.reset_index()
    #df_stratus.to_csv(OUTPUT_FILE)
    df_stratus.to_hdf(OUTPUT_FILE, key='StCu', mode='w')

    print('Failed files:')
    for filename, e in failed_files:
        print(f'{filename}: {e}\n')


if __name__ == "__main__":
    main()
