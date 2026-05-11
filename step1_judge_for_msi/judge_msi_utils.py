import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.stats
import xarray as xr
from typing import Tuple

def get_window_data(MCOP, MRGR, along_track_idx, across_track_nadir_idx=284, window_size=128):
    along_track_isel_slice = slice(along_track_idx - window_size//2, along_track_idx + window_size//2)
    # check if nadir + window size is larger than across_track dimension
    across_track_nadir_idx_1km = across_track_nadir_idx // 2  # Adjust for 1km resolution
    if across_track_nadir_idx_1km + window_size//2 > MCOP.dims['across_track']:
        across_track_sel_slice = slice(MCOP.dims['across_track'] - window_size - 1, MCOP.dims['across_track'] - 1)
        across_track_idx = MCOP.dims['across_track'] - window_size//2 - 1
    else:
        across_track_sel_slice = slice(across_track_nadir_idx_1km - window_size//2, across_track_nadir_idx_1km + window_size//2)
        across_track_idx = across_track_nadir_idx_1km

    COT_window = MCOP.cloud_optical_thickness.isel(along_track=along_track_isel_slice).sel(across_track=across_track_sel_slice)
    TIR1_window = MRGR.TIR1.isel(along_track=along_track_isel_slice).sel(across_track=across_track_sel_slice)
    TIR2_window = MRGR.TIR2.isel(along_track=along_track_isel_slice).sel(across_track=across_track_sel_slice)
    TIR3_window = MRGR.TIR3.isel(along_track=along_track_isel_slice).sel(across_track=across_track_sel_slice)

    def check_shape(data, window_size):
        if data.shape != (window_size, window_size):
            raise ValueError(f"Data shape {data.shape} does not match expected shape ({window_size}, {window_size})")
    check_shape(COT_window, window_size)
    check_shape(TIR1_window, window_size)
    check_shape(TIR2_window, window_size)
    check_shape(TIR3_window, window_size)

    # get scene midpoint and edge parameters
    values_at_scene_midpoint = MCOP.isel(along_track=along_track_idx, across_track=across_track_idx)
    lat_at_nadir = MCOP.latitude.sel(across_track=across_track_nadir_idx_1km)
    lat_descending = np.all(np.diff(lat_at_nadir) < 0)
    along_track_idx_left_of_window = max(0, along_track_idx - window_size//2 - 1)
    along_track_idx_left_within_window = along_track_idx - window_size//2
    along_track_idx_right_within_window = along_track_idx + window_size//2 - 1
    along_track_idx_right_of_window = min(along_track_idx + window_size//2, lat_at_nadir.shape[0] - 1)
    if lat_descending:
        # need to add half a latitude step to both sides to account for the the fact that the data is coarsened to 1km resolution
        lat_min_1 = lat_at_nadir.isel(along_track=along_track_idx_right_within_window).values
        lat_min_2 = lat_at_nadir.isel(along_track=along_track_idx_right_of_window).values
        lat_min = float((lat_min_1 + lat_min_2) / 2)
        lat_max_1 = lat_at_nadir.isel(along_track=along_track_idx_left_of_window).values
        lat_max_2 = lat_at_nadir.isel(along_track=along_track_idx_left_within_window).values
        lat_max = float((lat_max_1 + lat_max_2) / 2)
    else:
        lat_min_1 = lat_at_nadir.isel(along_track=along_track_idx_left_of_window).values
        lat_min_2 = lat_at_nadir.isel(along_track=along_track_idx_left_within_window).values
        lat_min = float((lat_min_1 + lat_min_2) / 2)
        lat_max_1 = lat_at_nadir.isel(along_track=along_track_idx_right_within_window).values
        lat_max_2 = lat_at_nadir.isel(along_track=along_track_idx_right_of_window).values
        lat_max = float((lat_max_1 + lat_max_2) / 2)

    # get CWP statistics
    CWP_window = MCOP.cloud_water_path.isel(along_track=along_track_isel_slice).sel(across_track=across_track_sel_slice)
    CWP_window_clean = CWP_window.where(np.isfinite(CWP_window))
    cwp_skew = scipy.stats.skew(CWP_window_clean.values.flatten(), bias=False, nan_policy='omit')
    cwp_inhomogeneity = np.nanstd(CWP_window_clean.values.flatten()) / np.nanmean(CWP_window_clean.values.flatten())
    cloud_cover = float(xr.where(CWP_window_clean > 0, 1, 0).mean().values)
    if 'land_flag' in MRGR.variables:
        land_flag_window = MRGR.land_flag.isel(along_track=along_track_isel_slice).sel(across_track=across_track_sel_slice)
        land_fraction = float(xr.where(land_flag_window > 0, 1, 0).mean().values)
        if land_fraction > 0.05:
            ocean_flag = 0
        else:
            ocean_flag = 1
    else:
        ocean_flag = -1  # land_flag not available

    scene_params = {'along_track_mid': float(values_at_scene_midpoint['along_track_values']),
                    'time_mid': values_at_scene_midpoint['time'].values,
                    'lon_mid': float(values_at_scene_midpoint['longitude']),
                    'lat_mid': float(values_at_scene_midpoint['latitude']),
                    'lat_min': lat_min,
                    'lat_max': lat_max,
                    'cwp_skew': cwp_skew,
                    'cwp_inhomogeneity': cwp_inhomogeneity,
                    'cloud_cover': cloud_cover,
                    'ocean_flag': ocean_flag}
    return COT_window, TIR1_window, TIR2_window, TIR3_window, scene_params

def clean_data(MCOP, MRGR):
    # at the borders of MSI frames are some NaN values
    # drop the across_track positions where all TIR1 values are NaN; use the same positions to drop in MCOP
    mrgr_valid_pixels_across = ~(
        MRGR.TIR1.isnull().all(dim='along_track') |
        MRGR.TIR2.isnull().all(dim='along_track') |
        MRGR.TIR3.isnull().all(dim='along_track')
    )
    # Apply the mask to both datasets
    MRGR = MRGR.where(mrgr_valid_pixels_across, drop=True)
    MCOP = MCOP.where(mrgr_valid_pixels_across, drop=True)

    # because the cropping to latitude/longitude values is not along fixed along_track values, but a bit diagonal, there can be some NaN values at low or high along_track positions
    mrgr_valid_pixels_along = ~(
        MRGR.TIR1.isnull().any(dim='across_track') |
        MRGR.TIR2.isnull().any(dim='across_track') |
        MRGR.TIR3.isnull().any(dim='across_track')
    )
    # Apply the mask to both datasets
    MRGR = MRGR.where(mrgr_valid_pixels_along, drop=True)
    MCOP = MCOP.where(mrgr_valid_pixels_along, drop=True)

    # check the shapes
    if not MCOP.cloud_optical_thickness.shape == MRGR.TIR1.shape:
        raise ValueError("Shapes of MCOP and MRGR do not match after dropping NaN values in MRGR.")
    if not MRGR.TIR1.shape[1] == 357:
        raise ValueError("MRGR.TIR1.shape[1] is not 357.")
    if not MRGR.TIR1.shape[0] > 128:
        raise ValueError("MRGR.TIR1.shape[0] is not greater than 128.")
    return MCOP, MRGR

def align_mcop_mnom_datasets(MCOP, MNOM, across_track_idx=284):
    """
    Align MCOP and MNOM datasets by finding best matching latitudes
    along a specific across_track line
    """
    # Get 1D latitude arrays along the selected across_track line
    mcop_lat_line = MCOP.latitude.isel(across_track=across_track_idx).values
    mnom_lat_line = MNOM.latitude.isel(across_track=across_track_idx).values
    
    # Remove NaN values
    mcop_valid_mask = ~np.isnan(mcop_lat_line)
    mnom_valid_mask = ~np.isnan(mnom_lat_line)
    
    mcop_valid_indices = np.where(mcop_valid_mask)[0]
    mnom_valid_indices = np.where(mnom_valid_mask)[0]
    
    mcop_lat_clean = mcop_lat_line[mcop_valid_mask]
    mnom_lat_clean = mnom_lat_line[mnom_valid_mask]
    
    print(f"MCOP latitude range: {mcop_lat_clean.min():.3f} to {mcop_lat_clean.max():.3f}")
    print(f"MNOM latitude range: {mnom_lat_clean.min():.3f} to {mnom_lat_clean.max():.3f}")
    
    # Find overlapping latitude range
    common_lat_min = max(mcop_lat_clean.min(), mnom_lat_clean.min())
    common_lat_max = min(mcop_lat_clean.max(), mnom_lat_clean.max())
    
    print(f"Common latitude range: {common_lat_min:.3f} to {common_lat_max:.3f}")
    
    # Find the smaller dataset size to match
    min_size = min(len(mcop_lat_clean), len(mnom_lat_clean))
    
    # Method 1: Simple crop to same size (take first N elements that are in common range)
    mcop_in_range = mcop_valid_indices[(mcop_lat_clean >= common_lat_min) & 
                                       (mcop_lat_clean <= common_lat_max)]
    mnom_in_range = mnom_valid_indices[(mnom_lat_clean >= common_lat_min) & 
                                       (mnom_lat_clean <= common_lat_max)]
    
    # Take the minimum length to ensure same dimensions
    final_length = min(len(mcop_in_range), len(mnom_in_range))
    
    mcop_final_indices = mcop_in_range[:final_length]
    mnom_final_indices = mnom_in_range[:final_length]
    
    # Crop datasets
    MCOP_aligned = MCOP.isel(along_track=mcop_final_indices)
    MNOM_aligned = MNOM.isel(along_track=mnom_final_indices)
    
    print(f"Original MCOP shape: along_track={MCOP.dims['along_track']}, across_track={MCOP.dims['across_track']}")
    print(f"Original MNOM shape: along_track={MNOM.dims['along_track']}, across_track={MNOM.dims['across_track']}")
    print(f"Aligned MCOP shape: along_track={MCOP_aligned.dims['along_track']}, across_track={MCOP_aligned.dims['across_track']}")
    print(f"Aligned MNOM shape: along_track={MNOM_aligned.dims['along_track']}, across_track={MNOM_aligned.dims['across_track']}")
    
    return MCOP_aligned, MNOM_aligned


def coarsen_to_1km(MCOP, MRGR):
    MCOP_1km = MCOP.coarsen(along_track=2, across_track=2, boundary='trim').mean()
    MRGR_1km = MRGR.coarsen(along_track=2, across_track=2, boundary='trim').mean()
    return MCOP_1km, MRGR_1km

def preprocess_data(MCOP, MRGR, window_size=128, debugging=False):
    # make sure both datasets are aligned in terms of latitude and have the same along_track dimension
    MCOP, MRGR = align_mcop_mnom_datasets(MCOP, MRGR, across_track_idx=284)
    MCOP = MCOP.assign(along_track_values=MRGR['along_track'])
    MCOP = MCOP[['cloud_optical_thickness','cloud_water_path','latitude','longitude','time','along_track_values']]
    MCOP['cloud_optical_thickness'] = MCOP['cloud_optical_thickness'].fillna(0) # Fill NaN values in COT with 0 s.t. coarsening works correctly
    MRGR = MRGR[['TIR1', 'TIR2', 'TIR3','land_flag']] # MRGR data already has no NaN values
    MCOP, MRGR = clean_data(MCOP, MRGR)
    # Coarsen data to 1km resolution
    MCOP, MRGR = coarsen_to_1km(MCOP, MRGR)

    # Determine along_track indices for scene extraction
    along_track_scene_idxs = np.arange(window_size//2, MCOP.dims['along_track'] - window_size//2 + 1, window_size//2)

    num_scenes = along_track_scene_idxs.shape[0]
    all_weirgb = []
    all_scenes_params = {'along_track_mid': [], 'time_mid': [], 'lon_mid': [], 'lat_mid': [], 'lat_min': [], 'lat_max': [], 'ice_flag': [], 
                         'cwp_skew': [], 'cwp_inhomogeneity': [], 'cloud_cover': [], 'ocean_flag': []}

    for i in range(num_scenes):

        COT_window, TIR1_window, TIR2_window, TIR3_window, scene_params = get_window_data(MCOP, 
                                                                            MRGR, 
                                                                            along_track_idx=along_track_scene_idxs[i], 
                                                                            across_track_nadir_idx=284, 
                                                                            window_size=128)
        # Convert to numpy arrays
        data11 = TIR1_window.values
        data22 = TIR2_window.values
        data33 = TIR3_window.values

        gama_R = 1
        gama_G = 2
        gama_B = 1
        R = data33-data22 #Red
        G = data22-data11 #Green
        B = data22 #Blue

        #归一化 y = (x - xmin) / (xmax - xmin)
        L_bond = np.min(R)
        H_bond = np.max(R)
        R = np.clip(R, L_bond, H_bond)
        R = (R - L_bond) / (H_bond - L_bond)
        R = np.power(R, 1 / gama_R)
        
        L_bond = np.min(G)
        H_bond = np.max(G)
        G = np.clip(G, L_bond, H_bond)
        G = (G - L_bond) / (H_bond - L_bond)
        G = np.power(G, 1 / gama_G)

        L_bond = np.min(B)
        H_bond = np.max(B)
        B = np.clip(B, L_bond, H_bond)
        B = (B - L_bond) / (H_bond - L_bond)
        B = np.power(B, 1 / gama_B)

        #归一化cot  y = (x - xmin) / (xmax - xmin)
        cot1 = COT_window.values
        L_bond = np.min(cot1)
        H_bond = np.max(cot1)
        # 检查是否存在除以零的情况
        if L_bond == H_bond:
            cot2 = cot1.copy()  
        else:
            cot2 = (cot1 - L_bond) / (H_bond - L_bond)
        
        weirgb = np.zeros((4, 128, 128))

        weirgb[0, :, :] = R  #Red
        weirgb[1, :, :] = G  #Green
        weirgb[2, :, :] = B  #Blue
        weirgb[3, :, :] = cot2

        if debugging:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 3))
            plt.subplot(1, 2, 1)
            # Explicitly slice the RGB channels and document the expected shape
            rgb_image = weirgb[0:3, :, :]  # Shape should be (3, 128, 128)
            plt.imshow(rgb_image.transpose(1, 2, 0))  # Transpose to (128, 128, 3) for imshow
            plt.title('RGB Image')
            plt.subplot(1, 2, 2)
            plt.imshow(weirgb[3], cmap='gray')
            plt.title('COT Channel')
            plt.suptitle('along_track index: {}'.format(scene_params['along_track_mid']))
            plt.show()

        all_weirgb.append(weirgb)
        all_scenes_params['along_track_mid'].append(scene_params['along_track_mid'])
        all_scenes_params['time_mid'].append(scene_params['time_mid'])
        all_scenes_params['lon_mid'].append(scene_params['lon_mid'])
        all_scenes_params['lat_mid'].append(scene_params['lat_mid'])
        all_scenes_params['lat_min'].append(scene_params['lat_min'])
        all_scenes_params['lat_max'].append(scene_params['lat_max'])
        all_scenes_params['cwp_skew'].append(scene_params['cwp_skew'])
        all_scenes_params['cwp_inhomogeneity'].append(scene_params['cwp_inhomogeneity'])
        all_scenes_params['cloud_cover'].append(scene_params['cloud_cover'])
        all_scenes_params['ocean_flag'].append(scene_params['ocean_flag'])

        ice_pixel = (TIR2_window.values < 273)
        ice_flag = np.sum(ice_pixel) / (window_size * window_size)
        all_scenes_params['ice_flag'].append(ice_flag)

    all_weirgb = np.array(all_weirgb)

    return all_weirgb, num_scenes, all_scenes_params


class JudgeDataset(Dataset):
    def __init__(self, MCOP, MRGR, debugging=False):
        self.data = []
        mat_data,num_layers,all_scenes_params = preprocess_data(MCOP, MRGR, debugging=debugging)
        print(mat_data.shape,num_layers)
        if mat_data is not None:
            data = torch.from_numpy(mat_data)
            self.data.extend((data))
            self.scene_params = all_scenes_params
        else:
            print("no data found!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data
    

