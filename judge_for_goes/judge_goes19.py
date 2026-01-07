import xarray as xr
import numpy as np
import pandas as pd
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import timm
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/home/jmayer/ACI_research/Code/mesoscale_structure/judge_for_goes/')
from functions_for_goes import crop_goes_to_latlon_box, convert_radiance_to_brightness_temp

def setup_region_parameters(region='SEP'):
    """Setup region-specific parameters"""
    if region == 'SEP':
        # SEP region (Southeast Pacific)
        lat_bounds = (-25, -2.5)
        lon_bounds = (-95, -75)
        times = ['1400', '1600', '1800', '2000', '2200', '0000']
    elif region == 'NEP':
        # NEP region (Northeast Pacific)
        lat_bounds = (15, 35)
        lon_bounds = (-135, -115)
        times = ['1600', '1800', '2000', '2200', '0000']
    else:
        raise ValueError("Region must be 'SEP' or 'NEP'")
    
    return lat_bounds, lon_bounds, times


def preprocess_data(BT_11, BT_14, BT_15, COD, window_size=128, xy_indices=None, debugging=False):
    """Preprocess GOES data for JUDGE algorithm"""
    # Simple interpolation without complex coordinate handling
    scale_factor = 2  # 2x upscaling for COD2KMF; 4x upsampling is needed if CODF is used
    COD_interpolated = zoom(COD.values, scale_factor, order=1)
    # Create new xarray DataArray
    COD_interpolated = xr.DataArray(
        COD_interpolated,
        dims=['y', 'x']
    )

    scale_factor = 2
    latitudes = zoom(BT_11.latitude.values, scale_factor, order=1)
    longitudes = zoom(BT_11.longitude.values, scale_factor, order=1)
    BT_11 = zoom(BT_11.values, scale_factor, order=1)
    BT_14 = zoom(BT_14.values, scale_factor, order=1)
    BT_15 = zoom(BT_15.values, scale_factor, order=1)
    BT_11 = xr.DataArray(BT_11, dims=['y', 'x'])
    BT_14 = xr.DataArray(BT_14, dims=['y', 'x'])
    BT_15 = xr.DataArray(BT_15, dims=['y', 'x'])

    if xy_indices is None:
        # make a grid of indices, s.t. they are always 128 pixels apart
        y_indices = np.arange(64, BT_11.shape[0] - 64, 64)
        x_indices = np.arange(64, BT_11.shape[1] - 64, 64)
        # Create all coordinate combinations
        y_grid, x_grid = np.meshgrid(y_indices, x_indices, indexing='ij')
        xy_indices = list(zip(x_grid.flatten(), y_grid.flatten()))

    num_scenes = len(xy_indices)
    all_weirgb = []
    all_scenes_params = {'y_idx': [], 'x_idx': [], 'lon_mid': [], 'lat_mid': [], 'ice_flag': []}
    
    for i,j in xy_indices:
        lat_mid = latitudes[j, i]
        lon_mid = longitudes[j, i]

        try:
            # get window data
            data11 = BT_11[j - window_size // 2 : j + window_size // 2,
                            i - window_size // 2 : i + window_size // 2].values
            data22 = BT_14[j - window_size // 2 : j + window_size // 2,
                            i - window_size // 2 : i + window_size // 2].values
            data33 = BT_15[j - window_size // 2 : j + window_size // 2,
                            i - window_size // 2 : i + window_size // 2].values

            gama_R = 1
            gama_G = 2
            gama_B = 1
            R = data33 - data22  # Red
            G = data22 - data11  # Green
            B = data22  # Blue

            # Normalization y = (x - xmin) / (xmax - xmin)
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

            # Normalize COT y = (x - xmin) / (xmax - xmin)
            cot1 = COD_interpolated[j - window_size // 2 : j + window_size // 2,
                                    i - window_size // 2 : i + window_size // 2].values
            L_bond = np.min(cot1)
            H_bond = np.max(cot1)
            # Check for division by zero
            if L_bond == H_bond:
                cot2 = cot1.copy()  
            else:
                cot2 = (cot1 - L_bond) / (H_bond - L_bond)
            
            weirgb = np.zeros((4, 128, 128))

            weirgb[0, :, :] = R  # Red
            weirgb[1, :, :] = G  # Green
            weirgb[2, :, :] = B  # Blue
            weirgb[3, :, :] = cot2

            if debugging:
                plt.figure(figsize=(8, 3))
                plt.subplot(1, 2, 1)
                # Explicitly slice the RGB channels and document the expected shape
                rgb_image = weirgb[0:3, :, :]  # Shape should be (3, 128, 128)
                plt.imshow(rgb_image.transpose(1, 2, 0))  # Transpose to (128, 128, 3) for imshow
                plt.title('RGB Image')
                plt.subplot(1, 2, 2)
                plt.imshow(weirgb[3], cmap='gray')
                plt.title('COT Channel')
                plt.show()

            ice_pixel = (data22 < 273)
            ice_flag = np.sum(ice_pixel) / (window_size * window_size)

        except Exception as e:
            print(f"Error processing window at x:{i}, y:{j} - {e}")
            ice_flag = np.nan
            weirgb = np.nan * np.empty((4, 128, 128))

        all_scenes_params['y_idx'].append(j)
        all_scenes_params['x_idx'].append(i)
        all_scenes_params['lon_mid'].append(lon_mid)
        all_scenes_params['lat_mid'].append(lat_mid)
        all_scenes_params['ice_flag'].append(ice_flag)

        all_weirgb.append(weirgb)

    all_weirgb = np.array(all_weirgb)

    return all_weirgb, all_scenes_params


class JudgeDataset(Dataset):
    """Dataset class for JUDGE algorithm"""
    def __init__(self, BT11, BT14, BT15, COD, debugging=False):
        self.data = []
        mat_data, all_scenes_params = preprocess_data(BT11, BT14, BT15, COD, window_size=128, debugging=False)
        print(mat_data.shape)
        if mat_data is not None:
            data = torch.from_numpy(mat_data)
            self.data.extend((data))
            self.scenes_params = all_scenes_params
        else:
            print("no data found!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data


def judge(judge_loader, model, class_names, device, save_to=None):
    """Run JUDGE classification on data"""
    model.eval()
    all_cats = []
    all_certs = []

    with torch.no_grad():
        for i, input in enumerate(judge_loader, start=1):
            # Debug shapes
            print(f"Input type: {type(input)}")
            print(f"Input shape: {input.shape if hasattr(input, 'shape') else 'No shape attr'}")
            
            # Handle the tensor properly
            if isinstance(input, (list, tuple)):
                input = input[0]  # Extract the tensor from list/tuple
            
            input = input.to(torch.float32).to(device)
            
            # Ensure 4D: (batch, channels, height, width)
            if input.dim() == 3:
                input = input.unsqueeze(0)  # Add batch dimension
                
            print(f"Final input shape: {input.shape}")
            output = model(input)
            y_pred = torch.softmax(output, dim=1).cpu()
            cert = torch.max(y_pred, dim=1).values.cpu() / torch.sum(y_pred, dim=1).cpu() 
            cat = torch.argmax(y_pred, dim=1).cpu().numpy()  
            print(cat, cert)
            all_cats.extend(cat)
            all_certs.extend(cert)
    
    all_cats = np.array(all_cats)
    all_certs = np.array(all_certs)
    print(all_cats.shape)
    return all_cats, all_certs


def judge_MSI(BT11, BT14, BT15, COD, model_path, device_num='1', debugging=False):
    """Main JUDGE function for MSI classification"""
    class_names = ['Solid stratus', 'Closed MCC', 'Open MCC', 'Disorganized MCC', 'Clustered Cu', 'Suppressed Cu']
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = torch.device('cuda:' + device_num if torch.cuda.is_available() else 'cpu')

    test_dataset = JudgeDataset(BT11, BT14, BT15, COD, debugging=debugging)
    test_len = JudgeDataset.__len__(test_dataset)
    scenes_params = test_dataset.scenes_params
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)  
    
    model = timm.create_model('resnet50d', pretrained=False)
    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(2048, 6)
    model = model.to(device)
    weights = torch.load(model_path, map_location=device)
    model.load_state_dict(weights)
    model = model.to(device)
    
    cats, cert = judge(test_dataloader, model, class_names, device)

    del test_dataset
    del test_dataloader
    return cats, cert, scenes_params


def process_goes_data(region='SEP', data_dir='/home/jmayer/Object_Data/GOES19/',
                     model_path='/home/jmayer/ACI_research/Code/mesoscale_structure/judge_original/64epochs_accuracy0.91456.pth',
                     save_path='/home/jmayer/ACI_research/Code/resources/',
                     plot_path=None, days=None, device_num='1'):
    """Main processing function"""
    
    # Setup region parameters
    lat_bounds, lon_bounds, times = setup_region_parameters(region)
    
    # Setup file paths
    partial_results_filename = os.path.join(save_path, f'judge_results_GOES_{region}_partial.csv')
    results_filename = os.path.join(save_path, f'judge_results_GOES_{region}.csv')
    
    if plot_path is None:
        plot_path = f'/home/jmayer/ACI_research/Code/causal_aci/figures_judge_goes_{region}/'
    os.makedirs(plot_path, exist_ok=True)
    
    # Load existing results or create new DataFrame
    if os.path.exists(partial_results_filename):
        df_results = pd.read_csv(partial_results_filename)
        df_results['datetime'] = pd.to_datetime(df_results['datetime'])
    else:
        df_results = pd.DataFrame(columns=['datetime', 'y_idx', 'x_idx', 'lon_mid', 'lat_mid', 'ice_flag', 'cats', 'cert'])

    # Default to all days in August 2025 if not specified
    if days is None:
        days = [f'{i:02d}' for i in range(1, 32)]

    # Process each day and time
    for day in days:
        date = f'202508{day}'
        for time in times:
            ts = pd.to_datetime(date + time)
            if ts in df_results['datetime'].values:
                print(f'{date}{time} already contained in judge results for GOES. Skipping this datetime...')
                continue
            
            print('.' * 85)
            print(f"Processing date {date} time {time}...")
            
            try:
                # Load and crop GOES data
                print("Loading C11 data...")
                with xr.open_dataset(f'{data_dir}/goes19_C11_RadF_{date}_{time}.nc') as ds:
                    ds = convert_radiance_to_brightness_temp(ds)
                    ds_11_cropped = crop_goes_to_latlon_box(ds, lat_bounds, lon_bounds)

                print("Loading C14 data...")
                with xr.open_dataset(f'{data_dir}/goes19_C14_RadF_{date}_{time}.nc') as ds:
                    ds = convert_radiance_to_brightness_temp(ds)
                    ds_14_cropped = crop_goes_to_latlon_box(ds, lat_bounds, lon_bounds)

                print("Loading C15 data...")
                with xr.open_dataset(f'{data_dir}/goes19_C15_RadF_{date}_{time}.nc') as ds:
                    ds = convert_radiance_to_brightness_temp(ds)
                    ds_15_cropped = crop_goes_to_latlon_box(ds, lat_bounds, lon_bounds)

                print("Loading COD data...")
                with xr.open_dataset(f'{data_dir}/goes19_COD2KMF_{date}_{time}.nc') as ds:
                    ds_COD_cropped = crop_goes_to_latlon_box(ds, lat_bounds, lon_bounds)

                print('.' * 85)
                print('Running JUDGE...')
                cats, cert, scenes_params = judge_MSI(
                    ds_11_cropped['BT'], ds_14_cropped['BT'], ds_15_cropped['BT'], 
                    ds_COD_cropped['COD'], model_path, device_num
                )

                # Save results to dataframe
                df = pd.DataFrame(scenes_params)
                df['cats'] = cats
                df['cert'] = cert
                df['datetime'] = pd.to_datetime(date + time)
                df_results = pd.concat([df_results, df], ignore_index=True)
                df_results.to_csv(partial_results_filename, index=False)

                # Visualize results
                palette_goes_cats = {0: 'purple', 1: 'steelblue', 2: 'green', 3: 'orange', 4: 'red', 5: 'hotpink', np.nan: 'grey'}
                fig, axs = plt.subplots(1, 2, figsize=(21, 10))
                COD_interpolated = zoom(ds_COD_cropped['COD'].values, 2, order=1)
                axs[0].imshow(COD_interpolated, vmax=35, cmap='Greys_r')
                BT_14_interpolated = zoom(ds_14_cropped['BT'].values, 2, order=1)
                axs[1].imshow(BT_14_interpolated, vmax=290, vmin=280, cmap='Greys')
                for ax in axs:
                    sns.scatterplot(ax=ax, x=scenes_params['x_idx'], y=scenes_params['y_idx'], 
                                  hue=cats, size=12, lw=0, palette=palette_goes_cats)
                plt.suptitle(f'{date} {time} UTC, GOES-19 MSI Judgement Results', fontsize=16)
                plt.savefig(os.path.join(plot_path, f'judge_{date}{time}.png'))
                plt.close()
                
            except Exception as e:
                print(f"Error processing {date} {time}: {e}")
                continue

    # Save final results
    df_results.to_csv(results_filename, index=False)
    print(f"Processing complete. Results saved to {results_filename}")
    
    return df_results


def main():
    """Main function to run the script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process GOES-19 data with JUDGE algorithm')
    parser.add_argument('--region', choices=['SEP', 'NEP'], default='SEP', help='Region to process')
    parser.add_argument('--data_dir', default='/home/jmayer/Object_Data/GOES19/', help='Directory containing GOES data')
    parser.add_argument('--model_path', default='/home/jmayer/ACI_research/Code/mesoscale_structure/judge_original/64epochs_accuracy0.91456.pth', help='Path to JUDGE model')
    parser.add_argument('--save_path', default='/home/jmayer/ACI_research/Code/resources/', help='Directory to save results')
    parser.add_argument('--device_num', default='1', help='GPU device number')
    parser.add_argument('--days', nargs='+', help='Specific days to process (e.g., 01 02 03)')
    
    args = parser.parse_args()
    
    df_results = process_goes_data(
        region=args.region,
        data_dir=args.data_dir,
        model_path=args.model_path,
        save_path=args.save_path,
        device_num=args.device_num,
        days=args.days
    )
    
    print(f"Processed {len(df_results)} scenes")


if __name__ == "__main__":
    main()
    