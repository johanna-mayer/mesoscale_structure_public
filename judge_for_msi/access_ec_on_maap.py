"""
Functions are adapted from https://docs.maap-project.org/en/develop/science/EarthCARE/EarthCARE_access_and_visualize.html
"""

from pathlib import Path
from pystac_client import Client
import fsspec
import xarray as xr
import numpy as np
import requests
import sys
sys.path.append("/home/jmayer/ACI_research/ectools/")
from ectools import ecio

catalog_url = 'https://catalog.maap.eo.esa.int/catalogue/'
catalog = Client.open(catalog_url)

# Path to credentials file
CREDENTIALS_FILE = Path('/home/jmayer/ACI_research/Code/mesoscale_structure/judge_for_msi/credentials_maap.txt')

io_params = {
    "fsspec_params": {
        "cache_type": "blockcache",
        "block_size": 8 * 1024 * 1024
    },
    "h5py_params": {
        "driver_kwds": {
            "rdcc_nbytes": 8 * 1024 * 1024
        }
    }
}

bbox_regions = {
    'global': {'lat': (-90, 90), 'lon': (-180, 180)},
    'SEA': {'lat': (-22.5, -2.5), 'lon': (-5, 15)},    # South East Atlantic
    'NEA': {'lat': (10, 30), 'lon': (-40, -20)},   # North East Atlantic
    'SEP': {'lat': (-22.5, -2.5), 'lon': (-95, -75)},  # South East Pacific
    'NEP': {'lat': (15, 35), 'lon': (-135, -115)}, # North East Pacific
}

def load_credentials(file_path=CREDENTIALS_FILE):
    """Read key-value pairs from a credentials file into a dictionary."""
    creds = {}
    if not file_path.exists():
        raise FileNotFoundError(f"Credentials file not found: {file_path}")
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            creds[key.strip()] = value.strip()
    return creds


# --- ESA MAAP API ---

def get_token():
    """Use OFFLINE_TOKEN to fetch a short-lived access token."""
    creds = load_credentials()

    OFFLINE_TOKEN = creds.get("OFFLINE_TOKEN")
    CLIENT_ID = creds.get("CLIENT_ID")
    CLIENT_SECRET = creds.get("CLIENT_SECRET")
    print(CLIENT_SECRET)

    if not all([OFFLINE_TOKEN, CLIENT_ID, CLIENT_SECRET]):
        raise ValueError("Missing OFFLINE_TOKEN, CLIENT_ID, or CLIENT_SECRET in credentials file")

    url = "https://iam.maap.eo.esa.int/realms/esa-maap/protocol/openid-connect/token"
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": OFFLINE_TOKEN,
        "scope": "offline_access openid"
    }

    response = requests.post(url, data=data)
    response.raise_for_status()

    response_json = response.json()
    access_token = response_json.get('access_token')

    if not access_token:
        raise RuntimeError("Failed to retrieve access token from IAM response")

    return access_token


def region_to_bbox(region):
    """Convert region name to bounding box coordinates."""
    lat_min, lat_max = bbox_regions[region]['lat']
    lon_min, lon_max = bbox_regions[region]['lon']
    return [lon_min, lat_min, lon_max, lat_max]


def crop_by_lon_or_lat(ds_EC, lon_or_lat_var, l_min, l_max, sel_across_track_idx=None):
    if sel_across_track_idx is not None:
        ds = ds_EC.sel(across_track=sel_across_track_idx)
    else:
        ds = ds_EC

    if np.isfinite(ds[lon_or_lat_var].values[0]) and np.isfinite(ds[lon_or_lat_var].values[-1]):
        lonlat_descending = ds[lon_or_lat_var].values[0] > ds[lon_or_lat_var].values[-1]
    else:
        raise ValueError(f"Cannot determine whether {lon_or_lat_var} is ascending or descending due to NaN values at the edges.")
    
    if lonlat_descending:
        l_start = l_max
        l_stop = l_min
    else:
        l_start = l_min
        l_stop = l_max

    i_start = np.nanargmin(np.abs(ds[lon_or_lat_var].values - l_start))
    i_stop = np.nanargmin(np.abs(ds[lon_or_lat_var].values - l_stop))

    print(f"\t Crop by {lon_or_lat_var}: Selecting frame from {i_start} ({lon_or_lat_var}={ds[lon_or_lat_var].values[i_start]:.2f}) to {i_stop} ({lon_or_lat_var}={ds[lon_or_lat_var].values[i_stop]:.2f})")
    ds_cropped = ds_EC.isel({'along_track':slice(i_start, i_stop)})

    return ds_cropped


def crop_to_stcu_region(ds_EC, region, sel_across_track_idx=None):
    lat_min, lat_max = bbox_regions[region]['lat']
    lon_min, lon_max = bbox_regions[region]['lon']
    
    tmp = crop_by_lon_or_lat(ds_EC=ds_EC, lon_or_lat_var='latitude', l_min=lat_min, l_max=lat_max, sel_across_track_idx=sel_across_track_idx)
    ds_cropped = crop_by_lon_or_lat(ds_EC=tmp, lon_or_lat_var='longitude', l_min=lon_min, l_max=lon_max, sel_across_track_idx=sel_across_track_idx)
    
    return ds_cropped


def get_ec_data_url_via_maap(datetime_range, 
                             product, 
                             region=None,
                             frame=None,
                             orbitNumber=None,
                             productVersion = 'ba',
                             ec_collection=['EarthCAREL2Validated_MAAP'],
                             max_items=5):

    if datetime_range is None:
        datetime_today = np.datetime64('now', 'D')
        datetime_range = ['2024-07-01T00:00:00Z', str(datetime_today) + 'T23:59:00Z']
    filter = f"productType = '{product}'"
    if frame is not None:
        filter += f" and frame = '{frame}'"
    if orbitNumber is not None:
        filter += f" and orbitNumber = {orbitNumber}"
    if productVersion is not None:
        filter += f" and productVersion = '{productVersion}'"
    if region is None:
        region = 'global'
    if orbitNumber is not None and product == 'MSI_NOM_1B':
        region = 'global'  # override region to global for specific orbitNumber queries for MSI_NOM_1B; there are some problems with the bbox for this product

    search = catalog.search(
        collections=ec_collection,
        filter = filter,
        bbox = region_to_bbox(region),
        datetime = datetime_range,
        method = 'GET',
        max_items=max_items  # Adjust as needed, given the large amount of products it is recommended to set a limit if especially if you display results in pandas dataframe or similiar
    )

    items = list(search.items())
    print(f"Accessing {len(items)} items (limited by max_items).")
    print(f"{search.matched()} items found that matched the query.")
    if len(items) == 0:
        raise FileNotFoundError("No items found for the given query parameters.")
    if len(items) < search.matched():
        print("Warning: The number of retrieved items is less than the total matched items. Consider increasing max_items.")

    enclosures = ["enclosure_h5", "enclosure_hdr"]
    ds_url_list = []
    for item in items:
        ds_url = None
        try:
            for enc in enclosures:
                asset = getattr(item, "assets", {}).get(enc)
                if asset and getattr(asset, "href", None) and asset.href.lower().endswith(".h5"):
                    ds_url = asset.href
                    print(f"Found .h5 file in {enc}: {ds_url}")
                    #break #TODO uncomment if only first found is needed
        except Exception as e:
            print(f"Error while searching for .h5 file: {e}")
            raise FileNotFoundError("No .h5 file found via enclosure_h5 or enclosure_hdr.")
        print("HDF5 URL:", ds_url)
        orbit = item.properties['title'][-6:]
        print("Orbit:", orbit)
        ds_url_list.append((ds_url, orbit))

    return ds_url_list


def get_mcop_via_maap(datetime_range=None, 
                        region=None, 
                        frame=None,
                        orbitNumber=None,
                        ec_collection = ['EarthCAREL2Validated_MAAP'],
                        mcop_variables=['time', 'latitude', 'longitude', 'cloud_water_path', 'cloud_optical_thickness'],
                        max_items=1,
                        trim=True
                        ):
    """Function to access EarthCARE MSI data via MAAP."""

    ds_url_list = get_ec_data_url_via_maap(datetime_range, 'MSI_COP_2A', region, frame, orbitNumber=orbitNumber, ec_collection=ec_collection, max_items=max_items)

    print("Fetching ESA MAAP token...")
    ESA_TOKEN = get_token()
    fs = fsspec.filesystem(
        "https",
        headers={"Authorization": f"Bearer {ESA_TOKEN}"},
        **io_params["fsspec_params"]  )

    MCOP_list = []
    for ds_url, orbit in ds_url_list:
        print("Dataset URL:", ds_url)
        print("Orbit:", orbit)
        # Open the file and read it into an xarray Dataset
        with fs.open(ds_url, "rb") as f:
            with xr.open_dataset(f,
                                engine="h5netcdf",
                                **io_params["h5py_params"],
                                group="ScienceData") as MCOP:
                # Select only the requested variables
                MCOP = MCOP[mcop_variables]
                MCOP.encoding = {'source': ds_url}  # Keep track of source URL
                if trim:
                    idx_nadir = 284
                    MCOP = ecio.trim_to_frame(MCOP, latvar='latitude', sel_nadir={'across_track':idx_nadir})
                if region is not None:
                    print(f"Cropping to region {region}... lat_min, lat_max = {bbox_regions[region]['lat']}, lon_min, lon_max = {bbox_regions[region]['lon']}")
                    print(f"\tbefore cropping: lat_min, lat_max = {MCOP.latitude.min().values}, {MCOP.latitude.max().values}, lon_min, lon_max = {MCOP.longitude.min().values}, {MCOP.longitude.max().values}")
                    MCOP = crop_to_stcu_region(MCOP, region, sel_across_track_idx=284)
                    print(f" \tafter cropping: lat_min, lat_max = {MCOP.latitude.min().values}, {MCOP.latitude.max().values}, lon_min, lon_max = {MCOP.longitude.min().values}, {MCOP.longitude.max().values}")
                if MCOP is None:
                    continue
                MCOP = MCOP.load() # Load data into memory
        MCOP_list.append((MCOP, orbit))
    return MCOP_list


def get_mrgr_via_maap(datetime_range=None, 
                        region=None,
                        frame=None,
                        orbitNumber=None,
                        ec_collection = ['EarthCAREL1Validated_MAAP'],
                        mrgr_variables=['time', 'latitude', 'longitude', 'TIR1', 'TIR2', 'TIR3'],
                        max_items=1,
                        trim=True
                        ):
    """Function to access EarthCARE MSI data via MAAP."""

    ds_url_list = get_ec_data_url_via_maap(datetime_range, 'MSI_RGR_1C', region, frame, orbitNumber=orbitNumber, ec_collection=ec_collection, max_items=max_items)

    print("Fetching ESA MAAP token...")
    ESA_TOKEN = get_token()
    fs = fsspec.filesystem(
        "https",
        headers={"Authorization": f"Bearer {ESA_TOKEN}"},
        **io_params["fsspec_params"]  )

    MRGR_list = []
    for ds_url, orbit in ds_url_list:
        print("Dataset URL:", ds_url)
        print("Orbit:", orbit)
        with fs.open(ds_url, "rb") as f:
            with xr.open_dataset(f,
                                engine="h5netcdf",
                                **io_params["h5py_params"],
                                group="ScienceData") as MRGR:
            
                if 'TIR1' in mrgr_variables:
                    # For MSI_RGR_1C, pixel_values has an extra band dimension
                    MRGR['TIR1'] = MRGR.pixel_values.sel(band='TIR1').where(MRGR.pixel_quality_status.sel(band='TIR1') == 0)
                if 'TIR2' in mrgr_variables:
                    MRGR['TIR2'] = MRGR.pixel_values.sel(band='TIR2').where(MRGR.pixel_quality_status.sel(band='TIR2') == 0)
                if 'TIR3' in mrgr_variables:
                    MRGR['TIR3'] = MRGR.pixel_values.sel(band='TIR3').where(MRGR.pixel_quality_status.sel(band='TIR3') == 0)
                # Select only the requested variables
                MRGR = MRGR[mrgr_variables]
                MRGR.encoding = {'source': ds_url}  # Keep track of source URL
                if trim:
                    idx_nadir = 284
                    MRGR = ecio.trim_to_frame(MRGR, latvar='latitude', sel_nadir={'across_track':idx_nadir})
                if region is not None:
                    MRGR = crop_to_stcu_region(MRGR, region, sel_across_track_idx=284)
                if MRGR is None:
                    continue
                MRGR = MRGR.load() # Load data into memory
        MRGR_list.append((MRGR, orbit))
    return MRGR_list

    
def get_mnom_via_maap(datetime_range=None, 
                        region=None,
                        frame=None,
                        orbitNumber=None,
                        ec_collection = ['EarthCAREL1Validated_MAAP'],
                        mnom_variables=['time', 'latitude', 'longitude', 'TIR1', 'TIR2', 'TIR3','land_flag'],
                        max_items=1,
                        trim=True
                        ):
    """Function to access EarthCARE M-NOM data via MAAP."""

    ds_url_list = get_ec_data_url_via_maap(datetime_range, 'MSI_NOM_1B', region, frame, orbitNumber=orbitNumber, ec_collection=ec_collection, max_items=max_items)

    print("Fetching ESA MAAP token...")
    ESA_TOKEN = get_token()
    fs = fsspec.filesystem(
        "https",
        headers={"Authorization": f"Bearer {ESA_TOKEN}"},
        **io_params["fsspec_params"]  )

    MNOM_list = []
    for ds_url, orbit in ds_url_list:
        print("Dataset URL:", ds_url)
        print("Orbit:", orbit)
        with fs.open(ds_url, "rb") as f:
            with xr.open_dataset(f,
                                engine="h5netcdf",
                                **io_params["h5py_params"],
                                group="ScienceData") as MNOM:
            
                if 'TIR1' in mnom_variables:
                    # For MSI_RGR_1C, pixel_values has an extra band dimension
                    MNOM['TIR1'] = MNOM.pixel_values.isel(band=4).where(MNOM.pixel_quality_status.isel(band=4) == 0)
                if 'TIR2' in mnom_variables:
                    MNOM['TIR2'] = MNOM.pixel_values.isel(band=5).where(MNOM.pixel_quality_status.isel(band=5) == 0)
                if 'TIR3' in mnom_variables:
                    MNOM['TIR3'] = MNOM.pixel_values.isel(band=6).where(MNOM.pixel_quality_status.isel(band=6) == 0)
                if 'latitude' in mnom_variables:
                    MNOM['latitude'] = MNOM.latitude.isel(band=4)
                if 'longitude' in mnom_variables:
                    MNOM['longitude'] = MNOM.longitude.isel(band=4)
                if 'land_flag' in mnom_variables:
                    MNOM['land_flag'] = MNOM.land_flag.isel(band=4)
                ## Select only the requested variables
                MNOM = MNOM[mnom_variables]
                MNOM.encoding = {'source': ds_url}  # Keep track of source URL
                if trim:
                    idx_nadir = 284
                    MNOM = ecio.trim_to_frame(MNOM, latvar='latitude', sel_nadir={'across_track':idx_nadir})
                if region is not None:
                    print(f"Cropping to region {region}... lat_min, lat_max = {bbox_regions[region]['lat']}, lon_min, lon_max = {bbox_regions[region]['lon']}")
                    print(f"\tbefore cropping: lat_min, lat_max = {MNOM.latitude.min().values}, {MNOM.latitude.max().values}, lon_min, lon_max = {MNOM.longitude.min().values}, {MNOM.longitude.max().values}")
                    MNOM = crop_to_stcu_region(MNOM, region, sel_across_track_idx=284)
                    print(f" \tafter cropping: lat_min, lat_max = {MNOM.latitude.min().values}, {MNOM.latitude.max().values}, lon_min, lon_max = {MNOM.longitude.min().values}, {MNOM.longitude.max().values}")
                if MNOM is None:
                    continue
                MNOM = MNOM.load() # Load data into memory
        MNOM_list.append((MNOM, orbit))
    return MNOM_list


def get_acmcap_via_maap(datetime_range=None, 
                        region=None, 
                        frame=None,
                        orbitNumber=None,
                        ec_collection = ['EarthCAREL2Validated_MAAP'],
                        acmcap_variables=None,
                        max_items=1
                        ):
    """Function to access EarthCARE MSI data via MAAP."""

    ds_url_list = get_ec_data_url_via_maap(datetime_range, 'ACM_CAP_2B', region, frame, orbitNumber=orbitNumber, ec_collection=ec_collection, max_items=max_items)

    print("Fetching ESA MAAP token...")
    ESA_TOKEN = get_token()
    fs = fsspec.filesystem(
        "https",
        headers={"Authorization": f"Bearer {ESA_TOKEN}"},
        **io_params["fsspec_params"]  )

    ACMCAP_list = []
    for ds_url, orbit in ds_url_list:
        print("Dataset URL:", ds_url)
        print("Orbit:", orbit)
        # Open the file and read it into an xarray Dataset
        with fs.open(ds_url, "rb") as f:
            with xr.open_dataset(f,
                                engine="h5netcdf",
                                **io_params["h5py_params"],
                                group="ScienceData") as ACMCAP:
                # Select only the requested variables
                if acmcap_variables is not None:
                    ACMCAP = ACMCAP[acmcap_variables]
                ACMCAP.encoding = {'source': ds_url}  # Keep track of source URL
                if region is not None:
                    print(f"Cropping to region {region}... lat_min, lat_max = {bbox_regions[region]['lat']}, lon_min, lon_max = {bbox_regions[region]['lon']}")
                    print(f"\tbefore cropping: lat_min, lat_max = {ACMCAP.latitude.min().values:.2f}, {ACMCAP.latitude.max().values:.2f}, lon_min, lon_max = {ACMCAP.longitude.min().values:.2f}, {ACMCAP.longitude.max().values:.2f}")
                    ACMCAP = crop_to_stcu_region(ACMCAP, region)
                    print(f" \tafter cropping: lat_min, lat_max = {ACMCAP.latitude.min().values:.2f}, {ACMCAP.latitude.max().values:.2f}, lon_min, lon_max = {ACMCAP.longitude.min().values:.2f}, {ACMCAP.longitude.max().values:.2f}")
                if ACMCAP is None:
                    continue
                ACMCAP = ACMCAP.load() # Load data into memory
        ACMCAP_list.append((ACMCAP, orbit))
    return ACMCAP_list
