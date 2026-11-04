"""
Functions adapted from:
https://docs.maap-project.org/en/develop/science/EarthCARE/EarthCARE_access_and_visualize.html
"""

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import sys

import fsspec
import numpy as np
import requests
import xarray as xr
from pystac_client import Client

sys.path.append("/home/jmayer/ACI_research/ectools/")
from ectools import ecio

catalog_url = "https://catalog.maap.eo.esa.int/catalogue/"
catalog = Client.open(catalog_url)

# Path to credentials file
CREDENTIALS_FILE = Path(
    "/home/jmayer/ACI_research/Code/mesoscale_structure/credentials_maap.txt"
)

io_params = {
    "fsspec_params": {
        "cache_type": "blockcache",
        "block_size": 8 * 1024 * 1024,
    },
    "h5py_params": {
        "driver_kwds": {
            "rdcc_nbytes": 8 * 1024 * 1024,
        }
    },
}

bbox_regions = {
    "global": {"lat": (-90, 90), "lon": (-180, 180)},
    "SEA": {"lat": (-22.5, -2.5), "lon": (-5, 15)},
    "NEA": {"lat": (10, 30), "lon": (-40, -20)},
    "SEP": {"lat": (-22.5, -2.5), "lon": (-95, -75)},
    "NEP": {"lat": (15, 35), "lon": (-135, -115)},
    "SEA_coastal": {"lat": (-22.5, -2.5), "lon": (-5, 15)},
    "SEP_coastal": {"lat": (-22.5, -2.5), "lon": (-90, -70)},
    "NEP_coastal": {"lat": (15, 35), "lon": (-125, -105)},
}

default_product_version = "ba"


def load_credentials(file_path=CREDENTIALS_FILE):
    """Read key-value pairs from a credentials file into a dictionary."""
    creds = {}
    if not file_path.exists():
        raise FileNotFoundError(f"Credentials file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            creds[key.strip()] = value.strip()
    return creds


def list_available_collections():
    """Return collection IDs and titles exposed by the MAAP EarthCARE catalog."""
    return [
        (collection.id, getattr(collection, "title", None))
        for collection in catalog.get_collections()
    ]


@lru_cache(maxsize=1)
def get_token():
    """Use OFFLINE_TOKEN to fetch a short-lived access token."""
    creds = load_credentials()

    offline_token = creds.get("OFFLINE_TOKEN")
    client_id = creds.get("CLIENT_ID")
    client_secret = creds.get("CLIENT_SECRET")

    if not all([offline_token, client_id, client_secret]):
        raise ValueError("Missing OFFLINE_TOKEN, CLIENT_ID, or CLIENT_SECRET in credentials file")

    url = "https://iam.maap.eo.esa.int/realms/esa-maap/protocol/openid-connect/token"
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "refresh_token",
        "refresh_token": offline_token,
        "scope": "offline_access openid",
    }

    response = requests.post(url, data=data, timeout=60)
    response.raise_for_status()

    access_token = response.json().get("access_token")
    if not access_token:
        raise RuntimeError("Failed to retrieve access token from IAM response")

    return access_token


def region_to_bbox(region):
    """Convert region name to [lon_min, lat_min, lon_max, lat_max]."""
    if region not in bbox_regions:
        raise KeyError(f"Unknown region '{region}'. Available: {list(bbox_regions)}")
    lat_min, lat_max = bbox_regions[region]["lat"]
    lon_min, lon_max = bbox_regions[region]["lon"]
    return [lon_min, lat_min, lon_max, lat_max]


def _to_iso_utc(value):
    """Convert common datetime inputs to RFC3339 UTC with trailing Z."""
    if isinstance(value, str):
        s = value.strip()
        if s.endswith("Z"):
            return s
        if "T" in s:
            return s + "Z"
        return s + "T00:00:00Z"

    if isinstance(value, np.datetime64):
        if np.isnat(value):
            raise ValueError("NaT is not a valid datetime input.")
        value_ns = value.astype("datetime64[ns]").astype("int64")
        dt = datetime.fromtimestamp(value_ns / 1e9, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    if isinstance(value, datetime):
        dt = value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    raise TypeError(f"Unsupported datetime value type: {type(value)}")


def _normalize_datetime_range(datetime_range):
    """
    Return a STAC-compatible datetime interval string.

    This avoids datetime arithmetic that can trigger NumPy datetime ufunc errors.
    """
    if datetime_range is None:
        start = "2024-07-01T00:00:00Z"
        end = datetime.now(timezone.utc).strftime("%Y-%m-%dT23:59:00Z")
        return f"{start}/{end}"

    if isinstance(datetime_range, str):
        return datetime_range

    if isinstance(datetime_range, (list, tuple)) and len(datetime_range) == 2:
        start, end = datetime_range
        return f"{_to_iso_utc(start)}/{_to_iso_utc(end)}"

    raise ValueError("datetime_range must be None, a string, or a 2-element list/tuple.")


def crop_by_lon_or_lat(ds_ec, lon_or_lat_var, l_min, l_max, sel_across_track_idx=None):
    ds = ds_ec.sel(across_track=sel_across_track_idx) if sel_across_track_idx is not None else ds_ec

    first = ds[lon_or_lat_var].values[0]
    last = ds[lon_or_lat_var].values[-1]
    if not (np.isfinite(first) and np.isfinite(last)):
        raise ValueError(
            f"Cannot determine whether {lon_or_lat_var} is ascending or descending due to NaN values at the edges."
        )

    lonlat_descending = first > last
    l_start, l_stop = (l_max, l_min) if lonlat_descending else (l_min, l_max)

    values = ds[lon_or_lat_var].values
    i_start = int(np.nanargmin(np.abs(values - l_start)))
    i_stop = int(np.nanargmin(np.abs(values - l_stop)))

    print(
        f"\tCrop by {lon_or_lat_var}: selecting frame from {i_start} "
        f"({lon_or_lat_var}={values[i_start]:.2f}) to {i_stop} ({lon_or_lat_var}={values[i_stop]:.2f})"
    )
    return ds_ec.isel({"along_track": slice(i_start, i_stop)})


def crop_to_stcu_region(ds_ec, region, latvar='latitude', lonvar='longitude', sel_across_track_idx=None):
    lat_min, lat_max = bbox_regions[region]["lat"]
    lon_min, lon_max = bbox_regions[region]["lon"]

    ds_tmp = crop_by_lon_or_lat(
        ds_ec=ds_ec,
        lon_or_lat_var=latvar,
        l_min=lat_min,
        l_max=lat_max,
        sel_across_track_idx=sel_across_track_idx,
    )
    return crop_by_lon_or_lat(
        ds_ec=ds_tmp,
        lon_or_lat_var=lonvar,
        l_min=lon_min,
        l_max=lon_max,
        sel_across_track_idx=sel_across_track_idx,
    )


def _build_search_filter(product, frame=None, orbit_number=None, product_version=default_product_version):
    filters = [f"productType = '{product}'"]
    if frame is not None:
        filters.append(f"frame = '{frame}'")
    if orbit_number is not None:
        filters.append(f"orbitNumber = {orbit_number}")
    if product_version is not None:
        filters.append(f"productVersion = '{product_version}'")
    return " and ".join(filters)


def _extract_h5_url(item):
    for enc in ("enclosure_h5", "enclosure_hdr"):
        asset = getattr(item, "assets", {}).get(enc)
        href = getattr(asset, "href", None) if asset else None
        if href and href.lower().endswith(".h5"):
            return href
    raise FileNotFoundError("No .h5 file found via enclosure_h5 or enclosure_hdr.")


def _extract_orbit(item):
    title = item.properties.get("title", "")
    return title[-6:] if len(title) >= 6 else "unknown"


def get_ec_data_url_via_maap(
    datetime_range,
    product,
    region=None,
    frame=None,
    orbitNumber=None,
    productVersion=default_product_version,
    ec_collection=None,
    max_items=5,
):
    ec_collection = ec_collection or ["EarthCAREL2Validated_MAAP"]
    datetime_query = _normalize_datetime_range(datetime_range)

    if orbitNumber is not None and product == "MSI_NOM_1B":
        # Some MSI_NOM_1B bbox queries are unreliable; query globally for orbit-specific requests.
        region = "global"

    search_kwargs = {
        "collections": ec_collection,
        "filter": _build_search_filter(product, frame, orbitNumber, productVersion),
        "datetime": datetime_query,
        "method": "GET",
        "max_items": max_items,
    }
    if region is not None:
        search_kwargs["bbox"] = region_to_bbox(region)

    search = catalog.search(**search_kwargs)
    items = list(search.items())

    print(f"Accessing {len(items)} items (limited by max_items).")
    print(f"{search.matched()} items found that matched the query.")

    if not items:
        raise FileNotFoundError("No items found for the given query parameters.")
    if len(items) < search.matched():
        print("Warning: Retrieved items are fewer than matched items. Consider increasing max_items.")

    ds_url_list = []
    for item in items:
        ds_url = _extract_h5_url(item)
        orbit = _extract_orbit(item)
        print(f"Found .h5 URL: {ds_url}")
        print(f"Orbit: {orbit}")
        ds_url_list.append((ds_url, orbit))

    return ds_url_list


def _create_fs(token):
    return fsspec.filesystem(
        "https",
        headers={"Authorization": f"Bearer {token}"},
        **io_params["fsspec_params"],
    )


def _apply_common_ops(
    ds,
    ds_url,
    region=None,
    trim=False,
    latvar="latitude",
    lonvar="longitude",
    idx_nadir=284,
    crop_across_track_idx=None,
):
    ds.encoding = {"source": ds_url}

    if trim:
        ds = ecio.trim_to_frame(ds, latvar=latvar, sel_nadir={"across_track": idx_nadir})

    if region is not None:
        print(
            f"Cropping to region {region}: "
            f"lat={bbox_regions[region]['lat']}, lon={bbox_regions[region]['lon']}"
        )
        print(
            f"\tbefore: lat=({ds[latvar].min().values}, {ds[latvar].max().values}), "
            f"lon=({ds[lonvar].min().values}, {ds[lonvar].max().values})"
        )
        ds = crop_to_stcu_region(ds, region, latvar=latvar, lonvar=lonvar, sel_across_track_idx=crop_across_track_idx)
        print(
            f"\tafter:  lat=({ds[latvar].min().values}, {ds[latvar].max().values}), "
            f"lon=({ds[lonvar].min().values}, {ds[lonvar].max().values})"
        )

    return ds.load()


def _preprocess_mrgr(ds, selected_variables):
    mapping = {
        "TIR1": "TIR1",
        "TIR2": "TIR2",
        "TIR3": "TIR3",
    }
    for var_name, band_name in mapping.items():
        if var_name in selected_variables:
            ds[var_name] = ds.pixel_values.sel(band=band_name).where(
                ds.pixel_quality_status.sel(band=band_name) == 0
            )
    return ds


def _preprocess_mnom(ds, selected_variables):
    band_index = {"TIR1": 4, "TIR2": 5, "TIR3": 6}
    for var_name, bidx in band_index.items():
        if var_name in selected_variables:
            ds[var_name] = ds.pixel_values.isel(band=bidx).where(
                ds.pixel_quality_status.isel(band=bidx) == 0
            )

    for coord_name in ("latitude", "longitude", "land_flag"):
        if coord_name in selected_variables:
            ds[coord_name] = ds[coord_name].isel(band=4)

    return ds


def _fetch_product_datasets(
    product,
    datetime_range=None,
    region=None,
    frame=None,
    orbit_number=None,
    product_version=default_product_version,
    ec_collection=None,
    selected_variables=None,
    max_items=1,
    trim=False,
    group="ScienceData",
    engine="h5netcdf",
    preprocess=None,
    crop_across_track_idx=None,
    latvar="latitude",
    lonvar="longitude"
):
    ds_url_list = get_ec_data_url_via_maap(
        datetime_range=datetime_range,
        product=product,
        region=region,
        frame=frame,
        orbitNumber=orbit_number,
        productVersion=product_version,
        ec_collection=ec_collection,
        max_items=max_items,
    )

    print("Fetching ESA MAAP token...")
    fs = _create_fs(get_token())

    results = []
    for ds_url, orbit in ds_url_list:
        print(f"Dataset URL: {ds_url}")
        print(f"Orbit: {orbit}")

        with fs.open(ds_url, "rb") as f:
            with xr.open_dataset(
                f,
                engine=engine,
                **io_params["h5py_params"],
                group=group,
            ) as ds:
                if preprocess is not None:
                    ds = preprocess(ds, selected_variables or [])

                if selected_variables is not None:
                    ds = ds[selected_variables]

                ds = _apply_common_ops(
                    ds,
                    ds_url=ds_url,
                    region=region,
                    trim=trim,
                    latvar=latvar,
                    lonvar=lonvar,
                    crop_across_track_idx=crop_across_track_idx,
                )

        results.append((ds, orbit))

    return results


def get_mcop_via_maap(
    datetime_range=None,
    region=None,
    frame=None,
    orbitNumber=None,
    productVersion=default_product_version,
    ec_collection=None,
    mcop_variables=None,
    max_items=1,
    trim=True,
):
    """Access EarthCARE MSI_COP_2A data via MAAP."""
    if mcop_variables is None:
        mcop_variables = [
            "time",
            "latitude",
            "longitude",
            "cloud_water_path",
            "cloud_optical_thickness",
        ]

    return _fetch_product_datasets(
        product="MSI_COP_2A",
        datetime_range=datetime_range,
        region=region,
        frame=frame,
        orbit_number=orbitNumber,
        product_version=productVersion,
        ec_collection=ec_collection or ["EarthCAREL2Validated_MAAP"],
        selected_variables=mcop_variables,
        max_items=max_items,
        trim=trim,
        engine=None,
        crop_across_track_idx=284,
    )


def get_mrgr_via_maap(
    datetime_range=None,
    region=None,
    frame=None,
    orbitNumber=None,
    productVersion=default_product_version,
    ec_collection=None,
    mrgr_variables=None,
    max_items=1,
    trim=True,
):
    """Access EarthCARE MSI_RGR_1C data via MAAP."""
    if mrgr_variables is None:
        mrgr_variables = ["time", "latitude", "longitude", "TIR1", "TIR2", "TIR3"]

    return _fetch_product_datasets(
        product="MSI_RGR_1C",
        datetime_range=datetime_range,
        region=region,
        frame=frame,
        orbit_number=orbitNumber,
        product_version=productVersion,
        ec_collection=ec_collection or ["EarthCAREL1Validated_MAAP"],
        selected_variables=mrgr_variables,
        max_items=max_items,
        trim=trim,
        preprocess=_preprocess_mrgr,
        crop_across_track_idx=284,
    )


def get_mnom_via_maap(
    datetime_range=None,
    region=None,
    frame=None,
    orbitNumber=None,
    productVersion=default_product_version,
    ec_collection=None,
    mnom_variables=None,
    max_items=1,
    trim=True,
):
    """Access EarthCARE MSI_NOM_1B data via MAAP."""
    if mnom_variables is None:
        mnom_variables = ["time", "latitude", "longitude", "TIR1", "TIR2", "TIR3", "land_flag"]

    return _fetch_product_datasets(
        product="MSI_NOM_1B",
        datetime_range=datetime_range,
        region=region,
        frame=frame,
        orbit_number=orbitNumber,
        product_version=productVersion,
        ec_collection=ec_collection or ["EarthCAREL1Validated_MAAP"],
        selected_variables=mnom_variables,
        max_items=max_items,
        trim=trim,
        preprocess=_preprocess_mnom,
        crop_across_track_idx=284,
    )


def get_acmcap_via_maap(
    datetime_range=None,
    region=None,
    frame=None,
    orbitNumber=None,
    productVersion=default_product_version,
    ec_collection=None,
    acmcap_variables=None,
    max_items=1,
):
    """Access EarthCARE ACM_CAP_2B data via MAAP."""
    return _fetch_product_datasets(
        product="ACM_CAP_2B",
        datetime_range=datetime_range,
        region=region,
        frame=frame,
        orbit_number=orbitNumber,
        product_version=productVersion,
        ec_collection=ec_collection or ["EarthCAREL2Validated_MAAP"],
        selected_variables=acmcap_variables,
        max_items=max_items,
    )


def get_aebd_via_maap(
    datetime_range=None,
    region=None,
    frame=None,
    orbitNumber=None,
    productVersion=default_product_version,
    ec_collection=None,
    aebd_variables=None,
    max_items=1,
):
    """Access EarthCARE ATL_EBD_2A data via MAAP."""
    return _fetch_product_datasets(
        product="ATL_EBD_2A",
        datetime_range=datetime_range,
        region=region,
        frame=frame,
        orbit_number=orbitNumber,
        product_version=productVersion,
        ec_collection=ec_collection or ["EarthCAREL2Validated_MAAP"],
        selected_variables=aebd_variables,
        max_items=max_items,
    )


def get_actc_via_maap(
    datetime_range=None,
    region=None,
    frame=None,
    orbitNumber=None,
    productVersion=default_product_version,
    ec_collection=None,
    actc_variables=None,
    max_items=1,
):
    """Access EarthCARE AC__TC__2B data via MAAP."""
    return _fetch_product_datasets(
        product="AC__TC__2B",
        datetime_range=datetime_range,
        region=region,
        frame=frame,
        orbit_number=orbitNumber,
        product_version=productVersion,
        ec_collection=ec_collection or ["EarthCAREL2Validated_MAAP"],
        selected_variables=actc_variables,
        max_items=max_items,
    )


def get_cfmr_via_maap(
    datetime_range=None,
    region=None,
    frame=None,
    orbitNumber=None,
    productVersion=default_product_version,
    ec_collection=None,
    cfmr_variables=None,
    max_items=1,
):
    """Access EarthCARE CPR_FMR_2A data via MAAP."""
    return _fetch_product_datasets(
        product="CPR_FMR_2A",
        datetime_range=datetime_range,
        region=region,
        frame=frame,
        orbit_number=orbitNumber,
        product_version=productVersion,
        ec_collection=ec_collection or ["EarthCAREL2Validated_MAAP"],
        selected_variables=cfmr_variables,
        max_items=max_items,
    )


def get_ccd_via_maap(
    datetime_range=None,
    region=None,
    frame=None,
    orbitNumber=None,
    productVersion=default_product_version,
    ec_collection=None,
    ccd_variables=None,
    max_items=1,
):
    """Access EarthCARE CPR_CD__2A data via MAAP."""
    return _fetch_product_datasets(
        product="CPR_CD__2A",
        datetime_range=datetime_range,
        region=region,
        frame=frame,
        orbit_number=orbitNumber,
        product_version=productVersion,
        ec_collection=ec_collection or ["EarthCAREL2Validated_MAAP"],
        selected_variables=ccd_variables,
        max_items=max_items,
    )

def get_ccld_via_maap(
    datetime_range=None,
    region=None,
    frame=None,
    orbitNumber=None,
    productVersion=default_product_version,
    ec_collection=None,
    ccld_variables=None,
    max_items=1,
):
    """Access EarthCARE CPR_CLD_2A data via MAAP."""
    return _fetch_product_datasets(
        product="CPR_CLD_2A",
        datetime_range=datetime_range,
        region=region,
        frame=frame,
        orbit_number=orbitNumber,
        product_version=productVersion,
        ec_collection=ec_collection or ["EarthCAREL2Validated_MAAP"],
        selected_variables=ccld_variables,
        max_items=max_items,
    )


def get_bmaflx_via_maap(
    datetime_range=None,
    region=None,
    frame=None,
    orbitNumber=None,
    productVersion=default_product_version,
    ec_collection=None,
    bmaflx_variables=None,
    max_items=1,
):
    """Access EarthCARE BMA_FLX_2B data via MAAP."""
    return _fetch_product_datasets(
        product="BMA_FLX_2B",
        datetime_range=datetime_range,
        region=region,
        frame=frame,
        orbit_number=orbitNumber,
        product_version=productVersion,
        ec_collection=ec_collection or ["EarthCAREL2Validated_MAAP"],
        selected_variables=bmaflx_variables,
        max_items=max_items,
        group="ScienceData/AssessmentResolution",
    )


def get_acmrt_via_maap(
    datetime_range=None,
    region=None,
    frame=None,
    orbitNumber=None,
    productVersion=default_product_version,
    ec_collection=None,
    acmrt_variables=None,
    latvar="latitude_active",
    lonvar="longitude_active",
    max_items=1,
):
    """Access EarthCARE ACM_RT__2B data via MAAP."""
    return _fetch_product_datasets(
        product="ACM_RT__2B",
        datetime_range=datetime_range,
        region=region,
        frame=frame,
        orbit_number=orbitNumber,
        product_version=productVersion,
        ec_collection=ec_collection or ["EarthCAREL2Validated_MAAP"],
        selected_variables=acmrt_variables,
        latvar=latvar,
        lonvar=lonvar,
        max_items=max_items
    )


def get_auxmet_via_maap(
    datetime_range=None,
    region=None,
    frame=None,
    orbitNumber=None,
    productVersion=default_product_version,
    ec_collection=None,
    auxmet_variables=None,
    max_items=1,
):
    """Access EarthCARE AUX_MET_1D data via MAAP."""
    return _fetch_product_datasets(
        product="AUX_MET_1D",
        datetime_range=datetime_range,
        region=region,
        frame=frame,
        orbit_number=orbitNumber,
        product_version=productVersion,
        ec_collection=ec_collection or ["EarthCAREXMETL1DProducts10_MAAP"],
        selected_variables=auxmet_variables,
        max_items=max_items,
    )
