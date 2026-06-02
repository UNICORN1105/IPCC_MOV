#!/usr/bin/env python3
"""
AR7 MOV extended index calculator.

Example local run:
python compute_ar7_indices.py \
  --local-root /workspace1/CMIP6_LME \
  --scenario ssp370 \
  --model cesm2 \
  --outdir /workspace1/CMIP6_LME/AR7_indices \
  --min-members 6
"""

import os
import re
import glob
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr


KNOWN_SCENARIOS = ["ssp585", "ssp370", "ssp245", "ssp126", "rcp85", "rcp45", "rcp26"]
SST_CANDIDATES_DEFAULT = ["tos", "ts", "sst"]
LOCAL_MODEL_FIRST_ROOT_DEFAULT = "/workspace1/CMIP6_LME"


# =============================================================================
# MOV index calculation notes
# =============================================================================
# General workflow from the email:
#   - Use monthly large-ensemble data.
#   - Keep only models with at least --min-members members.
#   - Estimate the forced component as the model/scenario ensemble mean.
#   - Compute most SST/precipitation indices from residual fields:
#         residual = member field - model/scenario ensemble mean field.
#   - Save raw versions with suffix "_raw" where useful for checking sensitivity
#     to forced-component removal, especially for annular modes.
#
# Common helper functions used by the index functions:
#   monthly_anomaly()       : remove 1991-2020 monthly climatology.
#   monthly_standardize()   : remove 1991-2020 monthly climatology and divide by monthly std.
#   seasonal_mean()         : DJF/MAM/JJA/SON/ANN temporal aggregation.
#   area_weighted_mean()    : cosine-latitude weighted regional mean.
#   zonal_mean_at_lat()     : zonal mean then interpolation to a target latitude.
#   lowpass_10yr()          : 10-year centered rolling mean.
#   forced_response_and_residual(): ensemble mean and residual calculation.
#
# Input variables used by indices:
#   tos / ts / sst : SST-like field for ENSO_SST, RONI, dSSTx, IOD, TPI/PDV,
#                    AMV, IOB, AMM, AZM.
#   pr             : precipitation for ENSO precipitation Niño3.4.
#   psl            : mean sea-level pressure for NAM, SAM, NAO.
#
# Trend calculation has been removed in v13. This script only writes:
#   <model>_AR7_indices.nc
# =============================================================================


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def section(title):
    print("\n" + "=" * 100, flush=True)
    log(title)
    print("=" * 100, flush=True)


def subsection(title):
    print("\n" + "-" * 100, flush=True)
    log(title)
    print("-" * 100, flush=True)


def normalize_member(member):
    member = str(member)
    member = re.sub(r"(r\d+i\d+p\d+)f\d+", r"\1", member)
    member = re.sub(r"(\d{3,4}\.\d{3}i\d+p\d+)f\d+", r"\1", member)
    return member


def raw_member_from_path(path):
    base = os.path.basename(path)
    patterns = [
        r"r\d+i\d+p\d+f\d+",
        r"r\d+i\d+p\d+",
        r"\d{3,4}\.\d{3}i\d+p\d+f\d+",
        r"\d{3,4}\.\d{3}i\d+p\d+",
        r"\d{3,4}\.\d{3}",
        r"member[_-]?\d+",
        r"realization[_-]?\d+",
        r"ens[_-]?\d+",
    ]
    for pat in patterns:
        m = re.search(pat, base)
        if m:
            return normalize_member(m.group(0))
    return normalize_member(os.path.basename(os.path.dirname(path)))


def filename_lower(path):
    return os.path.basename(path).lower()


def file_is_pure_historical(path):
    b = filename_lower(path)
    return ("historical" in b) and not any(s in b for s in KNOWN_SCENARIOS)


def file_is_combined_historical_scenario(path, scenario):
    b = filename_lower(path)
    return ("historical" in b) and (scenario.lower() in b)


def file_is_future_scenario(path, scenario):
    b = filename_lower(path)
    return (scenario.lower() in b) and not file_is_combined_historical_scenario(path, scenario)


def select_files_for_scenario(files, scenario):
    files = sorted(files)
    combined = [f for f in files if file_is_combined_historical_scenario(f, scenario)]
    if combined:
        return sorted(set(combined)), f"combined_historical_{scenario}"

    historical = [f for f in files if file_is_pure_historical(f)]
    future = [f for f in files if file_is_future_scenario(f, scenario)]
    if future:
        return sorted(set(historical + future)), f"historical_plus_{scenario}"

    return [], f"missing_{scenario}"


def parse_yyyymm_range_from_filename(path):
    base = os.path.basename(path)
    m = re.search(r"_(\d{6})-(\d{6})\.nc$", base)
    if m is None:
        m = re.search(r"(\d{6})-(\d{6})", base)
    if m is None:
        return None, None
    return m.group(1), m.group(2)


def collect_member_files(base_dir, varname, model, scenario):
    """
    Local model-first input layout:

        /workspace1/CMIP6_LME/<model>/<var>/*.nc

    Example:

        /workspace1/CMIP6_LME/cesm2/pr/*.nc
        /workspace1/CMIP6_LME/cesm2/psl/*.nc
        /workspace1/CMIP6_LME/cesm2/tas/*.nc
        /workspace1/CMIP6_LME/cesm2/tos/*.nc

    This function replaces only the input-file discovery logic.
    The index calculation logic below is unchanged.

    The old argument base_dir is kept for compatibility with the original code,
    but is not used in the local model-first layout.
    """
    local_root = Path(os.environ.get("CMIP6_LME_ROOT", LOCAL_MODEL_FIRST_ROOT_DEFAULT))

    model_dir = local_root / model / varname

    log(f"Collecting files [LOCAL MODEL-FIRST]: var={varname}, model={model}")
    log(f"    Local root: {local_root}")
    log(f"    Directory : {model_dir}")

    if not model_dir.exists():
        log(f"    Directory missing, skip: {model_dir}")
        return {}

    files = sorted(glob.glob(str(model_dir / "**" / "*.nc"), recursive=True))
    log(f"    Total NetCDF files found before scenario filtering: {len(files)}")

    raw = {}
    for f in files:
        raw.setdefault(raw_member_from_path(f), []).append(f)

    out = {}
    counts = {}
    for member, flist in raw.items():
        selected, selection_type = select_files_for_scenario(flist, scenario)
        if selected:
            out[member] = selected
            counts[selection_type] = counts.get(selection_type, 0) + 1

    log(f"    Members detected after scenario filtering: {len(out)}")
    log(f"    Scenario/member counts: {counts}")
    log(f"    First members: {list(sorted(out))[:10]}")

    return out


def normalize_time_to_noleap_month_start(ds):
    if "time" not in ds.coords:
        return ds

    years = np.asarray(ds["time"].dt.year.values, dtype=int)
    months = np.asarray(ds["time"].dt.month.values, dtype=int)

    try:
        import cftime
        times = [cftime.DatetimeNoLeap(int(y), int(m), 1) for y, m in zip(years, months)]
    except Exception:
        times = xr.cftime_range(
            start=f"{years[0]:04d}-{months[0]:02d}-01",
            periods=len(years),
            freq="MS",
            calendar="noleap",
        )

    return ds.assign_coords(time=("time", times))


def open_mfdataset_robust(files, time_chunk):
    """
    Very safe local-server opener.

    Local server fix:
    - Do NOT use xr.open_mfdataset(... parallel=True).
    - Do NOT decode CF time first.
    - Open files one by one with decode_times=False.
    - Build a synthetic monthly noleap time axis from the YYYYMM-YYYYMM filename.
    - Concat serially.

    This changes only input reading. The downstream index calculations are unchanged.
    """
    files = sorted(files)
    log(f"    [SAFE LOCAL OPEN] nfiles={len(files)}")
    for f in files:
        log(f"        file: {f}")

    datasets = []

    for f in files:
        start, end = parse_yyyymm_range_from_filename(f)
        if start is None:
            raise RuntimeError(f"Cannot infer YYYYMM range from filename: {f}")

        log(f"    [SAFE LOCAL OPEN] opening: {f}")

        ds = xr.open_dataset(
            f,
            decode_times=False,
            chunks={"time": time_chunk},
            engine="netcdf4",
        )

        ntime = ds.sizes.get("time", 0)
        y0 = int(start[:4])
        m0 = int(start[4:6])

        times = xr.cftime_range(
            start=f"{y0:04d}-{m0:02d}-01",
            periods=ntime,
            freq="MS",
            calendar="noleap",
        )

        ds = ds.assign_coords(time=("time", times))
        datasets.append(ds)

        log(f"    [SAFE LOCAL OPEN] opened OK: ntime={ntime}, start={start}, end={end}")

    if len(datasets) == 1:
        out = datasets[0]
    else:
        log("    [SAFE LOCAL OPEN] concatenating files serially")
        out = xr.concat(
            datasets,
            dim="time",
            coords="minimal",
            compat="override",
            join="outer",
        ).sortby("time")

    log(f"    [SAFE LOCAL OPEN] done dims={dict(out.sizes)}")
    return out



def select_year_range(da, start_year, end_year):
    yrs = da["time"].dt.year
    return da.where((yrs >= start_year) & (yrs <= end_year), drop=True)


def get_lat_lon_names(da):
    lat_name = None
    lon_name = None
    for name in list(da.coords) + list(da.dims):
        lname = name.lower()
        if lname in ["lat", "latitude", "nav_lat", "y"]:
            lat_name = name
        if lname in ["lon", "longitude", "nav_lon", "x"]:
            lon_name = name
    if lat_name is None or lon_name is None:
        raise ValueError(f"Cannot identify lat/lon for {da.name}: coords={list(da.coords)}, dims={list(da.dims)}")
    return lat_name, lon_name


def ensure_lon_0_360(da):
    lat_name, lon_name = get_lat_lon_names(da)
    lon = da[lon_name]
    if lon.ndim != 1:
        raise ValueError(f"{da.name}: only 1D lon supported. lon.ndim={lon.ndim}")
    if float(lon.min()) < 0:
        da = da.assign_coords({lon_name: lon % 360})
        da = da.sortby(lon_name)
    return da


def area_weighted_mean(da, lat_bounds, lon_bounds):
    """
    Area-weighted regional mean.

    v11 fix:
    lon_bounds=(0, 360) or any 360-degree-wide interval means "use all longitudes".
    In v10, 360 % 360 became 0, so (0, 360) became slice(0, 0), giving
    Empty region errors in RONI tropical-mean SST.
    """
    lat_name, lon_name = get_lat_lon_names(da)
    lat0, lat1 = lat_bounds
    lon0_in, lon1_in = lon_bounds

    if da[lat_name][0] > da[lat_name][-1]:
        sub = da.sel({lat_name: slice(lat1, lat0)})
    else:
        sub = da.sel({lat_name: slice(lat0, lat1)})

    lon_span = float(lon1_in) - float(lon0_in)
    full_lon = np.isclose(abs(lon_span), 360.0) or np.isclose(abs(lon_span) % 360.0, 0.0)

    if full_lon:
        # Keep all longitudes. Do not convert 360 to 0.
        pass
    else:
        lon0 = lon0_in % 360
        lon1 = lon1_in % 360
        if lon0 <= lon1:
            sub = sub.sel({lon_name: slice(lon0, lon1)})
        else:
            sub1 = sub.sel({lon_name: slice(lon0, 360)})
            sub2 = sub.sel({lon_name: slice(0, lon1)})
            sub = xr.concat([sub1, sub2], dim=lon_name)

    if sub.sizes.get(lat_name, 0) == 0 or sub.sizes.get(lon_name, 0) == 0:
        raise RuntimeError(f"Empty region lat={lat_bounds}, lon={lon_bounds}")

    weights = np.cos(np.deg2rad(sub[lat_name]))
    return sub.weighted(weights).mean(dim=[lat_name, lon_name])


def zonal_mean_at_lat(da, target_lat):
    lat_name, lon_name = get_lat_lon_names(da)
    zm = da.mean(lon_name)
    return zm.interp({lat_name: target_lat})


def monthly_anomaly(da, baseline_start_year, baseline_end_year):
    base = select_year_range(da, baseline_start_year, baseline_end_year)
    if base.sizes.get("time", 0) == 0:
        raise RuntimeError(f"No baseline data for monthly anomaly {baseline_start_year}-{baseline_end_year}")
    clim = base.groupby("time.month").mean("time")
    return da.groupby("time.month") - clim


def monthly_standardize(da, baseline_start_year, baseline_end_year):
    base = select_year_range(da, baseline_start_year, baseline_end_year)
    if base.sizes.get("time", 0) == 0:
        raise RuntimeError(f"No baseline data for monthly standardization {baseline_start_year}-{baseline_end_year}")
    clim = base.groupby("time.month").mean("time")
    std = base.groupby("time.month").std("time")
    return (da.groupby("time.month") - clim) / std


def seasonal_mean(da, season):
    season = season.upper()
    if season == "ANN":
        return da.groupby("time.year").mean("time")

    months = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}[season]
    mon = da["time"].dt.month
    sub = da.where(mon.isin(months), drop=True)
    mon_sub = sub["time"].dt.month
    yr_sub = sub["time"].dt.year

    if season == "DJF":
        season_year = xr.where(mon_sub == 12, yr_sub + 1, yr_sub)
    else:
        season_year = yr_sub

    sub = sub.assign_coords(season_year=("time", np.asarray(season_year.values, dtype=int)))
    return sub.groupby("season_year").mean("time").rename({"season_year": "year"})


def clean_index(da, name=None):
    if name:
        da = da.rename(name)
    if "month" in da.coords:
        da = da.drop_vars("month", errors="ignore")
    for c in list(da.coords):
        if c not in da.dims and c not in ["year", "member"]:
            da = da.drop_vars(c, errors="ignore")
    return da.squeeze(drop=True)


def lowpass_10yr(da, min_periods=7):
    if "year" not in da.dims:
        raise RuntimeError(f"lowpass_10yr expects year dimension, got {dict(da.sizes)}")
    try:
        da = da.chunk({"year": -1})
    except Exception:
        pass
    return clean_index(da.rolling(year=10, center=True, min_periods=min_periods).mean())


def forced_response_and_residual(da):
    if "member" not in da.dims:
        raise RuntimeError("forced_response_and_residual expects member dimension")
    forced = da.mean("member")
    resid = da - forced
    return forced, resid


def convert_pr_to_mmday(da):
    return da * 86400.0


def open_member_variable(files, varname, args):
    ds = open_mfdataset_robust(files, args.time_chunk)

    if varname in ds.data_vars:
        da = ds[varname]
    else:
        dvs = list(ds.data_vars)
        if len(dvs) == 1:
            log(f"    {varname} not found; using {dvs[0]}")
            da = ds[dvs[0]]
            da.name = varname
        else:
            raise ValueError(f"{varname} not found. data_vars={dvs}")

    start_year = int(str(args.full_start)[:4])
    end_year = int(str(args.full_end)[:4])
    da = select_year_range(da, start_year, end_year)
    if da.sizes.get("time", 0) == 0:
        raise RuntimeError(f"{varname}: no data after year selection {start_year}-{end_year}")
    return ensure_lon_0_360(da)


def build_model_member_dataset(member_files, varname, args):
    arrays = []
    for member in sorted(member_files):
        log(f"  Opening member={member}, var={varname}, nfiles={len(member_files[member])}")
        da = open_member_variable(member_files[member], varname, args)
        arrays.append(da.expand_dims(member=[member]))
    if not arrays:
        raise RuntimeError(f"No members for {varname}")
    out = xr.concat(arrays, dim="member", coords="minimal", compat="override", join="outer")
    out = out.sortby("time")
    out = out.load()
    log(f"  Built {varname} array dims: {dict(out.sizes)}")
    return out


def calc_nino34_sst(ts):
    """
    ENSO SST Niño3.4 index.

    Email definition:
        SST NINO34 = SST anomaly averaged over 5°S-5°N, 170°W-120°W;
        DJF season.

    Input variable:
        ts: SST-like input, selected from tos / ts / sst.

    Helper functions used:
        monthly_anomaly(ts, 1991, 2020)
        area_weighted_mean(..., lat=(-5, 5), lon=(190, 240))
        seasonal_mean(..., "DJF")
        clean_index()

    Residual/raw treatment:
        process_model() calls this on ts_resid for nino34_sst_DJF and on raw ts
        for nino34_sst_DJF_raw.
    """
    subsection("Calculating ENSO SST NINO3.4 DJF")
    anom = monthly_anomaly(ts, 1991, 2020)
    idx = area_weighted_mean(anom, (-5, 5), (190, 240))
    return clean_index(seasonal_mean(idx, "DJF"), "nino34_sst_DJF")

def calc_roni(ts):
    """
    Relative Oceanic Niño Index style index (RONI).

    Definition implemented here:
        Niño3.4 SST anomaly minus tropical-mean SST anomaly; DJF season.
        Niño3.4 box: 5°S-5°N, 170°W-120°W.
        Tropical mean box: 20°S-20°N, all longitudes.

    Input variable:
        ts: SST-like input, selected from tos / ts / sst.

    Helper functions used:
        monthly_anomaly()
        area_weighted_mean() for Niño3.4 and tropical boxes
        seasonal_mean(..., "DJF")
        clean_index()
    """
    subsection("Calculating RONI DJF")
    anom = monthly_anomaly(ts, 1991, 2020)
    nino = area_weighted_mean(anom, (-5, 5), (190, 240))
    trop = area_weighted_mean(anom, (-20, 20), (0, 360))
    return clean_index(seasonal_mean(nino - trop, "DJF"), "roni_DJF")

def calc_precip_nino34(pr):
    """
    ENSO precipitation Niño3.4 index.

    Email definition:
        Precip NINO34 = precipitation anomaly averaged over 5°S-5°N,
        170°W-120°W; DJF season.

    Input variable:
        pr: precipitation, usually kg m-2 s-1 in CMIP files.

    Helper functions used:
        convert_pr_to_mmday() -> converts kg m-2 s-1 to mm day-1.
        monthly_anomaly(pr_mm, 1991, 2020)
        area_weighted_mean(..., lat=(-5, 5), lon=(190, 240))
        seasonal_mean(..., "DJF")
        clean_index()

    Residual/raw treatment:
        process_model() calls this on pr_resid for nino34_pr_DJF and on raw pr
        for nino34_pr_DJF_raw.
    """
    subsection("Calculating ENSO precipitation NINO3.4 DJF")
    pr_mm = convert_pr_to_mmday(pr)
    anom = monthly_anomaly(pr_mm, 1991, 2020)
    idx = area_weighted_mean(anom, (-5, 5), (190, 240))
    return clean_index(seasonal_mean(idx, "DJF"), "nino34_pr_DJF")

def calc_pacific_zonal_gradient(ts):
    """
    Pacific equatorial zonal SST-gradient index.

    Definition implemented here:
        western equatorial Pacific SST anomaly minus eastern equatorial Pacific
        SST anomaly; annual mean.
        West Pacific: 110°E-180°E, 5°S-5°N.
        East Pacific: 180°E-80°W, 5°S-5°N.

    Input variable:
        ts: SST-like input, selected from tos / ts / sst.

    Helper functions used:
        monthly_anomaly()
        area_weighted_mean() for west/east Pacific boxes
        seasonal_mean(..., "ANN")
        clean_index()

    Note:
        This is similar to dSSTx; both names are retained because Chapter 2
        explicitly requested dSSTx.
    """
    subsection("Calculating Pacific Zonal Gradient ANN")
    anom = monthly_anomaly(ts, 1991, 2020)
    wpac = area_weighted_mean(anom, (-5, 5), (110, 180))
    epac = area_weighted_mean(anom, (-5, 5), (180, 280))
    return clean_index(seasonal_mean(wpac - epac, "ANN"), "pacific_zonal_gradient_ANN")

def calc_dsstx(ts):
    """
    Chapter 2 dSSTx index.

    Email/Chapter 2 definition:
        dSSTx = wPac - ePac.
        wPac: 110°E-180°E, 5°S-5°N.
        ePac: 180°E-80°W, 5°S-5°N.
        Annual mean.

    Input variable:
        ts: SST-like input, selected from tos / ts / sst.

    Helper functions used:
        monthly_anomaly()
        area_weighted_mean() for wPac/ePac
        seasonal_mean(..., "ANN")
        clean_index()
    """
    subsection("Calculating Ch2 dSSTx ANN")
    anom = monthly_anomaly(ts, 1991, 2020)
    wpac = area_weighted_mean(anom, (-5, 5), (110, 180))
    epac = area_weighted_mean(anom, (-5, 5), (180, 280))
    return clean_index(seasonal_mean(wpac - epac, "ANN"), "dsstx_ANN")

def calc_tropical_mean_sst(ts):
    """
    Tropical-mean SST anomaly.

    Definition implemented here:
        SST anomaly averaged over 30°S-30°N, all longitudes; annual mean.

    Input variable:
        ts: SST-like input, selected from tos / ts / sst.

    Helper functions used:
        monthly_anomaly()
        area_weighted_mean(..., lat=(-30, 30), lon=(0, 360))
        seasonal_mean(..., "ANN")
        clean_index()
    """
    subsection("Calculating Tropical mean SST ANN")
    anom = monthly_anomaly(ts, 1991, 2020)
    idx = area_weighted_mean(anom, (-30, 30), (0, 360))
    return clean_index(seasonal_mean(idx, "ANN"), "tropical_mean_sst_ANN")

def calc_iod(ts, normalized=True):
    """
    Indian Ocean Dipole index.

    Email definition:
        IOD = difference in normalized SST anomalies between western and eastern
        equatorial Indian Ocean boxes; SON season.
        Western box: 10°S-10°N, 50°E-70°E.
        Eastern box: 10°S-0°, 90°E-110°E.

    Input variable:
        ts: SST-like input, selected from tos / ts / sst.

    Helper functions used:
        monthly_standardize(ts, 1991, 2020) if normalized=True
        monthly_anomaly(ts, 1991, 2020) if normalized=False
        area_weighted_mean() for western/eastern boxes
        seasonal_mean(..., "SON")
        clean_index()

    Residual/raw treatment:
        process_model() calls this on ts_resid for iod_SON and on raw ts for
        iod_SON_raw.
    """
    subsection("Calculating IOD SON")
    field = monthly_standardize(ts, 1991, 2020) if normalized else monthly_anomaly(ts, 1991, 2020)
    west = area_weighted_mean(field, (-10, 10), (50, 70))
    east = area_weighted_mean(field, (-10, 0), (90, 110))
    return clean_index(seasonal_mean(west - east, "SON"), "iod_SON")

def calc_tpi_pdv(ts):
    """
    TPI/PDV index.

    Email definition:
        TPI/PDV = equatorial Pacific SST anomaly minus the average of North and
        South Pacific mid-latitude SST anomalies.
        Equatorial Pacific: 10°S-10°N, 170°E-90°W.
        North Pacific: 25°N-45°N, 140°E-145°W.
        South Pacific: 50°S-15°S, 150°E-160°W.
        Annual mean followed by 10-year low-pass filtering.

    Input variable:
        ts: SST-like input, selected from tos / ts / sst.

    Helper functions used:
        monthly_anomaly()
        area_weighted_mean() for the three boxes
        seasonal_mean(..., "ANN")
        lowpass_10yr()
        clean_index()
    """
    subsection("Calculating TPI/PDV ANN 10-year low-pass")
    anom = monthly_anomaly(ts, 1991, 2020)
    eq = area_weighted_mean(anom, (-10, 10), (170, 270))
    north = area_weighted_mean(anom, (25, 45), (140, 215))
    south = area_weighted_mean(anom, (-50, -15), (150, 200))
    return clean_index(lowpass_10yr(seasonal_mean(eq - 0.5 * (north + south), "ANN")), "tpi_pdv_ANN_10yr")

def calc_amv(ts, subtract_global=True):
    """
    Atlantic Multidecadal Variability index.

    Email definition:
        AMV = North Atlantic SST average, 5°N-60°N, minus near-global SST
        average, 60°S-60°N. Annual mean followed by 10-year low-pass filtering.

    Approximation implemented here:
        North Atlantic box: 5°N-60°N, 80°W-0°.
        Near-global box: 60°S-60°N, all longitudes.

    Input variable:
        ts: SST-like input, selected from tos / ts / sst.

    Helper functions used:
        monthly_anomaly()
        area_weighted_mean() for North Atlantic and near-global boxes
        seasonal_mean(..., "ANN")
        lowpass_10yr()
        clean_index()

    Options:
        subtract_global=True  -> amv_ANN_10yr.
        subtract_global=False -> amv_noglob_ANN_10yr, North Atlantic only.
    """
    label = "amv_ANN_10yr" if subtract_global else "amv_noglob_ANN_10yr"
    subsection(f"Calculating {label}")
    anom = monthly_anomaly(ts, 1991, 2020)
    natl = area_weighted_mean(anom, (5, 60), (280, 360))
    if subtract_global:
        glob = area_weighted_mean(anom, (-60, 60), (0, 360))
        idx = natl - glob
    else:
        idx = natl
    return clean_index(lowpass_10yr(seasonal_mean(idx, "ANN")), label)

def calc_iob(ts):
    """
    Indian Ocean Basin mode index.

    Email definition:
        IOB = SST averaged over the entire Indian Ocean, coast up to 30°S;
        MAM season. The email notes that the exact mask should come from Annex 4.

    Approximation implemented here:
        Rectangular Indian Ocean box: 30°S-30°N, 20°E-120°E; MAM season.

    Input variable:
        ts: SST-like input, selected from tos / ts / sst.

    Helper functions used:
        monthly_anomaly()
        area_weighted_mean()
        seasonal_mean(..., "MAM")
        clean_index()

    Note:
        This is a rectangular-box approximation, not the final Annex-4 mask.
    """
    subsection("Calculating IOB MAM")
    anom = monthly_anomaly(ts, 1991, 2020)
    idx = area_weighted_mean(anom, (-30, 30), (20, 120))
    return clean_index(seasonal_mean(idx, "MAM"), "iob_MAM")

def calc_amm(ts):
    """
    Atlantic Meridional Mode index.

    Email definition:
        AMM = normalized SST difference between north and south tropical Atlantic;
        JJA season.
        North: 5°N-30°N, 20°W-60°W.
        South: 5°N-20°S, 5°E-25°W.

    Input variable:
        ts: SST-like input, selected from tos / ts / sst.

    Helper functions used:
        monthly_standardize()
        area_weighted_mean() for north/south tropical Atlantic boxes
        seasonal_mean(..., "JJA")
        clean_index()
    """
    subsection("Calculating AMM JJA")
    field = monthly_standardize(ts, 1991, 2020)
    north = area_weighted_mean(field, (5, 30), (300, 340))
    south = area_weighted_mean(field, (-20, 5), (335, 5))
    return clean_index(seasonal_mean(north - south, "JJA"), "amm_JJA")

def calc_azm(ts):
    """
    Atlantic Zonal Mode index.

    Email definition:
        AZM = SST anomaly over ATL3 region, 3°S-3°N, 0°-20°W; JJA season.

    Input variable:
        ts: SST-like input, selected from tos / ts / sst.

    Helper functions used:
        monthly_anomaly()
        area_weighted_mean(..., lat=(-3, 3), lon=(340, 360))
        seasonal_mean(..., "JJA")
        clean_index()
    """
    subsection("Calculating AZM JJA")
    anom = monthly_anomaly(ts, 1991, 2020)
    idx = area_weighted_mean(anom, (-3, 3), (340, 360))
    return clean_index(seasonal_mean(idx, "JJA"), "azm_JJA")

def calc_nam_index(psl, season, suffix):
    """
    Northern Annular Mode index.

    Email definition:
        NAM = difference of normalized zonally averaged monthly mean sea-level
        pressure between 35°N and 65°N, following Jianping and Wang (2003).
        Seasons requested: DJF and JJA.

    Input variable:
        psl: mean sea-level pressure.

    Helper functions used:
        zonal_mean_at_lat(psl_hpa, 35)
        zonal_mean_at_lat(psl_hpa, 65)
        monthly_standardize() at each latitude, using 1991-2020 monthly std
        seasonal_mean(..., season)
        clean_index()

    Unit handling:
        psl is converted from Pa to hPa by psl / 100.

    Residual/raw treatment:
        process_model() saves both nam_*_resid and nam_*_raw so we can check
        whether removing the ensemble-mean forced component matters.
    """
    subsection(f"Calculating NAM {season} {suffix}")
    psl_hpa = psl / 100.0
    zm35 = zonal_mean_at_lat(psl_hpa, 35)
    zm65 = zonal_mean_at_lat(psl_hpa, 65)
    idx = monthly_standardize(zm35, 1991, 2020) - monthly_standardize(zm65, 1991, 2020)
    return clean_index(seasonal_mean(idx, season), f"nam_{season}_{suffix}")

def calc_sam_index(psl, season, suffix):
    """
    Southern Annular Mode index.

    Email definition:
        SAM = difference of normalized zonally averaged monthly mean sea-level
        pressure between 40°S and 65°S, following Gong and Wang (1999).
        Seasons requested: DJF and JJA.

    Input variable:
        psl: mean sea-level pressure.

    Helper functions used:
        zonal_mean_at_lat(psl_hpa, -40)
        zonal_mean_at_lat(psl_hpa, -65)
        monthly_standardize() at each latitude, using 1991-2020 monthly std
        seasonal_mean(..., season)
        clean_index()

    Unit handling:
        psl is converted from Pa to hPa by psl / 100.

    Residual/raw treatment:
        process_model() saves both sam_*_resid and sam_*_raw so we can check
        whether removing the ensemble-mean forced component matters.
    """
    subsection(f"Calculating SAM {season} {suffix}")
    psl_hpa = psl / 100.0
    zm40 = zonal_mean_at_lat(psl_hpa, -40)
    zm65 = zonal_mean_at_lat(psl_hpa, -65)
    idx = monthly_standardize(zm40, 1991, 2020) - monthly_standardize(zm65, 1991, 2020)
    return clean_index(seasonal_mean(idx, season), f"sam_{season}_{suffix}")

def subset_nao_region(da):
    lat_name, lon_name = get_lat_lon_names(da)
    if da[lat_name][0] > da[lat_name][-1]:
        sub = da.sel({lat_name: slice(80, 20)})
    else:
        sub = da.sel({lat_name: slice(20, 80)})
    sub1 = sub.sel({lon_name: slice(270, 360)})
    sub2 = sub.sel({lon_name: slice(0, 40)})
    return xr.concat([sub1, sub2], dim=lon_name)


def calc_nao_eof(psl, season, suffix):
    """
    North Atlantic Oscillation index.

    Email definition:
        NAO = normalized principal component of the leading EOF of MSLP over
        20°N-80°N, 90°W-40°E. Seasons requested: DJF and JJA.

    Input variable:
        psl: mean sea-level pressure.

    Helper functions used:
        monthly_anomaly(psl_hpa, 1991, 2020)
        seasonal_mean(..., season)
        subset_nao_region() for 20°N-80°N, 90°W-40°E
        sqrt(cos(lat)) spatial weighting
        stack member/year and lat/lon into sample/space
        numpy.linalg.svd() for leading EOF/PC
        area_weighted_mean() sign-anchor check
        clean_index()

    Residual/raw treatment:
        process_model() saves both nao_*_resid and nao_*_raw.
    """
    subsection(f"Calculating NAO {season} {suffix}")
    psl_hpa = psl / 100.0
    anom = monthly_anomaly(psl_hpa, 1991, 2020)
    seas = seasonal_mean(anom, season)
    region = subset_nao_region(seas)
    lat_name, lon_name = get_lat_lon_names(region)

    weights = np.sqrt(np.cos(np.deg2rad(region[lat_name])))
    weighted = region * weights

    stacked = weighted.stack(sample=("member", "year")).stack(space=(lat_name, lon_name)).transpose("sample", "space")
    log(f"    NAO stacked dims before load: {dict(stacked.sizes)}")
    X = np.asarray(stacked.load().values, dtype=np.float64)

    valid = np.isfinite(X).all(axis=0)
    Xv = X[:, valid]
    if Xv.shape[0] < 5 or Xv.shape[1] < 5:
        raise RuntimeError(f"NAO EOF too few valid samples/spaces: {Xv.shape}")

    Xv = Xv - np.nanmean(Xv, axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(Xv, full_matrices=False)
    pc1 = U[:, 0] * s[0]

    try:
        south = area_weighted_mean(seas, (35, 45), (300, 350)).stack(sample=("member", "year")).values
        north = area_weighted_mean(seas, (60, 70), (300, 350)).stack(sample=("member", "year")).values
        anchor = south - north
        mask = np.isfinite(anchor)
        corr = np.corrcoef(pc1[mask], anchor[mask])[0, 1]
        if np.isfinite(corr) and corr < 0:
            pc1 = -pc1
    except Exception as e:
        log(f"    NAO sign anchor failed: {repr(e)}")

    pc1 = (pc1 - np.nanmean(pc1)) / np.nanstd(pc1)

    sample_index = stacked["sample"].to_index()
    pc_series = pd.Series(pc1, index=sample_index)
    pc_df = pc_series.unstack("year")

    pc_da = xr.DataArray(
        pc_df.values,
        dims=("member", "year"),
        coords={"member": pc_df.index.astype(str).values, "year": pc_df.columns.astype(int).values},
        name=f"nao_{season}_{suffix}",
    )
    return clean_index(pc_da, f"nao_{season}_{suffix}")

def safe_to_netcdf(ds, path, direct_write=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    log(f"Saving file: {path}")
    if path.exists():
        path.unlink()
    ds = ds.compute() 
    ds.to_netcdf(path, engine="netcdf4", encoding=encoding)
    log(f"Wrote {path}")


def find_candidate_models(args):
    """
    Local model-first candidate discovery.

    It scans:
        args.local_root/<model>/

    and accepts a model if any of psl/pr/tos/ts/sst has at least min_members
    for the requested scenario, consistent with the original candidate rule.
    """
    section("Finding candidate models [LOCAL MODEL-FIRST]")

    local_root = Path(args.local_root)
    log(f"Local root = {local_root}")

    if not local_root.exists():
        raise RuntimeError(f"Local root does not exist: {local_root}")

    models = sorted([p.name for p in local_root.iterdir() if p.is_dir()])
    log(f"Total model directories found: {len(models)}")
    log(f"First model directories: {models[:20]}")

    candidates = []

    for model in models:
        if args.model and model != args.model:
            continue

        log(f"Checking model candidate: {model}")

        counts = {}

        for var in ["psl", "pr"]:
            mf = collect_member_files(args.base_dir_atm, var, model, args.scenario)
            if mf:
                counts[var] = len(mf)

        for sst_var in args.sst_candidates:
            base = args.base_dir_ocean if sst_var in ["tos", "sst"] else args.base_dir_atm
            mf = collect_member_files(base, sst_var, model, args.scenario)
            if mf:
                counts[sst_var] = len(mf)

        max_members = max(counts.values()) if counts else 0
        log(f"    Member counts for {model}: {counts}")
        log(f"    Max members = {max_members}")

        if max_members >= args.min_members:
            candidates.append(model)
            log(f"    ACCEPTED: {model}")
        else:
            log(f"    REJECTED: {model}, fewer than {args.min_members} members")

    return candidates


def collect_sst_member_files(args, model):
    for sst_var in args.sst_candidates:
        base = args.base_dir_ocean if sst_var in ["tos", "sst"] else args.base_dir_atm
        mf = collect_member_files(base, sst_var, model, args.scenario)
        if len(mf) >= args.min_members:
            return sst_var, mf
    return None, {}


def common_members_if_possible(dicts, min_members):
    keys = [set(d.keys()) for d in dicts if d]
    if not keys:
        return None
    common = set.intersection(*keys)
    if len(common) >= min_members:
        return sorted(common)
    return None


def process_model(model, args):
    section(f"Processing model: {model}")

    out_index = Path(args.outdir) / f"{model}_AR7_indices.nc"

    if args.skip_existing and out_index.exists():
        try:
            ds_test = xr.open_dataset(out_index)
            nvars = len(ds_test.data_vars)
            ds_test.close()
            if nvars > 0:
                log(f"SKIP existing complete file: {out_index}")
                return "skipped"
        except Exception:
            log(f"Existing file is bad; will recompute: {out_index}")

    sst_var, sst_files = collect_sst_member_files(args, model)
    psl_files = collect_member_files(args.base_dir_atm, "psl", model, args.scenario)
    pr_files = collect_member_files(args.base_dir_atm, "pr", model, args.scenario)

    if args.use_common_members_when_possible:
        common = common_members_if_possible([sst_files, psl_files, pr_files], args.min_members)
        if common is not None:
            log(f"Using common members across SST/PSL/PR: n={len(common)}")
            sst_files = {m: sst_files[m] for m in common if m in sst_files}
            psl_files = {m: psl_files[m] for m in common if m in psl_files}
            pr_files = {m: pr_files[m] for m in common if m in pr_files}
        else:
            log("No common member set with enough members; using variable-specific members.")

    result_parts = []

    if sst_var is not None and len(sst_files) >= args.min_members:
        log(f"{model}: building SST variable {sst_var} with {len(sst_files)} members")
        ts = build_model_member_dataset(sst_files, sst_var, args)

        # SST-family input variable:
        #   sst_var is selected from ["tos", "ts", "sst"] depending on availability.
        # Forced-response treatment:
        #   ts_forced = model/scenario ensemble mean monthly SST field.
        #   ts_resid  = ts - ts_forced for each member.
        # Indices below without "_raw" use ts_resid, following the residual-based MOV workflow.
        # Indices with "_raw" use the original ts for sensitivity checks.
        ts_forced, ts_resid = forced_response_and_residual(ts)
        ds_sst = xr.Dataset()
        ds_sst["nino34_sst_DJF"] = calc_nino34_sst(ts_resid)
        ds_sst["roni_DJF"] = calc_roni(ts_resid)
        ds_sst["pacific_zonal_gradient_ANN"] = calc_pacific_zonal_gradient(ts_resid)
        ds_sst["dsstx_ANN"] = calc_dsstx(ts_resid)
        ds_sst["tropical_mean_sst_ANN"] = calc_tropical_mean_sst(ts_resid)
        ds_sst["iod_SON"] = calc_iod(ts_resid, normalized=True)
        ds_sst["tpi_pdv_ANN_10yr"] = calc_tpi_pdv(ts_resid)
        ds_sst["amv_ANN_10yr"] = calc_amv(ts_resid, subtract_global=True)
        ds_sst["amv_noglob_ANN_10yr"] = calc_amv(ts_resid, subtract_global=False)
        ds_sst["iob_MAM"] = calc_iob(ts_resid)
        ds_sst["amm_JJA"] = calc_amm(ts_resid)
        ds_sst["azm_JJA"] = calc_azm(ts_resid)
        ds_sst["nino34_sst_DJF_raw"] = calc_nino34_sst(ts)
        ds_sst["iod_SON_raw"] = calc_iod(ts, normalized=True)
        ds_sst["dsstx_ANN_raw"] = calc_dsstx(ts)
        ds_sst["tpi_pdv_ANN_10yr_raw"] = calc_tpi_pdv(ts)
        ds_sst["amv_ANN_10yr_raw"] = calc_amv(ts, subtract_global=True)
        ds_sst["amv_noglob_ANN_10yr_raw"] = calc_amv(ts, subtract_global=False)
        ds_sst["iob_MAM_raw"] = calc_iob(ts)
        ds_sst["amm_JJA_raw"] = calc_amm(ts)
        ds_sst["azm_JJA_raw"] = calc_azm(ts)
        forced_trop = clean_index(calc_tropical_mean_sst(ts_forced), "forced_tropical_mean_sst_ANN")
        ds_sst["forced_tropical_mean_sst_ANN"] = forced_trop
        result_parts.append(ds_sst)

    if len(pr_files) >= args.min_members:
        log(f"{model}: building PR with {len(pr_files)} members")
        pr = build_model_member_dataset(pr_files, "pr", args)

        # Precipitation input variable:
        #   pr is used for the ENSO precipitation Niño3.4 index.
        # Forced-response treatment:
        #   pr_resid = pr - ensemble_mean(pr).
        #   nino34_pr_DJF uses residual precipitation; nino34_pr_DJF_raw uses raw pr.
        _, pr_resid = forced_response_and_residual(pr)
        ds_pr = xr.Dataset()
        ds_pr["nino34_pr_DJF"] = calc_precip_nino34(pr_resid)
        ds_pr["nino34_pr_DJF_raw"] = calc_precip_nino34(pr)
        result_parts.append(ds_pr)

    if len(psl_files) >= args.min_members:
        log(f"{model}: building PSL with {len(psl_files)} members")
        psl = build_model_member_dataset(psl_files, "psl", args)

        # Sea-level pressure input variable:
        #   psl is used for NAM, SAM, and NAO.
        # Forced-response treatment:
        #   psl_resid = psl - ensemble_mean(psl).
        #   Both residual and raw annular/NAO indices are saved because the email asks
        #   to check whether the forced component should be removed for these indices.
        _, psl_resid = forced_response_and_residual(psl)
        ds_ann = xr.Dataset()
        ds_ann["nam_DJF_resid"] = calc_nam_index(psl_resid, "DJF", "resid")
        ds_ann["nam_JJA_resid"] = calc_nam_index(psl_resid, "JJA", "resid")
        ds_ann["sam_DJF_resid"] = calc_sam_index(psl_resid, "DJF", "resid")
        ds_ann["sam_JJA_resid"] = calc_sam_index(psl_resid, "JJA", "resid")
        ds_ann["nao_DJF_resid"] = calc_nao_eof(psl_resid, "DJF", "resid")
        ds_ann["nao_JJA_resid"] = calc_nao_eof(psl_resid, "JJA", "resid")
        ds_ann["nam_DJF_raw"] = calc_nam_index(psl, "DJF", "raw")
        ds_ann["nam_JJA_raw"] = calc_nam_index(psl, "JJA", "raw")
        ds_ann["sam_DJF_raw"] = calc_sam_index(psl, "DJF", "raw")
        ds_ann["sam_JJA_raw"] = calc_sam_index(psl, "JJA", "raw")
        ds_ann["nao_DJF_raw"] = calc_nao_eof(psl, "DJF", "raw")
        ds_ann["nao_JJA_raw"] = calc_nao_eof(psl, "JJA", "raw")
        result_parts.append(ds_ann)

    if not result_parts:
        raise RuntimeError(f"No indices computed for {model}")

    result = xr.merge(result_parts, compat="override", join="outer")
    result.attrs["model"] = model
    result.attrs["scenario"] = args.scenario
    result.attrs["source_dir_atmosphere"] = str(args.base_dir_atm)
    result.attrs["source_dir_ocean"] = str(args.base_dir_ocean)
    result.attrs["baseline"] = "1991-2020"
    result.attrs["full_period"] = f"{args.full_start} to {args.full_end}"
    result.attrs["member_selection"] = f"Only models with nb members >= {args.min_members} retained."
    result.attrs["forced_response"] = (
        "Residual indices remove the model/scenario ensemble-mean monthly field. "
        "Raw variants with suffix _raw are also saved."
    )
    result.attrs["version"] = "v13 annotated-no-trend"

    log(f"{model}: saving output")
    safe_to_netcdf(result, out_index, direct_write=args.direct_write)


    return "finished"


def parse_args():
    p = argparse.ArgumentParser(description="Compute AR7 MOV indices v13 annotated no-trend.")
    p.add_argument("--local-root", default=LOCAL_MODEL_FIRST_ROOT_DEFAULT)
    p.add_argument("--base-dir-atm", default="unused_local_model_first")
    p.add_argument("--base-dir-ocean", default="unused_local_model_first")
    p.add_argument("--outdir", default="/workspace1/CMIP6_LME/AR7_indices_output_local")
    p.add_argument("--scenario", default=None, choices=["ssp585", "ssp370", "ssp245", "rcp85"])
    p.add_argument("--scenario-priority", nargs="+", default=["ssp585", "ssp370", "ssp245", "rcp85"])
    p.add_argument("--model", default=None)
    p.add_argument("--min-members", type=int, default=6)
    p.add_argument("--stop-after-one", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--full-start", default="1850-01-01")
    p.add_argument("--full-end", default="2100-12-31")
    p.add_argument("--time-chunk", type=int, default=24)
    p.add_argument("--direct-write", action="store_true", default=True)
    p.add_argument("--no-direct-write", dest="direct_write", action="store_false")
    p.add_argument("--sst-candidates", nargs="+", default=SST_CANDIDATES_DEFAULT)
    p.add_argument("--use-common-members-when-possible", action="store_true", default=True)
    p.add_argument("--no-common-members", dest="use_common_members_when_possible", action="store_false")
    args = p.parse_args()
    if args.scenario is None:
        args.scenario = args.scenario_priority[0]
    return args


def print_config(args):
    section("Configuration")
    for k, v in vars(args).items():
        log(f"{k} = {v}")


def main():
    args = parse_args()
    print_config(args)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    candidates = find_candidate_models(args)
    log(f"Candidate models: {candidates}")

    finished = []
    failed = []
    skipped = []

    for model in candidates:
        try:
            status = process_model(model, args)
            if status == "skipped":
                skipped.append(model)
                log(f"SKIPPED MODEL: {model}")
            else:
                finished.append(model)
                log(f"FINISHED MODEL: {model}")
        except Exception as e:
            failed.append(model)
            log(f"FAILED MODEL: {model}: {repr(e)}")
            traceback.print_exc()

        if args.stop_after_one and (finished or skipped or failed):
            log("STOP_AFTER_ONE requested. Stopping.")
            break

    summary_path = Path(args.outdir) / f"run_summary_{args.scenario}.txt"
    with open(summary_path, "w") as f:
        f.write("Finished models:\n")
        for m in finished:
            f.write(f"{m}\n")
        f.write("\nSkipped models:\n")
        for m in skipped:
            f.write(f"{m}\n")
        f.write("\nFailed models:\n")
        for m in failed:
            f.write(f"{m}\n")

    section("Run summary")
    log(f"Finished: {finished}")
    log(f"Skipped: {skipped}")
    log(f"Failed: {failed}")
    log(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
