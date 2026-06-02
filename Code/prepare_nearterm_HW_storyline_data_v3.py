#!/usr/bin/env python3
"""
Prepare near-term HW storyline map data on NCI.

Implements:
1. Compute annual global GSAT for simulations used for MOVs.
2. Compute mean GSAT over 2026-2045.
3. Remove each model/scenario ensemble mean to get GSAT internal variability anomaly.
4. Pool all model/member/scenario GSAT anomalies and select upper decile.
5. For TAS and PR, compute:
       internal_map = selected-member mean over 2026-2045 - model ensemble mean over 2026-2045
       forced_map   = model ensemble mean over 2026-2045 - model ensemble mean over 2004-2023
       final_map    = internal_map + forced_map
6. Save NetCDF outputs for plotting later.

Strict scenario selection:
    ssp585: historical+ssp585 or historical_ssp585
    ssp370: historical+ssp370 or historical_ssp370
    rcp85 : historical+rcp85  or historical_rcp85

Default output:
    /home/552/sd6705/near_term_HW_storyline_v1
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
            return m.group(0)
    return os.path.basename(os.path.dirname(path))


def normalize_member(member):
    member = str(member)
    member = re.sub(r"(r\d+i\d+p\d+)f\d+", r"\1", member)
    member = re.sub(r"(\d{3,4}\.\d{3}i\d+p\d+)f\d+", r"\1", member)
    return member


def member_from_path(path):
    return normalize_member(raw_member_from_path(path))


def filename_lower(path):
    return os.path.basename(path).lower()


def file_is_pure_historical(path):
    b = filename_lower(path)
    if "historical" not in b:
        return False
    return not any(s in b for s in KNOWN_SCENARIOS)


def file_is_combined_historical_scenario(path, scenario):
    b = filename_lower(path)
    scenario = scenario.lower()
    return ("historical" in b) and (scenario in b)


def file_is_future_scenario(path, scenario):
    b = filename_lower(path)
    scenario = scenario.lower()
    return (scenario in b) and not file_is_combined_historical_scenario(path, scenario)


def select_files_for_scenario(files, scenario):
    files = sorted(files)

    combined = [f for f in files if file_is_combined_historical_scenario(f, scenario)]
    if len(combined) > 0:
        return sorted(set(combined)), f"combined_historical_{scenario}"

    historical = [f for f in files if file_is_pure_historical(f)]
    future = [f for f in files if file_is_future_scenario(f, scenario)]

    if len(future) > 0:
        return sorted(set(historical + future)), f"historical_plus_{scenario}"

    return [], f"missing_{scenario}"


def collect_member_files(base_dir, varname, model, scenario):
    model_dir = Path(base_dir) / varname / model
    if not model_dir.exists():
        return {}

    files = sorted(glob.glob(str(model_dir / "**" / "*.nc"), recursive=True))
    raw = {}

    for f in files:
        member = member_from_path(f)
        raw.setdefault(member, []).append(f)

    out = {}
    for member, flist in raw.items():
        selected, _ = select_files_for_scenario(flist, scenario)
        if selected:
            out[member] = selected

    return out


def get_index_dir(args, scenario):
    mapping = {
        "ssp585": args.index_dir_ssp585,
        "ssp370": args.index_dir_ssp370,
        "rcp85": args.index_dir_rcp85,
    }
    if scenario not in mapping:
        raise ValueError(f"No index directory for scenario {scenario}")
    return Path(mapping[scenario])


def list_mov_model_members(args):
    rows = []

    for scenario in args.scenarios:
        index_dir = get_index_dir(args, scenario)

        if not index_dir.exists():
            log(f"Missing index dir for {scenario}: {index_dir}")
            continue

        files = sorted(index_dir.glob("*_AR7_indices.nc"))
        log(f"{scenario}: {len(files)} MOV index files in {index_dir}")

        for f in files:
            model = f.name.replace("_AR7_indices.nc", "")

            try:
                ds = xr.open_dataset(f)
            except Exception as e:
                log(f"Cannot open {f}: {repr(e)}")
                continue

            if "member" not in ds.dims:
                ds.close()
                continue

            for member in ds["member"].values:
                rows.append({
                    "scenario_filter": scenario,
                    "model": model,
                    "member": normalize_member(str(member)),
                    "index_file": str(f),
                })

            ds.close()

    df = pd.DataFrame(rows).drop_duplicates()

    if df.empty:
        raise RuntimeError("No MOV model/member inventory found.")

    log(f"MOV inventory rows: {len(df)}")
    log(f"MOV inventory models: {df['model'].nunique()}")

    return df


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


def fix_lon(da):
    lat_name, lon_name = get_lat_lon_names(da)
    lon = da[lon_name]

    if lon.ndim != 1:
        raise ValueError(f"{da.name}: longitude is not 1D; regular grid assumed.")

    if float(lon.min()) < 0:
        da = da.assign_coords({lon_name: lon % 360})
        da = da.sortby(lon_name)

    return da


def open_member_variable(files, varname, args):
    log(f"Opening {varname}: {len(files)} files")
    log(f"  first: {files[0]}")
    log(f"  last : {files[-1]}")

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        use_cftime=True,
        chunks={"time": args.time_chunk},
        parallel=True,
    )

    if varname in ds.data_vars:
        da = ds[varname]
    else:
        dvs = list(ds.data_vars)
        if len(dvs) == 1:
            log(f"  {varname} not found; using {dvs[0]}")
            da = ds[dvs[0]]
            da.name = varname
        else:
            raise ValueError(f"{varname} not found. data_vars={dvs}")

    full_start = getattr(args, "full_start", "1850-01-01")
    full_end = getattr(args, "full_end", "2100-12-31")
    da = da.sel(time=slice(full_start, full_end))
    if da.sizes.get("time", 0) == 0:
        raise RuntimeError(f"{varname}: no data in {full_start} to {full_end}")

    da = fix_lon(da)
    return da


def annual_mean(da):
    return da.groupby("time.year").mean("time")


def period_mean(da, y0, y1):
    if "year" in da.dims:
        return da.sel(year=slice(y0, y1)).mean("year")
    years = da["time"].dt.year
    return da.where((years >= y0) & (years <= y1), drop=True).mean("time")


def global_mean_monthly(da):
    lat_name, lon_name = get_lat_lon_names(da)
    weights = np.cos(np.deg2rad(da[lat_name]))
    out = da.weighted(weights).mean(dim=[lat_name, lon_name])
    out.name = "GSAT"
    return out


def convert_pr_to_mmday(da):
    out = da * 86400.0
    out.attrs["units"] = "mm day-1"
    return out


def interp_to_target_grid(da, args):
    lat_name, lon_name = get_lat_lon_names(da)

    if da[lat_name][0] > da[lat_name][-1]:
        da = da.sortby(lat_name)
    if da[lon_name][0] > da[lon_name][-1]:
        da = da.sortby(lon_name)

    target_lat = np.arange(-90, 90 + args.grid_res, args.grid_res)
    target_lon = np.arange(0, 360, args.grid_res)

    out = da.interp(
        {lat_name: target_lat, lon_name: target_lon},
        kwargs={"fill_value": np.nan},
    )

    rename = {}
    if lat_name != "lat":
        rename[lat_name] = "lat"
    if lon_name != "lon":
        rename[lon_name] = "lon"
    if rename:
        out = out.rename(rename)

    return out


def compute_gsat_table(args, inventory):
    section("Step 1: compute GSAT internal variability anomalies")

    rows = []

    for (scenario, model), sub in inventory.groupby(["scenario_filter", "model"]):
        subsection(f"GSAT {scenario} {model}")

        tas_files = collect_member_files(args.base_dir_atm, "tas", model, scenario)
        if not tas_files:
            log(f"No tas files for {model} {scenario}")
            continue

        model_rows = []

        for member in sorted(set(sub["member"].values)):
            if member not in tas_files:
                log(f"  missing tas for MOV member {member}")
                continue

            try:
                tas = open_member_variable(tas_files[member], "tas", args)
                gsat = annual_mean(global_mean_monthly(tas))
                near = period_mean(gsat, args.near_start, args.near_end).load()
                ref = period_mean(gsat, args.ref_start, args.ref_end).load()

                near_val = float(near.values)
                ref_val = float(ref.values)

                model_rows.append({
                    "scenario_filter": scenario,
                    "model": model,
                    "member": member,
                    "GSAT_2026_2045": near_val,
                    "GSAT_2004_2023": ref_val,
                    "GSAT_change_2026_2045_minus_2004_2023": near_val - ref_val,
                })

            except Exception as e:
                log(f"FAILED GSAT {model} {member} {scenario}: {repr(e)}")
                traceback.print_exc()
                continue

        if not model_rows:
            continue

        df_model = pd.DataFrame(model_rows)
        ens_mean = df_model["GSAT_2026_2045"].mean()
        df_model["model_ensemble_mean_GSAT_2026_2045"] = ens_mean
        df_model["GSAT_internal_anom_2026_2045"] = df_model["GSAT_2026_2045"] - ens_mean

        rows.extend(df_model.to_dict("records"))

        log(f"{model} {scenario}: n={len(df_model)}, ensmean={ens_mean:.3f}")

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("GSAT table is empty.")

    threshold = df["GSAT_internal_anom_2026_2045"].quantile(args.upper_quantile)
    df["upper_decile"] = df["GSAT_internal_anom_2026_2045"] >= threshold
    df["upper_decile_threshold"] = threshold

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df.to_csv(outdir / "GSAT_internal_anomaly_2026_2045.csv", index=False)
    df[df["upper_decile"]].to_csv(outdir / "selected_upper_decile_members.csv", index=False)

    log(f"Upper quantile = {args.upper_quantile}")
    log(f"Upper-decile threshold = {threshold:.4f} K")
    log(f"Selected = {int(df['upper_decile'].sum())} / {len(df)}")

    return df


def compute_model_variable_maps(args, scenario, model, mov_members, selected_members, varname):
    files_all = collect_member_files(args.base_dir_atm, varname, model, scenario)
    if not files_all:
        log(f"No {varname} files for {model} {scenario}")
        return None, None

    mov_members = sorted(set(mov_members))
    selected_members = sorted(set(selected_members))

    available_mov = [m for m in mov_members if m in files_all]
    available_sel = [m for m in selected_members if m in files_all]

    if len(available_mov) < args.min_members:
        log(f"Skip {model} {scenario} {varname}: only {len(available_mov)} MOV members")
        return None, None

    if len(available_sel) == 0:
        log(f"Skip {model} {scenario} {varname}: no selected members")
        return None, None

    log(f"{model} {scenario} {varname}: MOV members={len(available_mov)}, selected={len(available_sel)}")

    near_maps = []
    ref_maps = []
    sel_near_maps = []

    for member in available_mov:
        try:
            da = open_member_variable(files_all[member], varname, args)

            if varname == "pr":
                da = convert_pr_to_mmday(da)

            near = period_mean(da, args.near_start, args.near_end).load()
            ref = period_mean(da, args.ref_start, args.ref_end).load()

            near_maps.append(near.expand_dims(member=[member]))
            ref_maps.append(ref.expand_dims(member=[member]))

            if member in available_sel:
                sel_near_maps.append(near.expand_dims(member=[member]))

        except Exception as e:
            log(f"FAILED map input {model} {member} {scenario} {varname}: {repr(e)}")
            traceback.print_exc()
            continue

    if len(near_maps) < args.min_members or len(sel_near_maps) == 0:
        return None, None

    near_all = xr.concat(near_maps, dim="member")
    ref_all = xr.concat(ref_maps, dim="member")
    near_sel = xr.concat(sel_near_maps, dim="member")

    ens_near = near_all.mean("member")
    ens_ref = ref_all.mean("member")

    internal = near_sel.mean("member") - ens_near
    forced = ens_near - ens_ref
    final = internal + forced

    internal_i = interp_to_target_grid(internal, args)
    forced_i = interp_to_target_grid(forced, args)
    final_i = interp_to_target_grid(final, args)
    ens_ref_i = interp_to_target_grid(ens_ref, args)

    ds = xr.Dataset()

    if varname == "tas":
        ds["tas_HW_internal_anom"] = internal_i
        ds["tas_forced_change_2026_2045_minus_2004_2023"] = forced_i
        ds["tas_HW_final"] = final_i
        ds["tas_ref_2004_2023"] = ens_ref_i
        for v in ds.data_vars:
            ds[v].attrs["units"] = "K"

    elif varname == "pr":
        ds["pr_HW_internal_anom_mmday"] = internal_i
        ds["pr_forced_change_mmday"] = forced_i
        ds["pr_HW_final_mmday"] = final_i
        ds["pr_ref_2004_2023_mmday"] = ens_ref_i
        ds["pr_HW_internal_anom_percent"] = 100.0 * ds["pr_HW_internal_anom_mmday"] / ds["pr_ref_2004_2023_mmday"]
        ds["pr_forced_change_percent"] = 100.0 * ds["pr_forced_change_mmday"] / ds["pr_ref_2004_2023_mmday"]
        ds["pr_HW_final_percent"] = 100.0 * ds["pr_HW_final_mmday"] / ds["pr_ref_2004_2023_mmday"]

        for v in ds.data_vars:
            ds[v].attrs["units"] = "%" if v.endswith("percent") else "mm day-1"

    ds.attrs["model"] = model
    ds.attrs["scenario_filter"] = scenario
    ds.attrs["variable"] = varname
    ds.attrs["near_period"] = f"{args.near_start}-{args.near_end}"
    ds.attrs["ref_period"] = f"{args.ref_start}-{args.ref_end}"
    ds.attrs["n_mov_members"] = len(available_mov)
    ds.attrs["n_selected_members"] = len(available_sel)
    ds.attrs["definition"] = (
        "HW_final = selected upper-decile internal anomaly over 2026-2045 "
        "+ model ensemble forced change from 2004-2023 to 2026-2045. "
        "Selection is based on upper-decile GSAT internal anomaly."
    )

    meta = {
        "scenario_filter": scenario,
        "model": model,
        "variable": varname,
        "n_mov_members": len(available_mov),
        "n_selected_members": len(available_sel),
        "selected_members": ",".join(available_sel),
    }

    return ds, meta



def clean_dataset_for_sample_concat(ds):
    """
    Remove non-dimension scalar coordinates such as 'height' before concatenating
    model/scenario datasets.

    Some tas files carry a scalar coordinate like height=2m while pr does not.
    xarray.concat with coords='different' then fails because 'height' is not
    present in all datasets. For the final map output, lat/lon are the only
    required coordinates.
    """
    drop_coords = []
    for cname in list(ds.coords):
        if cname not in ds.dims and cname not in ["lat", "lon"]:
            drop_coords.append(cname)

    if drop_coords:
        ds = ds.drop_vars(drop_coords, errors="ignore")

    return ds


def compute_maps(args, inventory, gsat_df):
    section("Step 2: compute TAS and PR maps")

    outdir = Path(args.outdir)
    map_dir = outdir / "model_scenario_maps"
    map_dir.mkdir(parents=True, exist_ok=True)

    selected_df = gsat_df[gsat_df["upper_decile"]].copy()

    all_maps = []
    metadata = []

    for (scenario, model), sub_inv in inventory.groupby(["scenario_filter", "model"]):
        subsection(f"Maps {scenario} {model}")

        mov_members = sorted(set(sub_inv["member"].values))
        sub_sel = selected_df[
            (selected_df["scenario_filter"] == scenario)
            & (selected_df["model"] == model)
        ]
        selected_members = sorted(set(sub_sel["member"].values))

        if len(selected_members) == 0:
            log(f"No selected members for {model} {scenario}; skip.")
            continue

        ds_out = xr.Dataset()
        model_meta = {
            "scenario_filter": scenario,
            "model": model,
            "n_selected_members": len(selected_members),
            "selected_members": ",".join(selected_members),
        }

        for varname in ["tas", "pr"]:
            ds_var, meta_var = compute_model_variable_maps(
                args, scenario, model, mov_members, selected_members, varname
            )

            if ds_var is None:
                continue

            ds_out = xr.merge([ds_out, ds_var], compat="override")

            for k, v in meta_var.items():
                model_meta[f"{varname}_{k}"] = v

        if len(ds_out.data_vars) == 0:
            continue

        ds_out = clean_dataset_for_sample_concat(ds_out)

        ds_out.attrs["model"] = model
        ds_out.attrs["scenario_filter"] = scenario
        ds_out.attrs["near_period"] = f"{args.near_start}-{args.near_end}"
        ds_out.attrs["ref_period"] = f"{args.ref_start}-{args.ref_end}"

        out_path = map_dir / f"{model}_{scenario}_HW_storyline_maps.nc"
        encoding = {v: {"zlib": True, "complevel": 4} for v in ds_out.data_vars}
        ds_out.to_netcdf(out_path, encoding=encoding)

        log(f"Wrote {out_path}")

        all_maps.append(ds_out.expand_dims(sample=[f"{model}_{scenario}"]))
        model_meta["file"] = str(out_path)
        metadata.append(model_meta)

    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(outdir / "model_scenario_map_metadata.csv", index=False)

    if not all_maps:
        log("No model maps produced.")
        return

    ds_all = xr.concat(all_maps, dim="sample", coords="minimal", compat="override", join="outer")
    ds_mean = ds_all.mean("sample", skipna=True)

    enc_all = {v: {"zlib": True, "complevel": 4} for v in ds_all.data_vars}
    enc_mean = {v: {"zlib": True, "complevel": 4} for v in ds_mean.data_vars}

    ds_all.to_netcdf(outdir / "HW_all_model_scenario_maps_on_common_grid.nc", encoding=enc_all)
    ds_mean.to_netcdf(outdir / "HW_multimodel_mean_maps.nc", encoding=enc_mean)

    log(f"Wrote {outdir / 'HW_all_model_scenario_maps_on_common_grid.nc'}")
    log(f"Wrote {outdir / 'HW_multimodel_mean_maps.nc'}")


def parse_args():
    p = argparse.ArgumentParser(description="Prepare near-term HW storyline map data.")

    p.add_argument("--base-dir-atm", default="/g/data/su28/MMLEAv2/atmosphere/monthly")
    p.add_argument("--outdir", default="/home/552/sd6705/near_term_HW_storyline_v1")
    p.add_argument("--scenarios", nargs="+", default=["ssp585", "ssp370", "rcp85"])

    p.add_argument("--index-dir-ssp585", default="/home/552/sd6705/AR7_indices_output_v6_ssp585")
    p.add_argument("--index-dir-ssp370", default="/home/552/sd6705/AR7_indices_output_v6_ssp370")
    p.add_argument("--index-dir-rcp85", default="/home/552/sd6705/AR7_indices_output_v6_rcp85")

    p.add_argument("--full-start", default="1850-01-01")
    p.add_argument("--full-end", default="2100-12-31")

    p.add_argument("--near-start", type=int, default=2026)
    p.add_argument("--near-end", type=int, default=2045)
    p.add_argument("--ref-start", type=int, default=2004)
    p.add_argument("--ref-end", type=int, default=2023)

    p.add_argument("--upper-quantile", type=float, default=0.9)
    p.add_argument("--min-members", type=int, default=6)
    p.add_argument("--time-chunk", type=int, default=120)
    p.add_argument("--grid-res", type=float, default=2.5)

    return p.parse_args()


def main():
    args = parse_args()

    section("Configuration")
    for k, v in vars(args).items():
        log(f"{k} = {v}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    inventory = list_mov_model_members(args)
    inventory.to_csv(outdir / "MOV_simulations_used_inventory.csv", index=False)
    log(f"Wrote {outdir / 'MOV_simulations_used_inventory.csv'}")

    gsat_df = compute_gsat_table(args, inventory)

    compute_maps(args, inventory, gsat_df)

    section("Done")
    log(f"Output directory: {args.outdir}")


if __name__ == "__main__":
    main()
