# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:30:30 2023

@author: nkk
"""
import xarray as xr
import rioxarray as rxr
import numpy as np
import geopandas as gpd

y_labels = {
    1: "Corn",
    5: "Soybeans",
    28: "Oats",
    36: "Alfafa",
    37: "Other Hay/Non Alfalfa",
    176: "Grassland/Pasture",
    205: "Triticale",
}


def get_df(year=2018, path='data/CSB'):
    year_two_digits = str(year)[-2:]
    df = gpd.read_file(f"{path}/{year}.gpkg")
    df = df[[f"R{year_two_digits}", "geometry"]].to_crs(3857)
    df.geometry = df.geometry.buffer(-10)
    df = df[df.area > 10000]  # at least 1 hectare
    
    return df
def X_year(
    year=2018,
    to_numpy=True,
    return_feature_index=True,
    end_datetime="10-15",
    add_ndvi=True,
):
    ds = xr.open_mfdataset(f"data/eds/{year}/*.nc")
    ds = ds.sel(time=slice(ds.time[0], f"{year}-{end_datetime}"))
    if add_ndvi:
        ds["ndvi"] = (ds["nir"] - ds["red"]) / (ds["nir"] + ds["red"])

    if return_feature_index:
        yidx = ds.feature.data
    if to_numpy:
        ds = (
            ds.to_array(dim="band")
            .transpose("feature", "time", "band", "stats")
            .sel(stats="mean")
            .to_numpy()
        )
    if return_feature_index:
        return ds, yidx
    return ds

def X_y(year, end_datetime="10-15", to_torch=False, path="data/CSB"):
    X, y_idx = X_year(year, end_datetime=end_datetime)
    if not to_torch:
        X = X.reshape(X.shape[0], -1)
    y = (
        gpd.read_file(f"{path}/{year}.gpkg")[f"R{str(year)[-2:]}"]
        .loc[y_idx]
        .to_numpy()
    )
    selected_features = np.in1d(y, list(y_labels.keys()))
    y = y[selected_features]
    X = X[selected_features, ...]
    if to_torch:
        to_drop = ~np.sum(np.isnan(X), axis=(1,2), dtype=bool)
        y = y[:,np.newaxis].repeat(X.shape[1],axis=1)
    else:
        to_drop = ~np.sum(np.isnan(X), axis=1, dtype=bool)

    y = y[to_drop]
    X = X[to_drop, ...]
    return X, y

def y_to_range(y):
    for idx,y_label in enumerate(list(y_labels.keys())):
        y = np.where(y==y_label,idx,y)
    return y