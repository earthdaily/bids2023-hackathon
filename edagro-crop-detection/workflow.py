# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 🛰️🌿Predict crops using EarthDaily simulated constellation
#
# The idea of this hackathon  is to use the **EarthDaily Simulated dataset** to predict the 2019 crops in the current season (as if we were july 1st).
# The dataset was pregenerated using the earthdaily python package and the code is available in this notebook.
#
# We have three different years (2018, 2019 and 2020) and our workflow train from a specific year (2018 or 2020) only on spectral bands to predict an independent year (2019 being Nebraska's wettest summer). We challenge you to have the maximum accuracy to **predict year 2019** by having only **data up to july 1st** (so a month and a half of data, from may 15th to july 1st).
#
# ## 🎯 Goal
# Have the **highest accuracy to predict soybeans and corn for year 2019**. This can be done by enhancing the training of the algorithms, by adding new features (weather, SAR...) or anything else except giving the ground truth for year 2019 of course ! 

# %% [markdown]
# ## Independent / local installation : using conda/mamba environnement
#
# Download mamba : https://github.com/conda-forge/miniforge#mambaforge.
#
# If you're using powershell copy/paste this to have mamba commands : ```powershell -ExecutionPolicy ByPass -NoExit -Command "& 'C:\Users\$Env:UserName\AppData\Local\mambaforge\condabin\mamba.bat' init powershell"```
#
# Then create the bids23 environment : `mamba env update --name bids23 --file requirements.yml`.
#
# Then run the notebook : `mamba activate bids23` and `jupyter notebook`.

# %% [markdown]
# ## Import and init env

# %%
# To run if you have some gdal missing GDAL_DATA warning
import os

if os.path.exists("/teamspace/studios/this_studio/eda-bids-hackathon-prep/"):
    os.chdir("/teamspace/studios/this_studio/eda-bids-hackathon-prep/edagro-crop-detection")
if os.environ.get('GDAL_DATA') is None: os.environ["CONDA_PREFIX"] + r"\Library\share\gdal"
if os.environ.get('PROJ_LIB') is None: os.environ["CONDA_PREFIX"] + r"\Library\share\proj"

# %%
from matplotlib import pyplot as plt

from earthdaily import (
    earthdatastore,
)  # if you consider to generate the dataset, warning it takes about 1 or 2 hours.
from sklearn import metrics
import numpy as np
import xarray as xr
import rioxarray as rxr
import pandas as pd
import geopandas as gpd
import utils  # specific codes for this hackathon

# %% [markdown]
# ## Generate dataset from the earthdaily simulation
#
# The **EarthDaily Simulated dataset** is a simulation using various Sensors (Venus, Sentinel-2, Landsat, Modis) in order to have a cloudless version of what we will be able to generate when **EarthDaily constellation** will be here : 5m spatial resolution on most of VNIR bands, daily revisit, radiometric scientific grade quality...
# As it is cloudless, it is an Analytics Ready Product.
#
# ### Ground truth
# The crops dataset (polygon and label) for each year is the extent extract from the [Crop Sequence Boundaries](https://www.nass.usda.gov/Research_and_Science/Crop-Sequence-Boundaries/index.php) (CSB). You can got this data directly via `utils.crops_layer(2020)` for year 2020.


# %%
# To generate the dataset for a specific year
year = 2020
df = utils.crops_layer(year)
bbox = df.to_crs(4326).total_bounds.tolist()
# Add crops names to the dataframe
crops_df = pd.DataFrame(
    list(utils.y_labels.values()),
    list(utils.y_labels.keys()),
    columns=["crop"],
)
df = df.merge(crops_df, how="left", left_on="R20", right_index=True)

# %% [markdown]
# Explore the dataset for this year

# %%
df.explore(column="crop", popup=True, tiles="CartoDB positron", cmap="Set1")

# %% [markdown]
# Here we generate the training data of the year defined earlier (warning it takes around 30 minutes per year).
#
# As it takes about 2 hours to generate the dataset (for the 3 years), **you provide you a pregenerated dataset for the Hackathon**. But this code has been used to generated the data you'll have.

# %%
generate_dataset = False
days_interval = 5  # one information every x days (default=5)

if generate_dataset:
    eds = earthdatastore.Auth()
    items = eds.search(
        "earthdaily-simulated-cloudless-l2a-cog-edagro",
        bbox=bbox,
        datetime=[f"{year}-05-15", f"{year}-10-15"], # it excludes 1st july
        prefer_alternate="download",
        query=dict(instruments={"contains": "vnir"}),
    )
    
    # get only one item every 5 days  (days_interval)
    items = [items[i] for i in np.arange(0, len(items), days_interval)]
    datacube_sr = earthdatastore.datacube(
        items,
        bbox=bbox,
        assets={
            "image_file_B": "blue",
            "image_file_G": "green",
            "image_file_Y": "yellow",
            "image_file_R": "red",
            "image_file_RE1": "redege1",
            "image_file_RE2": "redege2",
            "image_file_RE3": "redege3",
            "image_file_NIR": "nir",
        },
    )

    for data_var in datacube_sr:
        break
        data_var_nc = f"data/eds/{year}/{data_var}.nc"
        if os.path.exists(data_var_nc):
            continue
        os.makedirs(f"data/eds/{year}", exist_ok=True)
        ds_stats = earthdatastore.cube_utils.zonal_stats_numpy(
            datacube_sr[[data_var]], df
        )
        ds_stats.to_netcdf(data_var_nc)


# %% [markdown]
# ## Plot time series per polygon
# `utils.X_year` function returns the mean values for each polygon and per date.
#
# Here we plot polygon from index 500, 1000, 1500 and 2000.

# %%
ds = utils.X_year(year, to_numpy=False, return_feature_index=False)
for feature_idx in [500, 1000, 1500, 2000]:
    ds["ndvi"].isel(feature=feature_idx, stats=0).plot()
    plt.title(utils.y_labels[df.iloc[feature_idx][f"R{str(year)[2:]}"]])
    plt.show()

# %% [markdown]
# ## Available dates

# %%
print(f'These are available dates for year {year} : ')
print(ds.time.dt.strftime('%Y-%m-%d').data)

# %% [markdown]
# # Generate training/testing data
# We have data from **may 15th to october 15th**. But in order to predict the crop during the season, we chose an `end_datetime`, here for the **july 1st**.

# %%
# We suppose we have data only up to july 1st.
end_datetime = "07-01"  # july 1st
# you can go up to 10-15 (october 15th)
X_18, y_18, y_18_indices = utils.X_y(2018, end_datetime=end_datetime, return_indices=True)
X_19, y_19, y_19_indices = utils.X_y(2019, end_datetime=end_datetime, return_indices=True)
X_20, y_20, y_20_indices = utils.X_y(2020, end_datetime=end_datetime, return_indices=True)

# %% [markdown]
# Here we plot all the NDVI time series for a given year for a specific crop.

# %%
plt.title("Soy (NDVI)")
soy = np.in1d(y_19, 1)
plt.plot(
    X_19[soy, :][:, np.arange(8, X_19.shape[1], 9)].T, alpha=0.05, c="green"
)
plt.show()


# %%
plt.title("Corn (NDVI)")
corn = np.in1d(y_19, 5)
plt.plot(
    X_19[corn, :][:, np.arange(8, X_19.shape[1], 9)].T, alpha=0.05, c="gold"
)
plt.show()

# %%
plt.title("Meadow (NDVI)")
meadow = np.in1d(y_19, 176)
plt.plot(
    X_19[meadow, :][:, np.arange(8, X_19.shape[1], 9)].T, alpha=0.2, c="C2"
)
plt.show()

# %% [markdown]
# As you can see there can be a high standard deviation for the same crop among fields. But we clearly see that the soy crop starts his growing phase earlier than the corn. Also, as the CSB is a predicted model, we can guess that it has some mislabelling where some crops don't have any growing phase. 

# %% [markdown]
# # Machine Learning
# We use Random Forest and XGBoost to train with one or two years, and to **predict on year 2019**.

# %%
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()  # default parameters

# %% [markdown]
# Or use xgboost

# %%
import xgboost as xgb

model = xgb.XGBClassifier()

# %%
# class are not following number (they are like 1,5,205)... Torch and xgb needs following numbers (0,1,2,3)
y_18 = utils.y_to_range(y_18)
y_19 = utils.y_to_range(y_19)
y_20 = utils.y_to_range(y_20)

# %%
# confusion matrix kwargs
cm_plot_kwargs = dict(
    display_labels=list(utils.y_labels.values()),
    cmap="Blues",
    xticks_rotation="vertical",
    colorbar=False,
)
# %%
model.fit(X_18, y_18)
y_pred = model.predict(X_19)
score = metrics.accuracy_score(y_19, y_pred)
print(f"Score when training with 2018 : {score}")
y_pred_19 = model.predict(X_19)
metrics.ConfusionMatrixDisplay.from_predictions(
    y_19, y_pred, **cm_plot_kwargs
)

# %% [markdown]
# The purpose of the hackathon is to identify the main crops in preseason, here **Corn** and **Soybeans**. So the main idea is to lower the confusion between these two classes.
# Here **1420 corn fields** have been predicted as **soybeans**, whereas only **37 soybeans** were predicted as **corn**. So here we overpredict the soybeans and underpredict the corn.

# %%
model.fit(X_20, y_20)
y_pred = model.predict(X_19)
score = metrics.accuracy_score(y_19, y_pred)
print(f"Score when training with 2020 : {score}")
metrics.ConfusionMatrixDisplay.from_predictions(
    y_19, y_pred, **cm_plot_kwargs
)


# %%
model.fit(np.vstack((X_18, X_20)), np.hstack((y_18, y_20)))
y_pred = model.predict(X_19)
score = metrics.accuracy_score(y_19, y_pred)
print(f"Score when training with 2018 and 2020 : {score}")
metrics.ConfusionMatrixDisplay.from_predictions(
    y_19, y_pred, **cm_plot_kwargs
)


# %% [markdown]
# # Deep Learning (RNN) : ELECTS model
# ELECTS is a RNN algorithm which stands for : End-to-End Learned Early Classification of Time Series for In-Season Crop Type Mapping.
# This model takes as input an array using *n* dates and *n* dimensions per date. So if you want to add a feature (like SAR data), you need to add for the a feature for each of the *n* dates.
#
# You can see the ELECTS github page for more information : [ELECTS github](https://github.com/MarcCoru/elects/)


# %%
import elects

# We must define the number of bands
n_bands = 9  # (8 VNIR) + NDVI

# %%
train_ds = utils.torch_dataset(X_18, y_18, n_bands=n_bands)
test_ds = utils.torch_dataset(X_19, y_19, n_bands=n_bands)

model = elects.train(
    train_ds,
    test_ds,
    n_classes=len(utils.y_labels),
    epochs=100,
    n_bands=n_bands,
)

# %%
y_pred = model.predict(utils.x_to_torch(X_19, n_bands))[2].detach().numpy()

score = metrics.accuracy_score(y_19, y_pred)
print(f"Score when training with 2018 : {score}")

metrics.ConfusionMatrixDisplay.from_predictions(
    y_19, y_pred, **cm_plot_kwargs
)

# %% [markdown]
# # Train with two years
#
# We want to train using 2 years in order to predict year 2019. We have two solutions, to resume from the previous model the training and just train with 2020, or to train a new model using 2018 and 2020 years at once.
# %%
resume = True
# if resume, just add 2020 and use previous trained model
if resume:
    train_ds = utils.torch_dataset(X_20, y_20, n_bands=n_bands)
else:
    train_ds = utils.torch_dataset(
        np.vstack((X_18, X_20)), np.hstack((y_18, y_20)), n_bands=n_bands
    )

model = elects.train(
    train_ds,
    test_ds,
    n_classes=len(utils.y_labels),
    epochs=200,  # add 100 epochs
    n_bands=n_bands,
    resume=resume,  # in order to start using previous training
)

y_pred = model.predict(utils.x_to_torch(X_19, n_bands))[2].detach().numpy()

score = metrics.accuracy_score(y_19, y_pred)
print(f"Score when training with 2018 and 2020 : {score}")

metrics.ConfusionMatrixDisplay.from_predictions(
    y_19, y_pred, **cm_plot_kwargs
)

# %% [markdown]
# # Add new features to your samples
#
# ## Add global feature (one identical feature for all samples)
#
# Let's suppose you find some global weather information (the same information for each of the feature) and you want to learn also with these informations.
# You can add them with the `utils.add_features_on_X`. If you want to add a local feature (one specific feature for each field), you need to modify the behavior the this function.
# %%
print(f"My 2018 dataset has shape of {X_18.shape}")
n_bands = 9
n_dates = int(X_18.shape[1] / n_bands)

# %%
# Now we can add for each date a new feature
X_18_enhanced = utils.add_features_on_X(
    X_18, features=np.random.randint(0, 100, n_dates), n_dates=n_dates
)  # here we add random values from 0 to 100
print(
    f"My new feature for the first date of the first sample is : {X_18_enhanced[0,9]}"
)

# %%
# And you can add as many features as you want, they just need to have the same length as the number of dates
X_18_enhanced = utils.add_features_on_X(
    X_18_enhanced,
    features=np.random.randint(0, 100, n_dates),
    n_dates=n_dates,
)

print(f"X_18_enhanced has now {X_18_enhanced.shape[1]/10} features per date")

# %% [markdown]
# ## Add Sentinel-1 features per each sample
#
# /!\ As ELECTS (Deep Learning part) requires a new feature at every time dimension, this approach will be mainly suitable only for the Machine Learning.
#  
#

# %%
eds = earthdatastore.Auth()

year = 2018
s1_datacube = eds.datacube(
    "sentinel-1-rtc",
    bbox=bbox,
    datetime=[f"{year}-05-15", f"{year}-07-01"])

s1_zonal_stats = earthdatastore.cube_utils.zonal_stats_numpy(s1_datacube, utils.crops_layer(year).loc[y_18_indices])
s1_numpy = s1_zonal_stats.sel(stats="mean").to_array().to_numpy().reshape(-1,s1_zonal_stats.feature.size).T
X_18_with_s1 = np.hstack((X_18,s1_numpy))
