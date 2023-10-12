# bids-hackation
This repository contains scripts and notebooks for the EDA Hackathon at [BiDS 2023](https://www.bigdatafromspace2023.org/). Roughly speaking we have material more focussed on data access and visualisation under the STAC category, and material for machine learning (ML). Of course you can combine these, e.g. train a model, then inference on data via STAC to build cool applications!

# Setup & authentication
In Goolge Colab & AWS Sagemaker Studio you just need to `pip install -r requirements.txt`

If querying the EDA STAC API is necessary to have authentication setup via a `.env` file (or using env variables if you prefer). Copy and rename `.env.sample` to `.env` and enter your credentials - note these are gitignored and will not be pushed! These notebooks have not been tested with other STAC endpoints, but should work with only minor modifications if the data is formatted differently.

# Notebooks - STAC
### venus-git-over-LEFR3.ipynb
This notebook demonstrates querying the EDA STAC API for a Venus chip using an ROI and a period of 1 year. An RGB PNG is saved for each date there is imagery, and finally a video (gif) is generated, showing change over a mining site in Australia.

### venus-denver-with-building-model.ipynb
This notebook demonstrates querying the EDA STAC API for a Venus chip using an ROI, saving RGB PNG and multispectral Geotiffs, and then using this imagery to train a simple model for building/not-building pixel level prediction. Note: running this notebook requires downloading the Colorado geojson file from [USBuildingFootprints](https://github.com/microsoft/USBuildingFootprints)

# Notebooks - ML
### train-eurosat.ipynb
This notebook demonstrates training a classifier on a Sentinel 2 dataset called [EuroSAT](https://github.com/phelber/EuroSAT). Training will be slow on a CPU so use a GPU machine. Experiment with the choice of model, hyperparameters and pretrained weights to iachieve the best performance you can. Once you are satisfied with the model performance, inference on data from the EDA STAC API to create an interesting application. Note: using the [wandb logger](https://wandb.ai/) is possible with a free account

### predict-eurosat.ipynb
This notebook demonstrates how to use a model to make predictions. A prediction is performed on a single image (suitable for CPU and serveress inferencing) and also performed on a large test dataset in batch mode (suitable for GPU machines and longer running processes). A confusion matrix is plotted to allow inspection of the model performance.

# License
TBC

