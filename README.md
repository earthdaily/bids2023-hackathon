# bids-hackation
This repository contains scripts and notebooks for the EDA Hackathon at [BiDS 2023](https://www.bigdatafromspace2023.org/)

# Setup & authentication
In Colab, sagemaker etc we just need to `pip install -r requirements.txt`

If querying the EDA STAC API is necessary to have authentication setup via a `.env` file (or using env variables if you prefer). Copy and rename `.env.sample` to `.env` and enter your credentials - note these are gitignored and will not be pushed!

# Notebooks
### venus-denver-with-building-model.ipynb
This notebook demonstrates querying the EDA STAC API for a Venus chip using an ROI, saving RGB PNG and multispectral Geotiffs, and then using this imagery to train a simple model for building/not-building pixel level prediction. Note: running this notebook requires downloading the Colorado geojson file from [USBuildingFootprints](https://github.com/microsoft/USBuildingFootprints)

### train-eurosat.ipynb
This notebook demonstrates training a chip classifier on a Sentinel 2 dataset. Training will be slow on a CPU so use a GPU machine. Experiment with the model choice and hyperparameters to improve the model. Once you are satisfied with the model performance, inference on data from the EDA STAC API. If you are feeling adventurous, try training a model that demonstrates some performance on Venus imagery! Note: using the [wandb logger](https://wandb.ai/) is possible with a free account

TBD if we create a test dataset

BONUS: [Investigate channel significance](http://matpalm.com/blog/evolved_channel_selection/)

# License
TBC

