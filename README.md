<div align="center">
  <p>
    <a href="https://pages.earthdaily.com/hackathon">
        <img src="banner.png" width="1100">
    </a>
</p>
</div>

This repository contains scripts and notebooks for the [EarthDaily Analytics Hackathon](https://pages.earthdaily.com/hackathon) at [BiDS 2023](https://www.bigdatafromspace2023.org/). There is material for people who want to focus on data access and data visualisation under the `STAC` category, and material for people who want to explore machine learning (`ML`). Of course you can combine these, for example by first training a model and then processing imagery generated via STAC to build cool analytics applications!

# Setup
To run the notebooks without leaving Github or setting up a python environment you can use a [codespace](https://github.com/features/codespaces). This approach will be demonstrated. Note however that codespaces do not provide a GPU (to the best of our knowledge)

In [Goolge Colab](https://research.google.com/colaboratory/), [AWS Sagemaker Studio Lab](https://studiolab.sagemaker.aws/) or on [lightning.ai](https://lightning.ai/) you just need to git clone this repository and then `pip install -r requirements.txt`

# Authentication
If querying the EarthDaily Analytics STAC API it is necessary to have authentication setup via a `.env` file (or by using environment variables directly if you prefer). Copy and rename `.env.sample` to `.env` and enter your credentials - note these are gitignored and will not be pushed to Github! Note that these notebooks have not been tested with other STAC endpoints, but they should work, potentially with minor modifications if the data is formatted differently.

# Notebooks - STAC
### venus-git-over-LEFR3.ipynb
This notebook demonstrates querying the EDA STAC API for a Venus chip using a region of interest (ROI) and a period of 1 year. An RGB PNG image is saved for each date that there is imagery, and finally a short timelapse video (in Gif format) is generated, showing change over a mining site in Australia.

### venus-denver-with-building-model.ipynb
This notebook demonstrates querying the STAC API for a Venus chip using an ROI, saving RGB PNG and multispectral Geotiffs, and then using this imagery to train a simple (not deep) model for building/not-building pixel level prediction.

# Notebooks - ML
Note that for faster training of deep learning models you should use a machine with a GPU. Both [Goolge Colab](https://research.google.com/colaboratory/) and [lightning.ai](https://lightning.ai/) provide limited access to free GPUs which will be sufficient to run the notebooks below.

### train-eurosat.ipynb
This notebook demonstrates training a chip classifier on a Sentinel 2 dataset called [EuroSAT](https://github.com/phelber/EuroSAT). Experiment with the choice of model, hyperparameters and pretrained weights to achieve the best performance you can. Note that using the [wandb logger](https://wandb.ai/) only requires a free account

### predict-eurosat.ipynb
This notebook demonstrates how to use your trained model to make predictions on imagery. A prediction is performed on a single image chip (feasible on CPU and with serveress inferencing) and also performed on a 5400 chip test dataset in batch mode (suitable for GPU machines and longer running processes such as AWS batch). A confusion matrix is plotted to allow inspection of the model performance, but you could experiment and create whole applications using this model

## oscd-train.ipynb
Shows training change detection model on OSCD dataset - to be extended with Venus dataset

# License
TBC

