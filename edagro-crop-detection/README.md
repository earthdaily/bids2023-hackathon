# Predict crops using EarthDaily simulated constellation

The idea of this hackathon is to use the EarthDaily Simulated dataset to predict the 2019 crops in the current season (as if we were july 15th). The dataset was pregenerated using the earthdaily python package and the code is available in this notebook.

We have three different years (2018, 2019 and 2020) and our workflow train from a specific year (2018 or 2020) only on spectral bands to predict an independent year (2019). We challenge you to have the maximum accuracy to predict year 2019 by having only data up to july 1st (so a month and a half of data, from may 15th to july 1st).

## Setup the environment

Download mamba : https://github.com/conda-forge/miniforge#mambaforge.

If you're using powershell copy/paste this to have mamba commands : ```powershell -ExecutionPolicy ByPass -NoExit -Command "& 'C:\Users\$Env:UserName\AppData\Local\mambaforge\condabin\mamba.bat' init powershell"```

Then using setup the bids23 environment : `mamba env update --name bids23 --file requirements.yml`.

To run the notebook : `mamba activate bids23` and `jupyter notebook`.

## Try it !

Now you can edit and try `workflow.ipynb`. Remember, the highest accuracy you can have to predict 2019 with training on 2018 and 2020, the better it is :).