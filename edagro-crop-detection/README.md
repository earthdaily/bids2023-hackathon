# Crop Type Prediction Using Simulated EarthDaily Constellation Imagery

The idea of this hackathon is to use the Simulated EarthDaily Constellation (EDC) dataset to predict the 2019 crops in the current season (as if we were july 15th). The dataset was pregenerated using the `earthdaily`` python package and the code is available in this notebook.

We have three different years (2018, 2019 and 2020) and our workflow trains from a specific year (2018 or 2020) only on spectral bands to predict an independent year (2019). We challenge you to have the maximum accuracy to predict year 2019 by having only data up to July 1st (so a month and a half of data, from May 15th to July 1st).

## Environment Setup

Refer to the base README.md in this repository for environment setup instructions. 

## Try It !

Now you can edit and try `workflow.ipynb`. Remember, the highest accuracy you can have to predict 2019 with training on 2018 and 2020, the better it is :).