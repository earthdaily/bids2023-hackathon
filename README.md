<div align="center">
  <p>
    <a href="https://pages.earthdaily.com/hackathon">
        <img src="banner.png" width="1100">
    </a>
</p>
</div>

This repository contains scripts and notebooks for the [EarthDaily Analytics Hackathon](https://pages.earthdaily.com/hackathon) at [BiDS 2023](https://www.bigdatafromspace2023.org/). There is material for people who want to focus on data access and data visualisation, and material for people who want to explore machine learning. Of course you can combine these, for example by first training a model and then processing imagery generated via STAC to build cool analytics applications!

The following notebooks are focused on data access and data visualization: 
- stac-notebooks
- stac-app

The following notebooks are focused on machine learning: 
- edagro-crop-detection
- sentinel2-modelling

Please see the README.md for each notebook for additional descriptions. 

We highly encourage participants in the BiDS 2023 Hackathon to set up their development environment and become familiar with running the notebooks prior to the event!

# Setup

## Choose Your Environment
To run the notebooks without leaving Github or setting up a Python environment you can use a [codespace](https://github.com/features/codespaces). This approach will be demonstrated. Note however that codespaces do not provide a GPU (to the best of our knowledge).

In [Google Colab](https://research.google.com/colaboratory/), [AWS Sagemaker Studio Lab](https://studiolab.sagemaker.aws/) or on [lightning.ai](https://lightning.ai/) you just need to git clone this repository. 


## Notebook Specific Setup 
Each notebook has specific environment requirements on top of the base requirements. To install these requirements: 
```
cd <notebook-subdir>  # E.g. edagro-crop-detection
pip install -r requirements.txt
```


## EDS Credentials Setup 
If you are participating in the BiDS 2023 Hackathon, you should have received credentials to access the EDS. Make a copy of `.env.sample` to `.env` and fill in the credentials and access information as provided. 


# Need Help? 
If you are participating in the BiDS 2023 Hackathon, you should have received an invite to the Slack channel for communication and coordination prior to the event. Prior to the event, feel free to ask questions in that channel. 