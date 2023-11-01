# Notebooks - STAC

These notebooks demonstrate data discovery and access from the EarthDataStore (EDS) STAC API. 

## Environment Setup
Refer to the README.md in the base level of this repository for environment setup instructions. 

## Notebooks

 ### venus-git-over-LEFR3.ipynb
 This notebook demonstrates querying the EDS STAC API for a Venus chip using a region of interest (ROI) and a period of 1 year. An RGB PNG image is saved for each date that there is imagery, and finally a short timelapse video (in Gif format) is generated, showing change over a mining site in Australia.

 ### venus-denver-with-building-model.ipynb
 This notebook demonstrates querying the STAC API for a Venus chip using an ROI, saving RGB PNG and multispectral Geotiffs, and then using this imagery to train a simple (not deep) model for building/not-building pixel level prediction.


