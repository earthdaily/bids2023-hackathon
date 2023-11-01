# Notebooks - STAC

These notebooks demonstrate data discovery and access from the EarthDataStore (EDS) STAC API. 

 ### venus-git-over-LEFR3.ipynb
 This notebook demonstrates querying the EDS STAC API for a Venus chip using a region of interest (ROI) and a period of 1 year. An RGB PNG image is saved for each date that there is imagery, and finally a short timelapse video (in Gif format) is generated, showing change over a mining site in Australia.

 ### venus-denver-with-building-model.ipynb
 This notebook demonstrates querying the STAC API for a Venus chip using an ROI, saving RGB PNG and multispectral Geotiffs, and then using this imagery to train a simple (not deep) model for building/not-building pixel level prediction.

 ### STAC-App
 This demonstration app enables users to make skyfox queries and render data within an AOI for two dates to enable quick comparison. If time permits, it may grow to include unsupervised segmentation capabilities. To run this app, run the following:
 ```
 cd stac-app
 ./run_app.sh
 ```
 Note: credentials for Skyfox access are not yet tested for external users. 