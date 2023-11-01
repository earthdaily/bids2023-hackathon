# Notebooks - Sentinel 2 Modelling
Note that for faster training of deep learning models you should use a machine with a GPU. Both [Google Colab](https://research.google.com/colaboratory/) and [lightning.ai](https://lightning.ai/) provide limited access to free GPUs which will be sufficient to run the notebooks below.

## Environment Setup
Refer to the README.md in the base level of this repository for environment setup instructions. 

## Training and Prediction Using the EuroSAT Dataset

### train-eurosat.ipynb
This notebook demonstrates training a chip classifier on a Sentinel 2 dataset called [EuroSAT](https://github.com/phelber/EuroSAT). Experiment with the choice of model, hyperparameters and pretrained weights to achieve the best performance you can. Note that using the [wandb logger](https://wandb.ai/) only requires a free account

### predict-eurosat.ipynb
This notebook demonstrates how to use your trained model to make predictions on imagery. A prediction is performed on a single image chip (feasible on CPU and with serveress inferencing) and also performed on a 5400 chip test dataset in batch mode (suitable for GPU machines and longer running processes such as AWS batch). A confusion matrix is plotted to allow inspection of the model performance, but you could experiment and create whole applications using this model

## Training using the OSCD Dataset

### oscd-train.ipynb
Shows training change detection model on OSCD dataset

