Hello!

Below you can find a outline of how to reproduce my solution for the BrisT1D Blood Glucose Prediction Competition.\
If you run into any trouble with the setup/code or have any questions please contact me at sebastiancuya20.50@gmail.com

## ARCHIVE CONTENTS
- brist1d                 : contains scripts with transformation and utility functions
- data                    : contains both the raw and the processed data
- hyperparameter_tuning   : contains the optimizer study and the best parameters for the model
- model                   : contains the trained model
- submission              : contains the prediction in submission format
- prepare_data.py         : code to preprocess train data
- train.py                : code to tune hyperparameters of the model and to fit the model
- predict.py              : code to generate predictions and generate the output submission file
- requirements.txt        : python packages used
- settings.json           : 

## HARDWARE: 
The solution was created in a local resource. Only CPU was used.\
Below are the specs:

- CPU: AMD Ryzen 7 5800H, 1 socket, 8 cores per socket, 2 threads per core
- RAM: 32 GB DDR4 3200 MHz 
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU (not used)

## SOFTWARE 
- Python 3.12.4 (python packages are detailed separately in `requirements.txt`)

## DATA SETUP 
Download train and test data from Kaggle in the raw data location (./data/raw). You can also use Kaggle API to load data.\
In case other files or new data wants to be used, make sure to either keep the original file names (train.csv, test.csv) or change the names in the settings.json file


## SOLUTION REPRODUCTION
The train/predict code will also call this script if it has not already been run on the relevant data.
python ./train_code/prepare_data.py --data_dir=data/stage1/ --output_dir=data/stage1_cleaned

#MODEL BUILD: There are three options to produce the solution.
1) very fast prediction
    a) runs in a few minutes
    b) uses precomputed neural network predictions
2) ordinary prediction
    a) expect this to run for 1-2 days
    b) uses binary model files
3) retrain models
    a) expect this to run about a week
    b) trains all models from scratch
    c) follow this with (2) to produce entire solution from scratch

shell command to run each build is below
#1) very fast prediction (overwrites comp_preds/sub1.csv and comp_preds/sub2.csv)
python ./predict_code/calibrate_model.py

#2) ordinary prediction (overwrites predictions in comp_preds directory)
sh ./predict_code/predict_models.sh

#3) retrain models (overwrites models in comp_model directory)
sh ./train_code/train_models.sh
