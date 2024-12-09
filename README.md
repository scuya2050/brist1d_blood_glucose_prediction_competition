## BrisT1D Blood Glucose Prediction Competition - 1st Place Solution

Below you can find a outline of how to reproduce my solution for the BrisT1D Blood Glucose Prediction Competition.\
If you run into any trouble with the setup/code or have any questions please contact me at sebastiancuya20.50@gmail.com

## ARCHIVE CONTENTS
- `brist1d`                 : contains scripts with transformation and utility functions, as well as the parameters that the scripts use.
- `data`                    : contains both the raw and the processed 
data
- `documentation`           : contains model summary document
- `eda`                     : contains code for the exploratory data analysis
- `hyperparameter_tuning`   : contains the optimizer study and the best parameters for the model
- `model`                   : contains the trained model
- `submission`              : contains the prediction in submission format
- `work`                    : contains intermediate data files generated during the proecess
- `prepare_data.py`         : code to preprocess train data
- `train.py`                : code to tune hyperparameters of the model and to fit the model
- `predict.py`              : code to generate predictions and generate the output submission file
- `end_to_end.py`           : code that preprocesses the
- `directory_structure.txt` : structure of the archive
- `requirements.txt`        : list of python packages
- `settings.json`           : settings file with paths

## HARDWARE: 
The solution was created in a local resource. Only CPU was used.\
Below are the specs:

- `CPU`: AMD Ryzen 7 5800H, 1 socket, 8 cores per socket, 2 threads per core
- `RAM`: 32 GB DDR4 3200 MHz 
- `GPU`: NVIDIA GeForce RTX 3060 Laptop GPU (not used)

## SOFTWARE 
- Python 3.12.4 (python packages are detailed separately in `requirements.txt`)

## DATA SETUP 
Download train and test data from Kaggle in the raw data location (./data/raw). You can also use Kaggle API to load data.\
In case other files or new data want to be used, make sure to either keep the original file names (train.csv, test.csv) or change the names in the settings.json file


## SOLUTION REPRODUCTION
The solution is produced in 3 steps:
1. `Preprocess`: Executed by `prepare_data.py`
2. `Train`: Executed by `train.py`
2. `Predict`: Executed by `predict.py`

There are 2 options to produce the solution.
1. Ordinary prediction
    - Produced by executing `end_to_end.py`, which runs the 3 steps sequentially
    - Runs in about 2 hours
    - Creates prediction from scratch
2. Fast prediction
    - Produced by executing `predict.py`
    - Runs in about 3 seconds
    - Uses pre-existent trained model

Additionally, each step can be executed individually by using the respective script directly

### Command to run each option
The working directory must be the top level directory.

1. Ordinary prediction\
python ./end_to_end.py

2. Fast prediction\
python ./predict.py
