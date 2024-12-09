# This is meant to run the entire process from beginning to end
# The input is the raw data and the output is the submission file with the prediction
# Some steps can be skipped if need, but with the following parameters I aim to replicate the chosen submission for the private leaderboard
# Note that each individual script (prepare_date.py, train.py, predict.py) can be run individually. This script is just meant to make thing simpler
# If just the prediction is need for an already trained model, just run the predict.py script

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from brist1d.utils import RAW_DATA_DIR, PROCESSED_DATA_DIR, WORK_DIR, MODEL_DIR, SUBMISSION_DIR
from brist1d.cross_validator import TabularExpandingWindowCV

from prepare_data import preprocess
from train import train
from predict import predict_and_generate_submission

# STEP 0: SET PARAMETERS (Except CV, since it needs to be generated according to the preprocessed data)

from brist1d.params import *

gap = GAP
n_prior = N_PRIOR
addition = ADDITION
regressor_type = REGRESSOR_TYPE
columns_to_remove = COLUMNS_TO_REMOVE
target_encoders = TARGET_ENCODERS
suffix = SUFFIX
lr = LR
n_trials = N_TRIALS
seed = SEED # Used to replicate results

skip_expansion = SKIP_EXPANSION
refit = REFIT
skip_hyperparameter_tuning = SKIP_HYPERPARAMETER_TUNING


# STEP 1: PREPROCESS (just train data)
# Just skip the expansion if it has been done before for the same data, and for the same parameters
preprocess(gap, n_prior, addition, skip_expansion=skip_expansion)


# STEP 2: TRAIN (and create CV)
# Just skip the hyperparameter tuning if either a study already exists or a fitted model already exists
# Use refit in case of new data for the already trained model. If refit == True, then no hyperparamter tuning is done
with open(Path(PROCESSED_DATA_DIR) / f'phase_1_indexes_{gap}_prior_{n_prior}_addition_{addition}.pkl', "rb") as input_file:
    phase_1_indexes = pickle.load(input_file)

initial_window = len(phase_1_indexes)
step_list = pd.read_pickle(Path(PROCESSED_DATA_DIR) / f'step_list_full_gap_{gap}_prior_{n_prior}_addition_{addition}.pkl')

cv = TabularExpandingWindowCV(initial_window=initial_window, step_list=step_list, initial_step=12, step_length=6, forecast_horizon=12)

train(
    gap, 
    n_prior, 
    addition, 
    columns_to_remove, 
    target_encoders, 
    suffix,
    refit=refit,
    skip_hyperparameter_tuning=skip_hyperparameter_tuning, 
    regressor_type=regressor_type, 
    cv=cv, 
    seed=seed, 
    lr=lr,
    n_trials=n_trials)


# STEP 3: PREDICT (includes test data tabular preprocessing)

predict_and_generate_submission(
    gap,
    n_prior,
    addition,
    regressor_type=regressor_type,
    suffix=suffix
)

# After all is done, the submission file will be generated in the SUBMISSION directory






