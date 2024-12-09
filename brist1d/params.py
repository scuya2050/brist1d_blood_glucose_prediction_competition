# Parameters:

# GAP: Time between observations (gap = 1 means getting observations every 5 minutes)
# N_PRIOR: Number of gaps that the data window will consider (gap = 12 means a window of 1 hour)
# ADDITION: Adds a fixed numer of data points to the window. Used during expansion
# REGRESSOR TYPE: Defines the algorithm to use
# COLUMNS_TO_REMOVE: Determines the columns that the pipeline will discard when performing the final transformation
# TARGET_ENCODERS: Determines what encoders the pipeline will calculate
# SUFFIX: Custom name to add at the end of some processes, in order not to overwrite files
# LR: Learning rate for regressor
# N_TRIALS: Number of trials for hyperparameter tuning study
# SEED: Random state for regressor and tuning study
# SKIP_EXPANSION: If true, the preprocessing step does not expand the data and will use the already expanded data from a previous expansion.
# REFIT: If true, will used the already existing pipeline to refit it with new data
# SKIP_HYPERPARAMETER_TUNING: If true, the training process will use the already existing optimal parameters found in a previous study


# Parameters for standard model
GAP = 1
N_PRIOR = 12
ADDITION = 0
REGRESSOR_TYPE = 'lgbm'
COLUMNS_TO_REMOVE = ([]
    + [f'bg_{lag}_lag' for lag in [35,40,45,50,55,60]]
    + [f'bg_{lag}_diff' for lag in [30,35,40,45,50,55]]
)
TARGET_ENCODERS = ['mean', 'std', 'skew', 'kurt']
SUFFIX = 'standard'
LR = 0.01
N_TRIALS = 25
SEED = 864 # Used to replicate results

SKIP_EXPANSION = False
REFIT = False  # If just an already existing model wants to be used, set this to True
SKIP_HYPERPARAMETER_TUNING = False


# # Parameters for simplified model
# GAP = 1
# N_PRIOR = 12
# ADDITION = 0
# REGRESSOR_TYPE = 'lgbm'
# COLUMNS_TO_REMOVE = ([]
#     + [f'bg_{lag}_lag' for lag in [20,25,30,35,40,45,50,55,60]]
#     + [f'bg_{lag}_diff' for lag in [15,20,25,30,35,40,45,50,55]]
#     + [f'insulin_{lag}_lag' for lag in [0,5,10,15,20,25,30,35,40,45,50,55]]
#     + [f'carbs_{lag}_lag' for lag in [0,5,10,15,20,25,30,35,40,45,50,55]]
#     + [f'hr_{lag}_lag' for lag in [0,5,10,15,20,25,30,35,40,45,50,55]]
#     + [f'steps_{lag}_lag' for lag in [0,5,10,15,20,25,30,35,40,45,50,55]]
#     + [f'cals_{lag}_lag' for lag in [0,5,10,15,20,25,30,35,40,45,50,55]]
#     + [f'activity_{lag}_lag' for lag in [0,5,10,15,20,25,30,35,40,45,50,55]]
#     + ['bg_gap']
# )
# TARGET_ENCODERS = ['mean']
# SUFFIX = 'simplified'
# LR = 0.01
# N_TRIALS = 25
# SEED = 864 # Used to replicate results

# SKIP_EXPANSION = False
# REFIT = False  # If just an already existing model wants to be used, set this to True
# SKIP_HYPERPARAMETER_TUNING = False