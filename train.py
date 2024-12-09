from time import sleep
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import lightgbm as lgb
from lightgbm import LGBMRegressor

from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline

import optuna
# from optuna.integration import LightGBMPruningCallback, XGBoostPruningCallback
# from optuna.pruners import MedianPruner
# from optuna import TrialPruned
from optuna.samplers import TPESampler

from brist1d.pipeline_transformers import pipeline_transformer_creator
from brist1d.cross_validator import TabularExpandingWindowCV
from brist1d.utils import PROCESSED_DATA_DIR, HYPERPARAMETER_TUNING_DIR, MODEL_DIR, make_dir, timer



# define function for cross validation and hyperparameter tuning
def objective_lgbm(trial, X, y, cv, lr, seed=101, columns_to_remove=None, target_encoders=None):
   
    param_grid = {
        "objective": 'root_mean_squared_error',
        "metric":'rmse',
        "random_state":seed,
        "verbose":-1,
        "n_estimators": 100000,
        "learning_rate": lr,
        "subsample_freq": 1,
        # "max_bin": 127,
        # "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 30, 500, step=10),
        # "max_depth": trial.suggest_int("max_depth", 5, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 25, 500, step=25),
        "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 1e-4, 1e-2, log=True),
        "reg_alpha": trial.suggest_int("reg_alpha", 0,20),
        "reg_lambda": trial.suggest_int("reg_lambda", 0, 20),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 1e-3, 1e-0, log=True),
        "subsample": trial.suggest_float("subsample", 0.3, 0.9),
        # "subsample_freq": trial.suggest_int("subsample_freq", 0, 3),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
    }

    transformer = pipeline_transformer_creator(X, columns_to_remove, target_encoders)
    regressor = LGBMRegressor(**param_grid)
    
    cv_scores = np.empty(cv.get_n_splits())
    cv_n_estimators = np.empty(cv.get_n_splits())
    
    # if param_grid['num_leaves'] >= param_grid['max_depth'] ** 2:
    #     print('num_leaves / max_depth restriction violated')
    #     sleep(1)
    #     raise TrialPruned()
        
    for idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        print(f'running fold n° {idx + 1}')
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        # train_weights, test_weights = weights.iloc[train_idx], weights.iloc[test_idx]

        X_train = transformer.fit_transform(X_train, y_train)
        X_test = transformer.transform(X_test)
        
        regressor.fit(
            X_train,
            y_train,
            # sample_weight=train_weights,
            eval_set=[(X_test, y_test)],
            # eval_sample_weight=[(test_weights)],
            eval_metric="rmse",
            callbacks=[
                # lgb.log_evaluation(
                #     100
                # ),
                lgb.early_stopping(
                    stopping_rounds=250, 
                    # min_delta=0.001, 
                    verbose=False
                ),
                # LightGBMPruningCallback(trial, 'rmse')
            ]
        )
            
        best_n_estimators = regressor.best_iteration_
        preds = regressor.predict(X_test)
        
        cv_scores[idx] = root_mean_squared_error(y_test, preds)
        cv_n_estimators[idx] = best_n_estimators

        print(f'RMSE: {cv_scores[idx]:.5f} at {best_n_estimators} estimators')
        sleep(1)
        
        # trial.report(cv_scores[idx], idx)
        # if trial.should_prune():
        #     raise TrialPruned()

    trial.set_user_attr("fold_rmse", cv_scores)
    trial.set_user_attr("best_n_estimators", cv_n_estimators)
    return np.mean(cv_scores)


# Hyperparameter tuning
def run_study(gap, n_prior, addition, regressor_type, cv, seed=101, lr=0.1, n_trials=10, columns_to_remove=None, target_encoders=None, suffix='default'):

    make_dir(Path(HYPERPARAMETER_TUNING_DIR))

    study = optuna.create_study(
        direction="minimize", 
        study_name='BrisT1D Blood Glucose Hyperparameter Tuning', 
        sampler=TPESampler(seed=seed), 
        # pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1),
    )

    X = pd.read_pickle(Path(PROCESSED_DATA_DIR) / f'X_full_gap_{gap}_prior_{n_prior}_addition_{addition}.pkl')
    y = pd.read_pickle(Path(PROCESSED_DATA_DIR) / f'y_full_gap_{gap}_prior_{n_prior}_addition_{addition}.pkl')

    cv = cv

    if regressor_type == 'lgbm':
        func = lambda trial: objective_lgbm(trial, X, y, cv, lr, seed, columns_to_remove, target_encoders)
        # other regressors can be added here
    else:
        raise Exception("Unrecognized regressor")

    study.optimize(func, n_trials=n_trials)

    optimizer_pkl_file = Path(HYPERPARAMETER_TUNING_DIR) / f'{regressor_type}_gap_{gap}_prior_{n_prior}_addition_{addition}_study_{suffix}.pkl'
    with open(optimizer_pkl_file, "wb") as output_file:
        pickle.dump(study, output_file)

    best_params=study.best_params

    best_params['n_estimators'] = int(study.best_trial.user_attrs['best_n_estimators'][-1]*1.5) 
    # The 1.5 factor is due to the n_estimator upward trend during hyperparameter tuning (around 15% for each fold, and this goes for 2 folds forward)
    # I expected it to use even more estimators, and since I saw overfitting at around 15% - 20% n_estimators after early stopping, I used 15% as addition
    # 1.15 * 1.15 * 1.15 = 1.52..., rounded down to 1.5
    # Even for new data, I would expect the number to be around the same, so it shouldn't affect too much, although it could be refined.

    best_params['learning_rate'] = lr
    best_params['subsample_freq'] = 1
    best_params['random_state'] = seed
    best_params['metric'] = 'rmse'
    best_params['verbose'] = -1

    best_params_pkl_file = Path(HYPERPARAMETER_TUNING_DIR) / f'{regressor_type}_gap_{gap}_prior_{n_prior}_addition_{addition}_best_params_{suffix}.pkl'
    with open(best_params_pkl_file, "wb") as output_file:
        pickle.dump(best_params, output_file)


# Model training with optimal parameters after hyperparameter tuning
def fit_model(gap, n_prior, addition, regressor_type, columns_to_remove=None, target_encoders=None, suffix='default'):

    make_dir(Path(MODEL_DIR))

    X = pd.read_pickle(Path(PROCESSED_DATA_DIR) / f'X_full_gap_{gap}_prior_{n_prior}_addition_{addition}.pkl')
    y = pd.read_pickle(Path(PROCESSED_DATA_DIR) / f'y_full_gap_{gap}_prior_{n_prior}_addition_{addition}.pkl')

    transformer = pipeline_transformer_creator(X, columns_to_remove, target_encoders)

    best_params_pkl_file = Path(HYPERPARAMETER_TUNING_DIR) / f'{regressor_type}_gap_{gap}_prior_{n_prior}_addition_{addition}_best_params_{suffix}.pkl'
    with open(best_params_pkl_file, "rb") as input_params_file:
        params = pickle.load(input_params_file)

    if regressor_type == 'lgbm':
        regressor = LGBMRegressor(**params)
        # other regressors can be added here
    else:
        raise Exception("Unrecognized regressor")

    model = Pipeline(
        steps=[
            ('transform', transformer),
            ('regressor', regressor),
        ]
    )

    model.fit(X, y)
    model_pkl_file = Path(MODEL_DIR) / f'{regressor_type}_gap_{gap}_prior_{n_prior}_addition_{addition}_model_{suffix}.pkl'
    with open(model_pkl_file, "wb") as output_model_file:
        pickle.dump(model, output_model_file)


# refit model with new data
def refit_model(gap, n_prior, addition, regressor_type, suffix='default'):

    X = pd.read_pickle(Path(PROCESSED_DATA_DIR) / f'X_full_gap_{gap}_prior_{n_prior}_addition_{addition}.pkl')
    y = pd.read_pickle(Path(PROCESSED_DATA_DIR) / f'y_full_gap_{gap}_prior_{n_prior}_addition_{addition}.pkl')

    with open(Path(MODEL_DIR) / f'{regressor_type}_gap_{gap}_prior_{n_prior}_addition_{addition}_model_{suffix}.pkl', "rb") as input_model_file:
        model = pickle.load(input_model_file)

    model.fit(X, y)

    model_pkl_file = Path(MODEL_DIR) / f'{regressor_type}_gap_{gap}_prior_{n_prior}_addition_{addition}_model_{suffix}.pkl'
    with open(model_pkl_file, "wb") as output_model_file:
        pickle.dump(model, output_model_file)


# train model with hyperparameter tuning (can be skipped)
# there's the option to just refit an already existing model. If refit == True, then no hyperparamter tuning is done
def train(gap, n_prior, addition, columns_to_remove, target_encoders, suffix='default', 
          refit=False, skip_hyperparameter_tuning=False, regressor_type=None, cv=None, seed=101, lr=0.1, n_trials=10):
    

    if refit:
        with timer("Refitting model"):       
            refit_model(
                gap=gap, 
                n_prior=n_prior, 
                addition=addition, 
                regressor_type=regressor_type, 
                suffix=suffix
                )
    else:
    
        if not skip_hyperparameter_tuning:
            with timer("Tuning Hyperparameters"):
                run_study(
                    gap=gap, 
                    n_prior=n_prior, 
                    addition=addition, 
                    regressor_type=regressor_type, 
                    cv=cv, 
                    seed=seed, 
                    lr=lr, 
                    n_trials=n_trials,
                    columns_to_remove=columns_to_remove,
                    target_encoders=target_encoders,
                    suffix=suffix
                    )

        with timer("Fitting model"):       
            fit_model(
                gap=gap, 
                n_prior=n_prior, 
                addition=addition, 
                regressor_type=regressor_type, 
                columns_to_remove=columns_to_remove,
                target_encoders=target_encoders,
                suffix=suffix
                )
    



if __name__ == '__main__':
    from brist1d.params import (
        GAP, N_PRIOR, ADDITION, COLUMNS_TO_REMOVE, TARGET_ENCODERS, 
        SUFFIX, REFIT, SKIP_HYPERPARAMETER_TUNING, REGRESSOR_TYPE, SEED, LR, N_TRIALS
        ) 

    with open(Path(PROCESSED_DATA_DIR) / f'phase_1_indexes_{GAP}_prior_{N_PRIOR}_addition_{ADDITION}.pkl', "rb") as input_file:
        phase_1_indexes = pickle.load(input_file)
    
    initial_window = len(phase_1_indexes)
    step_list = pd.read_pickle(Path(PROCESSED_DATA_DIR) / f'step_list_full_gap_{GAP}_prior_{N_PRIOR}_addition_{ADDITION}.pkl')

    cv = TabularExpandingWindowCV(initial_window=initial_window, step_list=step_list, initial_step=12, step_length=6, forecast_horizon=12)

    columns_to_remove = ([]
        + [f'bg_{lag}_lag' for lag in [35,40,45,50,55,60]]
        + [f'bg_{lag}_diff' for lag in [30,35,40,45,50,55]]
    )

    target_encoders = ['mean', 'std', 'skew', 'kurt']

    train(
        gap=GAP, 
        n_prior=N_PRIOR, 
        addition=ADDITION, 
        columns_to_remove=COLUMNS_TO_REMOVE, 
        target_encoders=TARGET_ENCODERS, 
        suffix=SUFFIX,
        refit=REFIT,
        skip_hyperparameter_tuning=SKIP_HYPERPARAMETER_TUNING, 
        regressor_type=REGRESSOR_TYPE, 
        cv=cv, 
        seed=SEED, 
        lr=LR,
        n_trials=N_TRIALS)
