import numpy as np
import pandas as pd
import gc
from pathlib import Path
import pickle
from brist1d.tabular_transformers import *
from brist1d.utils import RAW_TRAIN_FILE, RAW_TEST_FILE, PROCESSED_DATA_DIR, WORK_DIR, make_dir, timer 


def expand(gap, n_prior, addition):
    # Expands both train and test data
    # Expanding train data won't generate much additional data, but expaning test data will
    # The step list stores the iterations from the expansion. Given that it is just useful for the test data, it is set to -1 for the train data
    make_dir(Path(WORK_DIR))

    train_df = pd.read_csv(Path(RAW_TRAIN_FILE), low_memory=False)
    test_df = pd.read_csv(Path(RAW_TEST_FILE), low_memory=False)

    base_train_df_expanded, train_df_expanded = data_expander(train_df, gap=gap, n_prior=n_prior, addition=addition, data_source='train')
    base_train_df_expanded.to_pickle(Path(WORK_DIR) / f'base_train_df_expanded_shift_{gap*n_prior + addition}.pkl')
    train_df_expanded.to_pickle(Path(WORK_DIR) / f'train_df_expanded_shift_{gap*n_prior + addition}.pkl')
    step_list_df = pd.Series(index=train_df_expanded.index, data=-1)
    step_list_df.to_pickle(Path(WORK_DIR) / f'step_list_df_shift_{gap*n_prior + addition}.pkl')

    base_train_from_test_df, train_from_test_df, step_list_from_test_df = data_expander(test_df, gap=gap, n_prior=n_prior, addition=addition, data_source='train_from_test')
    base_train_from_test_df.to_pickle(Path(WORK_DIR) / f'base_train_from_test_df_shift_{gap*n_prior + addition}.pkl')
    train_from_test_df.to_pickle(Path(WORK_DIR) / f'train_from_test_df_shift_{gap*n_prior + addition}.pkl')
    step_list_from_test_df.to_pickle(Path(WORK_DIR) / f'step_list_from_test_df_{gap*n_prior + addition}.pkl')


def transform_train(gap, n_prior, addition):
    # Transforms the train data (expanded train and train from test after expansion)
    # Merges the transformed train and train from test data into a single dataset

    make_dir(Path(WORK_DIR))
    make_dir(Path(PROCESSED_DATA_DIR))

    train_df_expanded = pd.read_pickle(Path(WORK_DIR) / f'train_df_expanded_shift_{gap*n_prior + addition}.pkl')
    # test_df = pd.read_csv(Path(RAW_TEST_FILE), low_memory=False)
    train_from_test_df = pd.read_pickle(Path(WORK_DIR) / f'train_from_test_df_shift_{gap*n_prior + addition}.pkl')

    step_list_df = pd.read_pickle(Path(WORK_DIR) / f'step_list_df_shift_{gap*n_prior + addition}.pkl')
    step_list_from_test_df = pd.read_pickle(Path(WORK_DIR) / f'step_list_from_test_df_{gap*n_prior + addition}.pkl')
    
    X, y, patient_groups, day_groups = feature_transformer(train_df_expanded, gap=gap, n_prior=n_prior, data_source='train')
    # X_test = feature_transformer(test_df, gap=gap, n_prior=n_prior, data_source='test')
    X_from_test, y_from_test, patient_groups_from_test, day_groups_from_test = feature_transformer(train_from_test_df, gap=gap, n_prior=n_prior, data_source='train_from_test')
    
    train_df_full = pd.concat([train_df_expanded, train_from_test_df], ignore_index=True)
    X_full = pd.concat([X, X_from_test], ignore_index=True)
    y_full = pd.concat([y, y_from_test], ignore_index=True)
    patient_groups_full = pd.concat([patient_groups, patient_groups_from_test], ignore_index=True)
    day_groups_full = pd.concat([day_groups, day_groups_from_test], ignore_index=True)
    patient_day_groups_full = patient_groups_full.astype(str) + "/" + day_groups_full.astype(str)
    step_list_full = pd.concat([step_list_df, step_list_from_test_df], ignore_index=True)
    
    phase_1_indexes = train_df_expanded.index
    phase_2_indexes = train_from_test_df.index

    train_df_full.to_pickle(Path(PROCESSED_DATA_DIR) / f'train_df_full_gap_{gap}_prior_{n_prior}_addition_{addition}.pkl')
    X_full.to_pickle(Path(PROCESSED_DATA_DIR) / f'X_full_gap_{gap}_prior_{n_prior}_addition_{addition}.pkl')
    # X_test.to_pickle(Path(PROCESSED_DATA_DIR) / f'X_test_gap_{gap}_prior_{n_prior}_addition_{addition}.pkl')
    y_full.to_pickle(Path(PROCESSED_DATA_DIR) / f'y_full_gap_{gap}_prior_{n_prior}_addition_{addition}.pkl')
    patient_groups_full.to_pickle(Path(PROCESSED_DATA_DIR) / f'patient_groups_full_gap_{gap}_prior_{n_prior}_addition_{addition}.pkl')
    day_groups_full.to_pickle(Path(PROCESSED_DATA_DIR) / f'day_groups_gap_{gap}_prior_{n_prior}_addition_{addition}.pkl')
    patient_day_groups_full.to_pickle(Path(PROCESSED_DATA_DIR) / f'patient_day_groups_full_gap_{gap}_prior_{n_prior}_addition_{addition}.pkl')
    step_list_full.to_pickle(Path(PROCESSED_DATA_DIR) / f'step_list_full_gap_{gap}_prior_{n_prior}_addition_{addition}.pkl')

    with open(Path(PROCESSED_DATA_DIR) / f'phase_1_indexes_{gap}_prior_{n_prior}_addition_{addition}.pkl', "wb") as output_file:
        pickle.dump(phase_1_indexes, output_file)

    with open(Path(PROCESSED_DATA_DIR) / f'phase_2_indexes_{gap}_prior_{n_prior}_addition_{addition}.pkl', "wb") as output_file:
        pickle.dump(phase_2_indexes, output_file)


# prepare data and measure time
def preprocess(gap, n_prior, addition, skip_expansion=False):
    if not skip_expansion:
        with timer("Expanding data"):
            expand(gap, n_prior, addition)
    with timer("Transforming train data"):
        transform_train(gap, n_prior, addition)




if __name__ == '__main__':
    gap = 1
    n_prior = 12
    addition = 0

    preprocess(gap, n_prior, addition, skip_expansion=False)