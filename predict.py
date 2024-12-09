import numpy as np
import pandas as pd

from pathlib import Path
import pickle

from brist1d.tabular_transformers import feature_transformer
from brist1d.utils import WORK_DIR, RAW_TEST_FILE, PROCESSED_DATA_DIR, MODEL_DIR, SUBMISSION_DIR, make_dir, timer

# Transforms the test data for prediction
def transform_test(gap, n_prior, addition):

    make_dir(Path(WORK_DIR))
    make_dir(Path(PROCESSED_DATA_DIR))

    test_df = pd.read_csv(Path(RAW_TEST_FILE), low_memory=False)
    X_test = feature_transformer(test_df, gap=gap, n_prior=n_prior, data_source='test')
    X_test.to_pickle(Path(PROCESSED_DATA_DIR) / f'X_test_gap_{gap}_prior_{n_prior}_addition_{addition}.pkl')

# Make prediction based on trained model and generate csv file for submission
def predict_and_generate_submission(gap, n_prior, addition, regressor_type, suffix):
    with timer("Predict on test data and generate submission file"):
        make_dir(Path(SUBMISSION_DIR))

        transform_test(gap, n_prior, addition)

        test_df = pd.read_csv(Path(RAW_TEST_FILE), low_memory=False)
        X_test = pd.read_pickle(Path(PROCESSED_DATA_DIR) / f'X_test_gap_{gap}_prior_{n_prior}_addition_{addition}.pkl')


        with open(Path(MODEL_DIR) / f'{regressor_type}_gap_{gap}_prior_{n_prior}_addition_{addition}_pipeline_{suffix}.pkl', "rb") as input_pipeline_file:
            pipeline = pickle.load(input_pipeline_file)

        y_pred = pipeline.predict(X_test)
        
        submission_df = pd.DataFrame()
        submission_df['id'] = test_df.loc[X_test.index].id
        submission_df['bg+1:00'] = pd.Series(y_pred)
        submission_df.to_csv(Path(SUBMISSION_DIR) / f'{regressor_type}_gap_{gap}_prior_{n_prior}_addition_{addition}_submission_{suffix}.csv', index=False)


if __name__ == '__main__':
    predict_and_generate_submission(
        gap=1,
        n_prior=12,
        addition=0,
        regressor_type='lgbm',
        suffix='simplified'
    )