import numpy as np
import pandas as pd

from scipy.stats import kurtosis

from feature_engine.encoding import MeanEncoder
from feature_engine.dataframe_checks import check_X_y

from sklearn.compose import ColumnTransformer

class VarianceEncoder(MeanEncoder):
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        variables_ = self._check_or_select_variables(X)
        self._check_na(X, variables_)

        self.encoder_dict_ = {}

        y_prior = y.var()

        if self.unseen == "encode":
            self._unseen = y_prior

        if self.smoothing == "auto":
            y_var = y.var(ddof=0)
        for var in variables_:
            if self.smoothing == "auto":
                damping = y.groupby(X[var]).var(ddof=0) / y_var
            else:
                damping = self.smoothing
            counts = X[var].value_counts()
            counts.index = counts.index.infer_objects()
            _lambda = counts / (counts + damping)
            self.encoder_dict_[var] = (
                _lambda * y.groupby(X[var], observed=False).var()
                + (1.0 - _lambda) * y_prior
            ).to_dict()
            
        self.variables_ = variables_
        self._get_feature_names_in(X)
        return self

class StdandardDeviationEncoder(MeanEncoder):
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        variables_ = self._check_or_select_variables(X)
        self._check_na(X, variables_)

        self.encoder_dict_ = {}

        y_prior = y.std()

        if self.unseen == "encode":
            self._unseen = y_prior

        if self.smoothing == "auto":
            y_var = y.var(ddof=0)
        for var in variables_:
            if self.smoothing == "auto":
                damping = y.groupby(X[var]).var(ddof=0) / y_var
            else:
                damping = self.smoothing
            counts = X[var].value_counts()
            counts.index = counts.index.infer_objects()
            _lambda = counts / (counts + damping)
            self.encoder_dict_[var] = (
                _lambda * y.groupby(X[var], observed=False).std()
                + (1.0 - _lambda) * y_prior
            ).to_dict()
            
        self.variables_ = variables_
        self._get_feature_names_in(X)
        return self


class SkewnessEncoder(MeanEncoder):
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        variables_ = self._check_or_select_variables(X)
        self._check_na(X, variables_)

        self.encoder_dict_ = {}

        y_prior = y.skew()

        if self.unseen == "encode":
            self._unseen = y_prior

        if self.smoothing == "auto":
            y_var = y.var(ddof=0)
        for var in variables_:
            if self.smoothing == "auto":
                damping = y.groupby(X[var]).var(ddof=0) / y_var
            else:
                damping = self.smoothing
            counts = X[var].value_counts()
            counts.index = counts.index.infer_objects()
            _lambda = counts / (counts + damping)
            self.encoder_dict_[var] = (
                _lambda * y.groupby(X[var], observed=False).skew()
                + (1.0 - _lambda) * y_prior
            ).to_dict()
            
        self.variables_ = variables_
        self._get_feature_names_in(X)
        return self


class KurtosisEncoder(MeanEncoder):
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        variables_ = self._check_or_select_variables(X)
        self._check_na(X, variables_)

        self.encoder_dict_ = {}

        y_prior = y.kurt()

        if self.unseen == "encode":
            self._unseen = y_prior

        if self.smoothing == "auto":
            y_var = y.var(ddof=0)
        for var in variables_:
            if self.smoothing == "auto":
                damping = y.groupby(X[var]).var(ddof=0) / y_var
            else:
                damping = self.smoothing
            counts = X[var].value_counts()
            counts.index = counts.index.infer_objects()
            _lambda = counts / (counts + damping)
            self.encoder_dict_[var] = (
                _lambda * y.groupby(X[var], observed=False).apply(lambda x: kurtosis(x))
                + (1.0 - _lambda) * y_prior
            ).to_dict()
            
        self.variables_ = variables_
        self._get_feature_names_in(X)
        return self


class MedianEncoder(MeanEncoder):
    def fit(self, X: pd.DataFrame, y: pd.Series):
        X, y = check_X_y(X, y)
        variables_ = self._check_or_select_variables(X)
        self._check_na(X, variables_)

        self.encoder_dict_ = {}

        y_prior = y.median()

        if self.unseen == "encode":
            self._unseen = y_prior

        if self.smoothing == "auto":
            y_var = y.var(ddof=0)
        for var in variables_:
            if self.smoothing == "auto":
                damping = y.groupby(X[var]).var(ddof=0) / y_var
            else:
                damping = self.smoothing
            counts = X[var].value_counts()
            counts.index = counts.index.infer_objects()
            _lambda = counts / (counts + damping)
            self.encoder_dict_[var] = (
                _lambda * y.groupby(X[var], observed=False).median()
                + (1.0 - _lambda) * y_prior
            ).to_dict()

        # assign underscore parameters at the end in case code above fails
        self.variables_ = variables_
        self._get_feature_names_in(X)
        return self
    


# create transformer in pipeline mainly for encoders to avoid data leakage and to make cross validation easier to do
def pipeline_transformer_creator(X, columns_to_remove=None, target_encoders=None):
    cols = X.columns.to_list()

    if columns_to_remove is not None:
        cols = [col for col in cols if col not in columns_to_remove]
    
    cols.remove('p_num')

    encoders = []
    if target_encoders is not None:
        if 'mean' in target_encoders:
            encoders.append(('mean', MeanEncoder(), ['p_num']))
        if 'std' in target_encoders:
            encoders.append(('std', StdandardDeviationEncoder(), ['p_num']))
        if 'skew' in target_encoders:
            encoders.append(('skew', SkewnessEncoder(), ['p_num']))
        if 'kurt' in target_encoders:
            encoders.append(('kurt', KurtosisEncoder(), ['p_num']))
    

    pipeline_transformer = ColumnTransformer(
        transformers=encoders + [('col', 'passthrough', cols)],
        remainder='drop'
    )
    
    pipeline_transformer.set_output(transform="pandas")
    return pipeline_transformer