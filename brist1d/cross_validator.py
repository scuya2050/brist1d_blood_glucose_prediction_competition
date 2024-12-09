import numpy as np

from sklearn.model_selection import BaseCrossValidator

class TabularExpandingWindowSplitter(BaseCrossValidator):
    def __init__(self, initial_window, step_list, initial_step, step_length, forecast_horizon):
        self.initial_window = initial_window
        self.step_list = np.array(step_list)
        self.initial_step = initial_step
        self.step_length = step_length
        self.forecast_horizon = forecast_horizon

    def split(self, X, y=None, groups=None):       
        # Define indices for the fixed portion
        initial_window_indices = np.arange(self.initial_window)
        remaining_indices = np.arange(len(initial_window_indices), len(X))
        remaining_step_list = self.step_list[len(initial_window_indices): len(X)]
        
        last_step = self.step_list.max()
        n_splits = (((last_step - self.forecast_horizon) - self.initial_step) // self.step_length) + 1
        
        for k in np.arange(n_splits):
            last_step_window = self.initial_step + k * self.step_length
            window_indices = (len(initial_window_indices) + np.arange(len(remaining_step_list)))[remaining_step_list <= last_step_window]
            forecast_step = last_step_window + self.forecast_horizon
            forecast_indices = (len(initial_window_indices) + np.arange(len(remaining_step_list)))[remaining_step_list == forecast_step]
            
            train_indices = np.concatenate([initial_window_indices, window_indices])
            test_indices = forecast_indices
            yield train_indices, test_indices, forecast_step
            
    def get_n_splits(self, X=None, y=None, groups=None):
        last_step = self.step_list.max()
        n_splits = (((last_step - self.forecast_horizon) - self.initial_step) // self.step_length) + 1
        return n_splits

class TabularExpandingWindowCV(BaseCrossValidator):
    def __init__(self, initial_window, step_list, initial_step, step_length, forecast_horizon):
        self.initial_window = initial_window
        self.step_list = np.array(step_list)
        self.initial_step = initial_step
        self.step_length = step_length
        self.forecast_horizon = forecast_horizon

    def split(self, X, y=None, groups=None):       
        # Define indices for the fixed portion
        initial_window_indices = np.arange(self.initial_window)
        remaining_indices = np.arange(len(initial_window_indices), len(X))
        remaining_step_list = self.step_list[len(initial_window_indices): len(X)]
        
        last_step = self.step_list.max()
        n_splits = (((last_step - self.forecast_horizon) - self.initial_step) // self.step_length) + 1
        
        for k in np.arange(n_splits):
            last_step_window = self.initial_step + k * self.step_length
            window_indices = (len(initial_window_indices) + np.arange(len(remaining_step_list)))[remaining_step_list <= last_step_window]
            forecast_step = last_step_window + self.forecast_horizon
            forecast_indices = (len(initial_window_indices) + np.arange(len(remaining_step_list)))[remaining_step_list == forecast_step]
            
            train_indices = np.concatenate([initial_window_indices, window_indices])
            test_indices = forecast_indices
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        last_step = self.step_list.max()
        n_splits = (((last_step - self.forecast_horizon) - self.initial_step) // self.step_length) + 1
        return n_splits