import os
import time
import json
from datetime import datetime
from contextlib import contextmanager, redirect_stdout


# load file paths
settings = json.load(open("./settings.json"))
RAW_DATA_DIR = settings["RAW_DATA_DIR"]
RAW_TRAIN_FILE = settings["RAW_TRAIN_FILE"]
RAW_TEST_FILE = settings["RAW_TEST_FILE"]
PROCESSED_DATA_DIR = settings["PROCESSED_DATA_DIR"]
WORK_DIR = settings["WORK_DIR"]
HYPERPARAMETER_TUNING_DIR = settings["HYPERPARAMETER_TUNING_DIR"]
MODEL_DIR = settings["MODEL_DIR"]
SUBMISSION_DIR = settings["SUBMISSION_DIR"]

def seconds_to_hh_mm_ss(duration):
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = duration % 60

    return f"{hours} hours, {minutes} minutes, {seconds:.2f} seconds"

@contextmanager
def timer(name):
    print(f'{datetime.now()} - [{name}] ...')
    t0 = time.time()
    yield
    print(f'{datetime.now()} - [{name}] done in {seconds_to_hh_mm_ss(time.time() - t0)} \n')
    

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


class Logger(object):
    def __init__(self, file_path, mode="w", verbose=False):
        self.file_path = file_path
        self.verbose = verbose
        open(file_path, mode=mode)
        
    def append(self, line, print_line=None):
        if print_line or self.verbose:
            print(line)
        with open(self.file_path, "a") as f:
            with redirect_stdout(f):
                print(line)        