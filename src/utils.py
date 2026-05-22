"""
Utility functions for the project.
"""

import pandas as pd
import os, time, logging

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"*** {func.__name__} took {elapsed_time/60:.4f} mins to run.")
        return result
    return wrapper

def setup_logger(log_file_path):
    log_formatter = logging.Formatter('[%(levelname)s]: %(message)s')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(log_formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
    return logger

def print_or_log(logger, message):
    if logger:
        logger.info(message)
    else:
        print(message)

def get_file_paths(folder_path, end_pattern=".xml", logger=None):
    file_paths = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(end_pattern):
            file_path = os.path.join(folder_path, file_name)
            file_paths.append(file_path)

    print_or_log(logger, f"* found {len(file_paths)} {end_pattern} files in {folder_path}!")
    return file_paths

@timing_decorator
def load_df(file_path, logger=None):
    df = pd.read_csv(file_path, sep='\t')
    print_or_log(logger, f"* loaded {len(df)} records from {file_path}!")
    return df

@timing_decorator
def save_df(df, file_path, logger=None):
    df.to_csv(file_path, sep='\t', index=False)
    print_or_log(logger, f"* saved {len(df)} records to {file_path}!")
    return df