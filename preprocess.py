from data_utils.image_funcs import ImagePreProcessor, ImageGenerator
import os
from data_utils.data_funcs import train_test_val_split, arrange_directories
import pandas as pd
from config import preprocessing_config

config = preprocessing_config
pre = ImagePreProcessor()

# resize preprocess train
if config['process_train'] == True:
    pre.preprocess_directory(config['input_dir_train'], config['output_dir_train'], size=config['size'])

# preprocess test
if config['process_test'] == True:
    pre.preprocess_directory(config['input_dir_test'], config['output_dir_test'], size=config['size'])

#split training directories for keras flow from directory
if config['arrange_directories'] == True:
    y = pd.read_csv(config['y_path'])
    train, test, val = train_test_val_split(y, random_state=config['random_state'], split=config['split'])
    arrange_directories(test, 'test')
    arrange_directories(val, 'validation')
