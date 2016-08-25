import os
from keras.optimizers import Adam, Nadam, SGD

preprocessing_config = {
            'process_train': False,
            'process_test': False,
            'arrange_directories': True,
            'y_path': os.path.join('data', 'trainLabels.csv'),
            'size': (256, 256),
            'input_dir_train': os.path.abspath(os.path.join('E:', 'DR_Data', 'raw_train')),
            'input_dir_test': os.path.abspath(os.path.join('E:', 'DR_Data', 'raw_test')),
            'output_dir_train': os.path.abspath(os.path.join('E:', 'DR_Data', 'train_256')),
            'output_dir_test': os.path.abspath(os.path.join('E:', 'DR_Data', 'test_256')),
            'split': 0.8,
            'random_state': 1337
}

training_config = {
            'model_name': 'resnetv1_256',
            'prototype_model': False,
            'continue_training': False,
            'lower_lr': False,
            'optimizer': Adam(lr=0.001), #eg Adam(lr=0.0001), Nadam(0.0001),  SGD(lr=0.001, decay=0.0001, momentum=0.9, nesterov=True)
            'loss': 'mean_squared_error',
            'batch_size': 12,
            'nb_epoch': 1,
            'train_data_dir': os.path.abspath(os.path.join('E:', 'DR_Data', 'train_small')),
            'validation_data_dir': os.path.abspath(os.path.join('E:', 'DR_Data', 'validation_small')),
            'test_data_dir': os.path.abspath(os.path.join('E:', 'DR_Data', 'test_small')),
            'class_weight': None#{0:0.27218907, 1:2.8756447, 2:1.32751323, 3:8.04719359, 4:9.92259887} # based on http://gking.harvard.edu/files/0s.pdf (prior correction)
}

predict_config = {
            'prototype_predict': False,
}
