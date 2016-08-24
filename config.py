import os
from keras.optimizers import Adam, Nadam, SGD

preprocessing_config = {
            'process_train': True,
            'process_test': True,
            'arrange_directories': False,
            'y_path': os.path.join('data', 'trainLabels.csv'),
            'size': (512, 512),
            'input_dir_train': os.path.abspath(os.path.join('E:', 'DR_Data', 'raw_train')),
            'input_dir_test': os.path.abspath(os.path.join('E:', 'DR_Data', 'raw_test')),
            'output_dir_train': os.path.abspath(os.path.join('E:', 'DR_Data', 'train_512')),
            'output_dir_test': os.path.abspath(os.path.join('E:', 'DR_Data', 'test_512')),
            'split': 0.8,
            'random_state': 1337
}

training_config = {
            'model_name': 'resnetv1_Nadam_norm',
            'prototype_model': False,
            'continue_training': False,
            'lower_lr': False,
            'optimizer': Nadam(lr=0.0001), #eg Adam(lr=0.0001), Nadam(0.0001), SGD(lr=0.001)
            'loss': 'categorical_crossentropy',
            'batch_size': 3,
            'nb_epoch': 5,
            'train_data_dir': os.path.abspath(os.path.join('E:', 'DR_Data', 'train')),
            'validation_data_dir': os.path.abspath(os.path.join('E:', 'DR_Data', 'validation')),
            'test_data_dir': os.path.abspath(os.path.join('E:', 'DR_Data', 'test')),
            'class_weight': {0:0.27218907, 1:2.8756447, 2:1.32751323, 3:8.04719359, 4:9.92259887} # based on http://gking.harvard.edu/files/0s.pdf (prior correction)
}

predict_config = {
            'prototype_predict': True,
}
