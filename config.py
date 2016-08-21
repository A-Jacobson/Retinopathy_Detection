import os

preprocessing_config = {
            "process_train": True,
            "process_test": True,
            "arrange_directories": True,
            "y_path": os.path.join('data', 'trainLabels.csv'),
            "input_dir_train" = os.path.abspath(os.path.join("E:", "DR_Data", "Train")),
            "output_dir_train" = os.path.abspath(os.path.join("E:", "DR_Data", "train_512")),
            "input_dir_test" = os.path.abspath(os.path.join("E:", "DR_Data", "Test")),
            "output_dir_test" = os.path.abspath(os.path.join("E:", "DR_Data", "test_512")),
            "split": 0.8,
            "random_state": 1337
}

training_config = {}
