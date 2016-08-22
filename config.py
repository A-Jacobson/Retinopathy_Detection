import os

preprocessing_config = {
            "process_train": False,
            "process_test": False,
            "arrange_directories": False,
            "y_path": os.path.join('data', 'trainLabels.csv'),
            "size": (512, 512)
            "input_dir_train": os.path.abspath(os.path.join("E:", "DR_Data", "raw_train")),
            "input_dir_test": os.path.abspath(os.path.join("E:", "DR_Data", "raw_test")),
            "output_dir_train": os.path.abspath(os.path.join("E:", "DR_Data", "train_512")),
            "output_dir_test": os.path.abspath(os.path.join("E:", "DR_Data", "test_512")),
            "split": 0.8,
            "random_state": 1337
}

training_config = {
            "train_data_dir": os.path.abspath(os.path.join("E:", "DR_Data", "train")),
            "validation_data_dir": os.path.abspath(os.path.join("E:", "DR_Data", "validation")),
            "test_data_dir": os.path.abspath(os.path.join("E:", "DR_Data", "train"))
}
