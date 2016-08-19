from data_utils import image_funcs
import os

# resize train to 512
input_dir_train = os.path.abspath(os.path.join("E:", "DR_Data", "Train"))
output_dir_train = os.path.abspath(os.path.join("E:", "DR_Data", "train_512"))
image_funcs.resize_directory(input_dir_train, output_dir_train)

# resize test
input_dir_test = os.path.abspath(os.path.join("E:", "DR_Data", "Test"))
output_dir_test = os.path.abspath(os.path.join("E:", "DR_Data", "test_512"))
image_funcs.resize_directory(input_dir_test, output_dir_test)

# make (N, channels, width, height) matrices
image_funcs.matrix_from_dir(output_dir_train, name='X_train')
image_funcs.matrix_from_dir(output_dir_train, name='X_test')
