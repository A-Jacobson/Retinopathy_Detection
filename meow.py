from data_utils.data_funcs import get_labels
# from keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd


# idx_label =  get_labels(os.path.abspath(os.path.join('E:', 'DR_Data', 'test')))
#
# test_datagen = ImageGenerator()
#
# test_generator = test_datagen.flow_from_directory(
#         config['test_data_dir'],
#         target_size=(512, 512),
#         batch_size=5,
#         class_mode=None,
#         shuffle=False)
#
# print test_generator

loss = pd.read_csv(os.path.join('training_history', 'resnetv1_2.csv'))
print loss
