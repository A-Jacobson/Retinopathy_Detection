from keras.preprocessing.image import ImageDataGenerator
import os
from config import training_config




# test_datagen = ImageDataGenerator(samplewise_center=False)
#
# test_generator = test_datagen.flow_from_directory(
#         training_config['test_data_dir'],
#         target_size=(256, 256),
#         batch_size=5,
#         class_mode=None,
#         shuffle=False,
#         save_to_dir=os.path.join('samples'))
#
# print test_generator.filenames

print os.listdir(os.path.join(training_config['test_data_dir'], 'unknown'))
