from keras.models import Sequential, load_model
from keras.utils import np_utils
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from config import training_config
import pandas as pd
import glob

config = training_config

test_datagen = ImageDataGenerator(
validation_data_dir = config['validation_data_dir'])

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(512, 512),
        batch_size=5,
        class_mode=None)

X_sample = np.load(os.path.join('data', 'X_sample.npy'))
y_sample = np.load(os.path.join('data', 'y_sample.npy'))
y_sample = np_utils.to_categorical(y_sample, 5)

model = load_model(os.path.join("models", "saved_models", "conv1_1.hdf5"))

print np.argmax(model.predict_generator(validation_generator, 10), axis=1)

# get image id's

# img_ids = pd.read_csv(list(set(glob.glob(os.path.join(img_dir, "*.jpeg")))))
