
import os
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from models.conv_nets import conv1, conv0
from config import training_config
from keras.preprocessing.image import ImageDataGenerator

config = training_config

img_width, img_height = 512, 512

train_data_dir = config['train_data_dir']
validation_data_dir = config['validation_data_dir']
nb_train_samples = 22481
nb_validation_samples = 5620
nb_epoch = 1
batch_size = 5

# load data for sample training
# X_sample = np.load(os.path.join('data', 'train', 'X_sample.npy'))
# y_sample = np.load(os.path.join('data', 'train', 'y_sample.npy'))

# encode labels
# y_sample = np_utils.to_categorical(y_sample, 5)

#load model
model = conv1()
earlystop = EarlyStopping(monitor='loss', patience=10, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(filepath=os.path.join("models", "saved_models", "conv1_1.hdf5"), verbose=1, save_best_only=False, monitor='loss')
# history = model.fit(X_sample[:1], y_sample[:1], batch_size=5, nb_epoch=20, verbose=1, callbacks=[checkpointer, earlystop])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

history = model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples,
        callbacks=[checkpointer, earlystop])

print history.history['loss']
