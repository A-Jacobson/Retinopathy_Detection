
import os
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from models.conv_nets import conv1
from data_utils.image_funcs import ImageGenerator

X_train_path =

# load data
X_sample = np.load(os.path.join('data', 'train', 'X_sample.npy'))
y_sample = np.load(os.path.join('data', 'train', 'y_sample.npy'))

# encode labels
y_sample = np_utils.to_categorical(y_sample, 5)

#load model
model = conv1()

earlystop = EarlyStopping(monitor='loss', patience=10, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(filepath=os.path.join("models", "saved_models", "conv1_1.hdf5"), verbose=1, save_best_only=False, monitor='loss')
history = model.fit(X_sample[:1], y_sample[:1], batch_size=5, nb_epoch=20, verbose=1, callbacks=[checkpointer, earlystop])
print model.predict_classes(X_sample[:1])
print np.argmax(y_sample[0])

print history.history['loss']
