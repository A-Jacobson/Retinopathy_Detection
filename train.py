
import os
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from model_utils.conv_nets import conv1

img_rows, img_cols = 512, 512
img_channels = 3
nb_classes = 5

X_sample = np.load(os.path.join('data', 'train', 'X_sample.npy'))
y_sample = np.load(os.path.join('data', 'train', 'y_sample.npy'))
y_sample = np_utils.to_categorical(y_sample, nb_classes)

model = conv1()

earlystop = EarlyStopping(monitor='loss', patience=2, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(filepath=os.path.join("models", "tmp", "weights.hdf5"), verbose=1, save_best_only=False, monitor='loss')
history = model.fit(X, y, batch_size=5, nb_epoch=20, verbose=1, callbacks=[checkpointer, earlystop])
print model.predict(X_sample[:1])
