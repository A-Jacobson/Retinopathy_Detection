
import os
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from models.vgg_nets import conv1, conv0
from models.resnets import resnet_v1
from config import training_config
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

config = training_config

img_width, img_height = 512, 512

train_data_dir = config['train_data_dir']
validation_data_dir = config['validation_data_dir']
nb_train_samples = 22481
nb_validation_samples = 5620
nb_epoch = 1
batch_size = 5
continue_training = False
prototype_model = False


#load model
if continue_training == True:
    model = load_model(os.path.join("models", "saved_models", "conv1_1.hdf5"))

else:
    model = resnet_v1()


earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(filepath=os.path.join("models", "saved_models", "resnetv1_1.hdf5"), verbose=1, save_best_only=False, monitor='val_loss')

if prototype_model == True:
    # checks if model overfits one sample
    X_sample = np.load(os.path.join('data', 'X_sample.npy'))
    y_sample = np.load(os.path.join('data', 'y_sample.npy'))
    y_sample = np_utils.to_categorical(y_sample, 5)
    history = model.fit(X_sample[:1], y_sample[:1], batch_size=5, nb_epoch=20, verbose=1, callbacks=[checkpointer, earlystop])
else:
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

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
            callbacks=[checkpointer, earlystop],
            nb_worker=3)


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
