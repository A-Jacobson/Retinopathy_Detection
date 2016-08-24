import os
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from models.vgg_nets import conv1, conv0
from models.resnets import resnet_v1
from config import training_config, preprocessing_config
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

size = preprocessing_config['size']
config = training_config

nb_train_samples = 22481
nb_validation_samples = 5620

earlystop = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(
            filepath=os.path.join('models',
            'saved_models',
            config['model_name']+'.hdf5'),
            verbose=1,
            save_best_only=False,
            monitor='val_loss')

#load model
if config['continue_training'] == True:
    model = load_model(os.path.join('models', 'saved_models', config['model_name']+'.hdf5'))
    if config['lower_lr'] == True:
        current_lr =  model.optimizer.lr.get_value()
        new_lr = current_lr / 10.
        print "current_lr: ", current_lr
        print "new_lr:", current_lr / 10.
        model.optimizer.lr.set_value(new_lr)

else:
    model = resnet_v1(optimizer=config['optimizer'], loss=config['loss'])


if config['prototype_model'] == True:
    # checks if model overfits one sample
    X_sample = np.load(os.path.join('data', 'X_sample.npy'))
    y_sample = np.load(os.path.join('data', 'y_sample.npy'))
    y_sample = np_utils.to_categorical(y_sample, 5)
    history = model.fit(
            X_sample,
            y_sample,
            batch_size=config['batch_size'],
            nb_epoch=config['nb_epoch'],
            verbose=1,
            callbacks=[checkpointer],
            class_weight=config['class_weight'])
else:

    gen_sample_X = np.load(os.path.join('data', 'gen_sample_X.npy'))

    train_datagen = ImageDataGenerator(
                samplewise_center=True,
                zoom_range=0.1,
                horizontal_flip=True,
                vertical_flip=True,
                rotation_range=10,
                rescale=1/255.
                )

    test_datagen = ImageDataGenerator(
                samplewise_center=True,
                rescale=1/255.
                )

    train_generator = train_datagen.flow_from_directory(
            config['train_data_dir'],
            target_size=preprocessing_config['size'],
            batch_size=config['batch_size'],
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            config['validation_data_dir'],
            target_size=preprocessing_config['size'],
            batch_size=config['batch_size'],
            class_mode='categorical')

    history = model.fit_generator(
            train_generator,
            samples_per_epoch=nb_train_samples,
            nb_epoch=config['nb_epoch'],
            validation_data=validation_generator,
            nb_val_samples=nb_validation_samples,
            callbacks=[checkpointer, earlystop],
            nb_worker=4)

# list all data in history

if config['continue_training'] == True:
    with open(os.path.join('training_history', config['model_name']+'.csv'), 'a') as f:
        pd.DataFrame(history.history).to_csv(f, header=False)

else:
    pd.DataFrame(history.history).to_csv(os.path.join('training_history', config['model_name']+'.csv'))

# summarize history for accuracy
plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
