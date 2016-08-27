import os
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from config import training_config, predict_config
from data_utils.data_funcs import write_answer

config = predict_config

if config['prototype_predict'] == True:
    X_sample = np.load(os.path.join('data', 'X_sample.npy'))
    y_sample = np.load(os.path.join('data', 'y_sample.npy'))
    y_sample = np_utils.to_categorical(y_sample, 5)

else:
    test_datagen = ImageDataGenerator(samplewise_center=True)

    test_generator = test_datagen.flow_from_directory(
            training_config['test_data_dir'],
            target_size=(256, 256),
            batch_size=12,
            class_mode=None,
            shuffle=False)

model = load_model(os.path.join("models", "saved_models", training_config['model_name']+".hdf5"))
preds = pd.DataFrame(model.predict_generator(test_generator, 53576, nb_worker=4))
write_answer(preds, training_config['test_data_dir'])
