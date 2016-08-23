import os
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from config import training_config, predict_config

config = predict_config

if config['prototype_predict'] == True:
    X_sample = np.load(os.path.join('data', 'X_sample.npy'))
    y_sample = np.load(os.path.join('data', 'y_sample.npy'))
    y_sample = np_utils.to_categorical(y_sample, 5)

else:
    test_datagen = ImageDataGenerator(samplewise_center=True)

    test_generator = test_datagen.flow_from_directory(
            training_config['test_data_dir'],
            target_size=(512, 512),
            batch_size=5,
            class_mode=None,
            shuffle=False)

model = load_model(os.path.join("models", "saved_models", training_config['model_name']+".hdf5"))
print model.predict(X_sample)
print np.argmax(model.predict(X_sample), axis=1)
print y_sample


# gen
 pd.DataFrame(np.argmax(model.predict_generator(validation_generator, 10), axis=1)).to_csv('preds.csv')

# get image id's

# img_ids = pd.read_csv(list(set(glob.glob(os.path.join(img_dir, "*.jpeg")))))
