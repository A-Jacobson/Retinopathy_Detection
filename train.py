from models.resnet50 import ResNet50
from keras.preprocessing import image
from models.imagenet_utils import preprocess_input, decode_predictions
from keras.utils import np_utils
import numpy as np
import os

img_path = os.path.join('data', 'samples', '10_left.jpeg')
img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)
print x.shape
x = np.expand_dims(x, axis=0) # because the convnet expects a batch of images
print x.shape
y = np.array([0])
y = np_utils.to_categorical(y, 5)

y = np.expand_dims(y, axis=0) # because the convnet expects a batch of images
y = np.expand_dims(y, axis=0) # because the convnet expects a batch of images
print y.shape

model = ResNet50(weights=None, include_top=False)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x, y)
