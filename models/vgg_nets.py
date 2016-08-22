from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout

def conv0(input_shape=(3, 512, 512), nb_classes=5, activation='relu', optimizer='adam', loss='categorical_crossentropy'):
    model = Sequential()
    model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Convolution2D(16, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Convolution2D(32, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # classification block
    model.add(Flatten())
    model.add(Dense(1028))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(optimizer=optimizer, loss=loss)
    return model


# first model 13 layer convnet
def conv1(input_shape=(3, 512, 512), nb_classes=5, activation='relu', optimizer='adam', loss='categorical_crossentropy'):
    model = Sequential()
    model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Convolution2D(16, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Convolution2D(32, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Convolution2D(64, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Convolution2D(128, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(256, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Convolution2D(256, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # classification block
    model.add(Flatten())
    model.add(Dense(1028))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model
