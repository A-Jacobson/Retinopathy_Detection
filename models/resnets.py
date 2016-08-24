from models.resnet_layers import conv_block, identity_block
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import SGD, Adam


def resnet_v1(input_shape=(3, 512, 512), nb_classes=5, optimizer=Adam(), loss='categorical_crossentropy'):
    '''
    resnet adaptation for variable image shapes/number of classes

    '''

    inp = Input(shape=input_shape)
    out = Convolution2D(64, 7, 7, subsample=(2, 2), init='he_normal', name='conv1')(inp)
    out = BatchNormalization(axis=1, name='bn_conv1')(out)
    out = Activation('relu')(out)
    out = MaxPooling2D((3, 3), strides=(2, 2))(out)

    out = conv_block(out, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    out = identity_block(out, 3, [64, 64, 256], stage=2, block='b')
    out = identity_block(out, 3, [64, 64, 256], stage=2, block='c')

    out = conv_block(out, 3, [128, 128, 512], stage=3, block='a')
    out = identity_block(out, 3, [128, 128, 512], stage=3, block='b')
    out = identity_block(out, 3, [128, 128, 512], stage=3, block='c')
    out = identity_block(out, 3, [128, 128, 512], stage=3, block='d')

    out = conv_block(out, 3, [256, 256, 1024], stage=4, block='a')
    out = identity_block(out, 3, [256, 256, 1024], stage=4, block='b')
    out = identity_block(out, 3, [256, 256, 1024], stage=4, block='c')
    out = identity_block(out, 3, [256, 256, 1024], stage=4, block='d')
    out = identity_block(out, 3, [256, 256, 1024], stage=4, block='e')
    out = identity_block(out, 3, [256, 256, 1024], stage=4, block='f')

    out = conv_block(out, 3, [512, 512, 2048], stage=5, block='a')
    out = identity_block(out, 3, [512, 512, 2048], stage=5, block='b')
    out = identity_block(out, 3, [512, 512, 2048], stage=5, block='c')

    out = AveragePooling2D((7, 7))(out)
    out = Flatten()(out)
    out = Dense(nb_classes, init='he_normal', activation='softmax', name='fc10')(out)

    model = Model(inp, out)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
        )
    return model


    def resnet_v2(input_shape=(3, 512, 512), nb_classes=5, optimizer=Adam(), loss='categorical_crossentropy'):
        '''
        resnet adaptation for variable image shapes/number of classes

        '''

        inp = Input(shape=input_shape)
        out = Convolution2D(64, 7, 7, subsample=(2, 2), init='he_normal', name='conv1')(inp)
        out = BatchNormalization(axis=1, name='bn_conv1')(out)
        out = Activation('relu')(out)
        out = MaxPooling2D((3, 3), strides=(2, 2))(out)

        out = conv_block(out, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        out = identity_block(out, 3, [64, 64, 256], stage=2, block='b')
        out = identity_block(out, 3, [64, 64, 256], stage=2, block='c')

        out = conv_block(out, 3, [128, 128, 512], stage=3, block='a')
        out = identity_block(out, 3, [128, 128, 512], stage=3, block='b')
        out = identity_block(out, 3, [128, 128, 512], stage=3, block='c')
        out = identity_block(out, 3, [128, 128, 512], stage=3, block='d')

        out = conv_block(out, 3, [256, 256, 1024], stage=4, block='a')
        out = identity_block(out, 3, [256, 256, 1024], stage=4, block='b')
        out = identity_block(out, 3, [256, 256, 1024], stage=4, block='c')
        out = identity_block(out, 3, [256, 256, 1024], stage=4, block='d')
        out = identity_block(out, 3, [256, 256, 1024], stage=4, block='e')
        out = identity_block(out, 3, [256, 256, 1024], stage=4, block='f')

        out = AveragePooling2D((7, 7))(out)
        out = Flatten()(out)
        out = Dense(nb_classes, init='he_normal', activation='softmax', name='fc10')(out)

        model = Model(inp, out)

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
            )
        return model
