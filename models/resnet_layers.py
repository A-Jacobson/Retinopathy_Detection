# adapted from:
# https://github.com/raghakot/keras-resnet/blob/master/resnet.py,
# https://github.com/fchollet/keras/blob/master/examples/resnet_50.py
# https://github.com/KaimingHe

# based on:
# http://arxiv.org/abs/1512.03385
from keras.layers import merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input


# The names of layers in resnet50 are generated with the following format
# [type][stage][block]_branch[branch][layer]
# type: 'res' for conv layer, 'bn' and 'scale' for BN layer
# stage: from '2' to '5', current stage number
# block: 'a','b','c'... for different blocks in a stage
# branch: '1' for shortcut and '2' for main path
# layer: 'a','b','c'... for different layers in a block


def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    out = BatchNormalization(axis=1, name=bn_name_base + '2a')(input_tensor)
    out = Activation('relu')(out)
    out = Convolution2D(nb_filter1, 1, 1, init='he_normal', name=conv_name_base + '2a')(out)

    out = BatchNormalization(axis=1, name=bn_name_base + '2b')(out)
    out = Activation('relu')(out)
    out = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same', init='he_normal', name=conv_name_base + '2b')(out)

    out = BatchNormalization(axis=1, name=bn_name_base + '2c')(out)
    out = Convolution2D(nb_filter3, 1, 1, init='he_normal', name=conv_name_base + '2c')(out)

    out = merge([out, input_tensor], mode='sum') # always merge input with output of the block to retain information before transformations?
    return out


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    out = BatchNormalization(axis=1, name=bn_name_base + '2a')(input_tensor)
    out = Activation('relu')(out)
    out = Convolution2D(nb_filter1, 1, 1, init='he_normal', subsample=strides, name=conv_name_base + '2a')(out)

    out = BatchNormalization(axis=1, name=bn_name_base + '2b')(out)
    out = Activation('relu')(out)
    out = Convolution2D(nb_filter2, kernel_size, kernel_size, init='he_normal', border_mode='same', name=conv_name_base + '2b')(out)

    out = BatchNormalization(axis=1, name=bn_name_base + '2c')(out)
    out = Convolution2D(nb_filter3, 1, 1, init='he_normal', name=conv_name_base + '2c')(out)

    shortcut = BatchNormalization(axis=1, name=bn_name_base + '1')(input_tensor)
    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides, init='he_normal', name=conv_name_base + '1')(shortcut)

    out = merge([out, shortcut], mode='sum')
    return out
