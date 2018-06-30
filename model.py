import os
from keras.regularizers import l2
from keras.layers import *
from keras.models import *

from BilinearUpSampling import BilinearUpSampling2D


def AtrousFCN_Vgg16_16s(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(x)
    o2 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay))(x)
    o3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay))(o3)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(weight_decay))(x)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(2048, (7, 7), activation='relu', padding='same', dilation_rate=(2, 2),
                      name='fc1', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(2048, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)

    #classifying layer
    out_1 = Conv2D(32, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid',
                        strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    out_1 = BilinearUpSampling2D(target_size=tuple(image_size))(out_1)

    out_2 = Conv2D(2, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid',
                        strides=(1, 1), kernel_regularizer=l2(weight_decay))(o2)
    out_2 = BilinearUpSampling2D(target_size=tuple(image_size))(out_2)

    out_3 = Conv2D(8, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid',
                        strides=(1, 1), kernel_regularizer=l2(weight_decay))(o3)
    out_3 = BilinearUpSampling2D(target_size=tuple(image_size))(out_3)

    out = Concatenate()([out_1, out_2, out_3])

    out = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid',
                        strides=(1, 1), kernel_regularizer=l2(weight_decay))(out)

    model = Model(img_input, out)

    weights_path = os.path.expanduser(os.path.join('./model/model.hdf5'))
    model.load_weights(weights_path, by_name=True)
    return model