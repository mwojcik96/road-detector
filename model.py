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
    o1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay))(o1)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay))(x)
    o2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(o2)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    o3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay))(o3)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='fc1', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='fc3', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    out_4 = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block_deconv11', kernel_regularizer=l2(weight_decay))(x)

    out_4 = BilinearUpSampling2D(target_size=(16, 16))(out_4)

    out_3 = Concatenate()([o3, out_4])

    out_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block_deconv21', kernel_regularizer=l2(weight_decay))(out_3)
    out_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block_deconv22',
                   kernel_regularizer=l2(weight_decay))(out_3)

    out_3 = BilinearUpSampling2D(target_size=(32, 32))(out_3)

    out_2 = Concatenate()([o2, out_3])

    out_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block_deconv31', kernel_regularizer=l2(weight_decay))(out_2)
    out_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block_deconv32',
                   kernel_regularizer=l2(weight_decay))(out_2)

    out_2 = BilinearUpSampling2D(target_size=(64, 64))(out_2)

    out_1 = Concatenate()([o1, out_2])

    out_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block_deconv41', kernel_regularizer=l2(weight_decay))(out_1)
    out_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block_deconv42',
                   kernel_regularizer=l2(weight_decay))(out_1)

    out_1 = BilinearUpSampling2D(target_size=(128, 128))(out_1)

    out = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='sigmoid', padding='valid',
                        strides=(1, 1), kernel_regularizer=l2(weight_decay))(out_1)

    model = Model(img_input, out)

    weights_path = os.path.expanduser(os.path.join('./model/model.hdf5'))
    #model.load_weights(weights_path, by_name=True)
    return model