from keras import Model
from keras.models import Sequential
from keras.layers import Conv2D, UpSampling2D, Convolution2D, Input, MaxPooling2D, Lambda, Add, Activation
import tensorflow as tf
from keras.regularizers import l2


class Fully_Convolutional_Network:
    def __init__(self, width, height, classes=1, batches=False, weight_decay: float=0.0):
        self.model = Sequential()
        self.input_shape = (height, width)
        self.classes = classes
        if batches:
            self.batch_input_shape = (batches, height, width)
        self.batches = True if batches else False
        self.weight_decay = weight_decay

    def resize_bilinear(self, imgs):
        return tf.image.resize_bilinear(imgs, [self.input_shape[0], self.input_shape[1]])

    def create_model(self, input_tensor):
        if self.batches:
            input = Input(batch_shape=self.batch_input_shape)
        else:
            input = Input(self.input_shape)
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1', kernel_regularizer=l2(self.weight_decay))(input)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2', kernel_regularizer=l2(self.weight_decay))(conv1)
        pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv2)  # size = 64x64
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3', kernel_regularizer=l2(self.weight_decay))(pool1)
        conv4 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv4', kernel_regularizer=l2(self.weight_decay))(conv3)
        pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv4) # size = 32x32
        conv5 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv5', kernel_regularizer=l2(self.weight_decay))(pool2)
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv6', kernel_regularizer=l2(self.weight_decay))(conv5)
        conv7 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv7', kernel_regularizer=l2(self.weight_decay))(conv6)
        pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv7) # size = 16x16
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv8', kernel_regularizer=l2(self.weight_decay))(pool3)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv9', kernel_regularizer=l2(self.weight_decay))(conv8)
        pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv9) # size = 8x8
        conv10 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv10', kernel_regularizer=l2(self.weight_decay))(pool4)
        r8 = Lambda(self.resize_bilinear)(conv10)
        r16 = Lambda(self.resize_bilinear)(conv8)
        r32 = Lambda(self.resize_bilinear)(conv5)
        summed = Add()([r32, r16, r8])
        softmax = Activation('softmax')(summed)
        self.model = Model(input=input_tensor, output=softmax)




