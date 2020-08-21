import numpy as np
import random
from .config import config

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

def VGG16_conv(in_ch, out_ch, use_maxpool = True):
    conv = Sequential()
    conv.add(Conv2D(filters=out_ch, kernel_size=3, strides=1, 
        padding="same"))
    conv.add(BatchNormalization(axis=3))
    conv.add(Activation("relu"))
    
    if use_maxpool:
        conv.add(MaxPooling2D())
    conv.add(Dropout(rate=0.1))
    return conv

def DoubleConv(in_ch, out_ch):
    conv = Sequential([
            Conv2D(filters=out_ch*2, kernel_size=1, strides=1, padding="same"),
            BatchNormalization(axis=3),
            Activation("relu"),
            Conv2D(filters=out_ch, kernel_size=3, padding="same"),
            BatchNormalization(axis=3),
            Activation("relu"),
            Dropout(rate=0.1)
        ])
    return conv

def Self_attention(in_ch,inputs):
    return inputs
    
    f = Conv2DTranspose(filters=in_ch, kernel_size=1, strides=1, padding="same",data_format="channels_last")(inputs)

    g = Conv2D(filters=in_ch, kernel_size=1, strides=1, padding="same")(inputs)

    h = Conv2D(filters=in_ch, kernel_size=1, strides=1, padding="same")(inputs)

    sa = Multiply()([Multiply()([f,g]),h])

    sa = Conv2D(filters=in_ch, kernel_size=1, strides=1, padding="same")(sa)

    sa = Multiply()([inputs,sa])

    return sa

# def craft():
#     inputs = Input(shape=(None, None, 1))

#     conv1 = VGG16_conv(1, 16)(inputs)
#     conv2 = VGG16_conv(16,32, False)(conv1)

#     conv3 = VGG16_conv(32,64)(conv2)

#     conv4 = VGG16_conv(64,128)(conv3)

#     conv5 = VGG16_conv(128,128)(conv4)

#     conv6 = VGG16_conv(128,128, False)(conv5)
    
#     upconv1 = UpSampling2D(size=(2,2),interpolation='bilinear')(DoubleConv(128+128, 64)(concatenate([conv5, conv6], axis=3)))
    
#     upconv2 = UpSampling2D(size=(2,2),interpolation='bilinear')(DoubleConv(64+128, 64)(concatenate([upconv1, conv4], axis=3)))

#     upconv3 = UpSampling2D(size=(2,2),interpolation='bilinear')(DoubleConv(32+64, 16)(concatenate([upconv2, conv3], axis=3)))

#     upconv4 = DoubleConv(16+32, 8)(concatenate([upconv3, conv2], axis=3))

#     final_conv = Conv2D(filters=1, kernel_size=3, padding="same")(concatenate([upconv4, conv1], axis=3))

#     model = Model(input = inputs, output = final_conv)
#     model.compile(optimizer = Adadelta(lr = config.LEARNING_RATE), loss = 'mean_squared_error')
#     print("Model compiled successfully.")
#     model.summary()
#     return model


def craft():
    inputs = Input(shape=(None, None, 1))

    conv1 = VGG16_conv(1, 16)(inputs)

    sa1 = Self_attention(16,conv1)

    conv2 = VGG16_conv(16,32, False)(conv1)

    sa2 = Self_attention(32,conv2)

    conv3 = VGG16_conv(32,64)(conv2)

    sa3 = Self_attention(64,conv3)

    conv4 = VGG16_conv(64,128)(conv3)

    sa4 = Self_attention(128,conv4)

    conv5 = VGG16_conv(128,128)(conv4)

    sa5 = Self_attention(128,conv5)

    conv6 = VGG16_conv(128,128, False)(conv5)

    sa6 = Self_attention(128,conv6)
    
    upconv1 = UpSampling2D(size=(2,2),interpolation='bilinear')(DoubleConv(128+128, 64)(concatenate([sa5, sa6], axis=3)))
    
    upconv2 = UpSampling2D(size=(2,2),interpolation='bilinear')(DoubleConv(64+128, 64)(concatenate([upconv1, sa4], axis=3)))

    upconv3 = UpSampling2D(size=(2,2),interpolation='bilinear')(DoubleConv(32+64, 16)(concatenate([upconv2, sa3], axis=3)))

    upconv4 = DoubleConv(16+32, 8)(concatenate([upconv3, conv2], axis=3))
    final_conv1 = Conv2D(filters=8, kernel_size=3, padding="same", activation="relu")(upconv4)
    final_conv2 = Conv2D(filters=8, kernel_size=3, padding="same", activation="relu")(final_conv1)
    final_conv3 = Conv2D(filters=4, kernel_size=3, padding="same", activation="relu")(final_conv2)
    final_conv4 = Conv2D(filters=1, kernel_size=3, padding="same", activation="relu")(final_conv3)

    model = Model(input = inputs, output = final_conv4)
    model.compile(optimizer = Adadelta(lr = config.LEARNING_RATE), loss = 'mean_squared_error')
    print("Model compiled successfully.")
#    model.summary()
    return model

if __name__ == "__main__":
    model = craft()
