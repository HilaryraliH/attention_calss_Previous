import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K


'''之前模型的备份'''

# def unet(pretrained_weights = None,input_size = (9,200,1)):
#     inputs = Input(input_size)
#     BN0 = BatchNormalization()(inputs)
#     conv1 = Conv2D(32, (1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(BN0)
#     conv1 = Conv2D(32, (1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
#     pool1 = MaxPooling2D(pool_size=(1, 2))(conv1)
#
#     conv2 = Conv2D(64, (1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
#     conv2 = Conv2D(64, (1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
#     pool2 = MaxPooling2D(pool_size=(1, 2))(conv2)
#     conv3 = Conv2D(128, (1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
#     conv3 = Conv2D(128, (1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
#     pool3 = MaxPooling2D(pool_size=(1, 2))(conv3)
#     conv4 = Conv2D(256, (1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
#     conv4 = Conv2D(256, (1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
#     drop4 = Dropout(0.5)(conv4)
#     # drop4 为 9*25
#     # pool4 = MaxPooling2D(pool_size=(1, 2))(drop4) # 这里pool之后，是12.5，也就是12了，之后上采样变成24就会对不齐
#     # 可以省略这个pool，并相应省略之后的上采样部分
#     conv5 = Conv2D(512, (1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop4)
#     conv5 = Conv2D(512, (1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
#     drop5 = Dropout(0.5)(conv5)
#     print('drop5.shape: ',drop5.shape)
#     up6 = Conv2D(256, (1,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop5)# UpSampling2D(size = (1,2))(drop5))
#     merge6 = concatenate([drop4,up6], axis = 3)
#     conv6 = Conv2D(256, (1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
#     conv6 = Conv2D(256, (1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
#
#     up7 = Conv2D(128, (1,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (1,2))(conv6))
#     merge7 = concatenate([conv3,up7], axis = 3)
#     conv7 = Conv2D(128, (1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
#     conv7 = Conv2D(128, (1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
#
#     up8 = Conv2D(64, (1,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (1,2))(conv7))
#     merge8 = concatenate([conv2,up8], axis = 3)
#     conv8 = Conv2D(64, (1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
#     conv8 = Conv2D(64, (1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
#
#     up9 = Conv2D(32, (1,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (1,2))(conv8))
#     merge9 = concatenate([conv1,up9], axis = 3)
#     conv9 = Conv2D(32, (1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
#     conv9 = Conv2D(32, (1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#     conv9 = Conv2D(2, (1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#     conv10 = Conv2D(1, 1, activation = 'tanh')(conv9)
#
#     model = Model(input = inputs, output = conv10)
#
#     model.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_squared_error', metrics = ['mse'])
#
#     #model.summary()
#
#     if(pretrained_weights):
#     	model.load_weights(pretrained_weights)
#
#     print('In function: Unet (the way to create Unet)')
#     model.summary()
#
#     return model




'''----------'''
# 每个层都加了BN的
def unet(pretrained_weights=None, input_size=(9, 200, 1)):
    inputs = Input(input_size)
    #BN0 = BatchNormalization()(inputs)
    conv1 = Conv2D(32, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    BN1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN1)
    BN1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(1, 2))(BN1)

    conv2 = Conv2D(64, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    BN2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN2)
    BN2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(1, 2))(BN2)

    conv3 = Conv2D(128, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    BN3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN3)
    BN3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(1, 2))(BN3)

    conv4 = Conv2D(256, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    BN4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN4)
    BN4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(BN4)
    # drop4 为 9*25
    # pool4 = MaxPooling2D(pool_size=(1, 2))(drop4) # 这里pool之后，是12.5，也就是12了，之后上采样变成24就会对不齐
    # 可以省略这个pool，并相应省略之后的上采样部分
    conv5 = Conv2D(512, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(drop4)
    BN5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN5)
    BN5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(BN5)
    print('drop5.shape: ', drop5.shape)


    up6 = Conv2D(256, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(drop5)  # UpSampling2D(size = (1,2))(drop5))
    BN6 = BatchNormalization()(up6)
    merge6 = concatenate([drop4, BN6], axis=3)
    conv6 = Conv2D(256, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    BN6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN6)
    BN6 = BatchNormalization()(conv6)

    up7 = Conv2D(128, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(1, 2))(BN6))
    BN7 = BatchNormalization()(up7)
    merge7 = concatenate([conv3, BN7], axis=3)
    conv7 = Conv2D(128, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    BN7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN7)
    BN7 = BatchNormalization()(conv7)

    up8 = Conv2D(64, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(1, 2))(BN7))
    BN8 = BatchNormalization()(up8)
    merge8 = concatenate([conv2, BN8], axis=3)
    conv8 = Conv2D(64, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    BN8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN8)
    BN8 = BatchNormalization()(conv8)

    up9 = Conv2D(32, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(1, 2))(BN8))
    BN9 = BatchNormalization()(up9)
    merge9 = concatenate([conv1, BN9], axis=3)
    conv9 = Conv2D(32, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    BN9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN9)
    BN9 = BatchNormalization()(conv9)
    conv9 = Conv2D(2, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN9)
    BN9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(BN9) # 或者用tanh

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=['mse'])

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    print('In function: Unet (the way to create Unet)')
    model.summary()

    return model

'''----------'''
# 降为3层绿色，每个层都加了BN的
# def unet(pretrained_weights=None, input_size=(9, 200, 1)):
#     inputs = Input(input_size)
#     BN0 = BatchNormalization()(inputs)
#     conv1 = Conv2D(32, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN0)
#     BN1 = BatchNormalization()(conv1)
#     conv1 = Conv2D(32, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN1)
#     BN1 = BatchNormalization()(conv1)
#     pool1 = MaxPooling2D(pool_size=(1, 2))(BN1)
#
#     conv2 = Conv2D(64, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
#     BN2 = BatchNormalization()(conv2)
#     conv2 = Conv2D(64, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN2)
#     BN2 = BatchNormalization()(conv2)
#     pool2 = MaxPooling2D(pool_size=(1, 2))(BN2)
#
#     conv3 = Conv2D(128, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
#     BN3 = BatchNormalization()(conv3)
#     conv3 = Conv2D(128, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN3)
#     BN3 = BatchNormalization()(conv3)
#     pool3 = MaxPooling2D(pool_size=(1, 2))(BN3)
#
#     # conv4 = Conv2D(256, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
#     # BN4 = BatchNormalization()(conv4)
#     # conv4 = Conv2D(256, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN4)
#     # BN4 = BatchNormalization()(conv4)
#     # drop4 = Dropout(0.5)(BN4)
#     # drop4 为 9*25
#     # pool4 = MaxPooling2D(pool_size=(1, 2))(drop4) # 这里pool之后，是12.5，也就是12了，之后上采样变成24就会对不齐
#     # 可以省略这个pool，并相应省略之后的上采样部分
#
#     #########################################################################################################
#     conv5 = Conv2D(512, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
#     BN5 = BatchNormalization()(conv5)
#     conv5 = Conv2D(512, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN5)
#     BN5 = BatchNormalization()(conv5)
#     drop5 = Dropout(0.5)(BN5)
#     print('drop5.shape: ', drop5.shape)
#     #########################################################################################################
#
#
#     # up6 = Conv2D(256, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(drop5)  # UpSampling2D(size = (1,2))(drop5))
#     # BN6 = BatchNormalization()(up6)
#     # merge6 = concatenate([drop4, BN6], axis=3)
#     # conv6 = Conv2D(256, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
#     # BN6 = BatchNormalization()(conv6)
#     # conv6 = Conv2D(256, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN6)
#     # BN6 = BatchNormalization()(conv6)
#
#     up7 = Conv2D(128, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(1, 2))(drop5))
#     BN7 = BatchNormalization()(up7)
#     merge7 = concatenate([BN3, BN7], axis=3)
#     conv7 = Conv2D(128, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
#     BN7 = BatchNormalization()(conv7)
#     conv7 = Conv2D(128, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN7)
#     BN7 = BatchNormalization()(conv7)
#
#     up8 = Conv2D(64, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(1, 2))(BN7))
#     BN8 = BatchNormalization()(up8)
#     merge8 = concatenate([BN2, BN8], axis=3)
#     conv8 = Conv2D(64, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
#     BN8 = BatchNormalization()(conv8)
#     conv8 = Conv2D(64, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN8)
#     BN8 = BatchNormalization()(conv8)
#
#     up9 = Conv2D(32, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(1, 2))(BN8))
#     BN9 = BatchNormalization()(up9)
#     merge9 = concatenate([BN1, BN9], axis=3)
#     conv9 = Conv2D(32, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
#     BN9 = BatchNormalization()(conv9)
#     conv9 = Conv2D(32, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN9)
#     BN9 = BatchNormalization()(conv9)
#     conv9 = Conv2D(2, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN9)
#     BN9 = BatchNormalization()(conv9)
#
#     conv10 = Conv2D(1, 1, activation='sigmoid')(BN9) # 或者用tanh
#
#     model = Model(input=inputs, output=conv10)
#     model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=['mse'])
#
#     # model.summary()
#
#     if (pretrained_weights):
#         model.load_weights(pretrained_weights)
#
#     print('In function: Unet (the way to create Unet)')
#     model.summary()
#
#     return model

'''----------'''
# 去掉连接版（4层）
# def unet(pretrained_weights=None, input_size=(9, 200, 1)):
#     inputs = Input(input_size)
#     BN0 = BatchNormalization()(inputs)
#     conv1 = Conv2D(32, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN0)
#     BN1 = BatchNormalization()(conv1)
#     conv1 = Conv2D(32, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN1)
#     BN1 = BatchNormalization()(conv1)
#     pool1 = MaxPooling2D(pool_size=(1, 2))(BN1)
#
#     conv2 = Conv2D(64, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
#     BN2 = BatchNormalization()(conv2)
#     conv2 = Conv2D(64, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN2)
#     BN2 = BatchNormalization()(conv2)
#     pool2 = MaxPooling2D(pool_size=(1, 2))(BN2)
#
#     conv3 = Conv2D(128, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
#     BN3 = BatchNormalization()(conv3)
#     conv3 = Conv2D(128, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN3)
#     BN3 = BatchNormalization()(conv3)
#     pool3 = MaxPooling2D(pool_size=(1, 2))(BN3)
#
#     conv4 = Conv2D(256, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
#     BN4 = BatchNormalization()(conv4)
#     conv4 = Conv2D(256, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN4)
#     BN4 = BatchNormalization()(conv4)
#     drop4 = Dropout(0.5)(BN4)
#     # drop4 为 9*25
#     # pool4 = MaxPooling2D(pool_size=(1, 2))(drop4) # 这里pool之后，是12.5，也就是12了，之后上采样变成24就会对不齐
#     # 可以省略这个pool，并相应省略之后的上采样部分
#     conv5 = Conv2D(512, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(drop4)
#     BN5 = BatchNormalization()(conv5)
#     conv5 = Conv2D(512, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN5)
#     BN5 = BatchNormalization()(conv5)
#     drop5 = Dropout(0.5)(BN5)
#     print('drop5.shape: ', drop5.shape)
#
#
#     up6 = Conv2D(256, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(drop5)  # UpSampling2D(size = (1,2))(drop5))
#     BN6 = BatchNormalization()(up6)
#     #merge6 = concatenate([drop4, BN6], axis=3)
#     conv6 = Conv2D(256, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN6)
#     BN6 = BatchNormalization()(conv6)
#     conv6 = Conv2D(256, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN6)
#     BN6 = BatchNormalization()(conv6)
#
#     up7 = Conv2D(128, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(1, 2))(BN6))
#     BN7 = BatchNormalization()(up7)
#     #merge7 = concatenate([conv3, BN7], axis=3)
#     conv7 = Conv2D(128, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN7)
#     BN7 = BatchNormalization()(conv7)
#     conv7 = Conv2D(128, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN7)
#     BN7 = BatchNormalization()(conv7)
#
#     up8 = Conv2D(64, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(1, 2))(BN7))
#     BN8 = BatchNormalization()(up8)
#     #merge8 = concatenate([conv2, BN8], axis=3)
#     conv8 = Conv2D(64, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN8)
#     BN8 = BatchNormalization()(conv8)
#     conv8 = Conv2D(64, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN8)
#     BN8 = BatchNormalization()(conv8)
#
#     up9 = Conv2D(32, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(1, 2))(BN8))
#     BN9 = BatchNormalization()(up9)
#     #merge9 = concatenate([conv1, BN9], axis=3)
#     conv9 = Conv2D(32, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN9)
#     BN9 = BatchNormalization()(conv9)
#     conv9 = Conv2D(32, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN9)
#     BN9 = BatchNormalization()(conv9)
#     conv9 = Conv2D(2, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN9)
#     BN9 = BatchNormalization()(conv9)
#
#     conv10 = Conv2D(1, 1, activation='sigmoid')(BN9) # 或者用tanh
#
#     model = Model(input=inputs, output=conv10)
#     model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=['mse'])
#
#     # model.summary()
#
#     if (pretrained_weights):
#         model.load_weights(pretrained_weights)
#
#     print('In function: Unet (the way to create Unet)')
#     model.summary()
#
#     return model



'''----------'''

# def unet(pretrained_weights=None, input_size=(9, 200, 1)):
#     inputs = Input(input_size)
#     BN = BatchNormalization()(inputs)
#     conv1 = Conv2D(32, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(BN)
#     conv1 = Conv2D(32, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
#     pool1 = MaxPooling2D(pool_size=(1, 2))(conv1)
#     conv2 = Conv2D(64, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
#     conv2 = Conv2D(64, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
#     pool2 = MaxPooling2D(pool_size=(1, 2))(conv2)
#     conv3 = Conv2D(128, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
#     conv3 = Conv2D(128, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
#     pool3 = MaxPooling2D(pool_size=(1, 2))(conv3)
#     conv4 = Conv2D(256, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
#     conv4 = Conv2D(256, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
#     drop4 = Dropout(0.5)(conv4)
#     # drop4 为 9*25
#     # pool4 = MaxPooling2D(pool_size=(1, 2))(drop4) # 这里pool之后，是12.5，也就是12了，之后上采样变成24就会对不齐
#     # 可以省略这个pool，并相应省略之后的上采样部分
#     conv5 = Conv2D(512, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(drop4)
#     conv5 = Conv2D(512, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
#     drop5 = Dropout(0.5)(conv5)
#     print('drop5.shape: ', drop5.shape)
#     up6 = Conv2D(256, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
#         drop5)  # UpSampling2D(size = (1,2))(drop5))
#     merge6 = concatenate([drop4, up6], axis=3)
#     conv6 = Conv2D(256, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
#     conv6 = Conv2D(256, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
#
#     up7 = Conv2D(128, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(1, 2))(conv6))
#     merge7 = concatenate([conv3, up7], axis=3)
#     conv7 = Conv2D(128, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
#     conv7 = Conv2D(128, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
#
#     up8 = Conv2D(64, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(1, 2))(conv7))
#     merge8 = concatenate([conv2, up8], axis=3)
#     conv8 = Conv2D(64, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
#     conv8 = Conv2D(64, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
#
#     up9 = Conv2D(32, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(1, 2))(conv8))
#     merge9 = concatenate([conv1, up9], axis=3)
#     conv9 = Conv2D(32, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
#     conv9 = Conv2D(32, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
#     conv9 = Conv2D(2, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
#     conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
#
#     model = Model(input=inputs, output=conv10)
#
#     model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=['mse'])
#
#     # model.summary()
#
#     if (pretrained_weights):
#         model.load_weights(pretrained_weights)
#
#     print('In function: Unet (the way to create Unet)')
#     model.summary()
#
#     return model


'''----------'''

# def unet(pretrained_weights=None, input_size=(9, 200, 1)):
#     inputs = Input(input_size)
#     BN = BatchNormalization()(inputs)
#     conv1 = Conv2D(32, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(BN)
#     conv1 = Conv2D(32, (3, 10), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
#     pool1 = MaxPooling2D(pool_size=(1, 2))(conv1)
#     conv2 = Conv2D(64, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
#     conv2 = Conv2D(64, (3, 10), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
#     pool2 = MaxPooling2D(pool_size=(1, 2))(conv2)
#     conv3 = Conv2D(128, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
#     conv3 = Conv2D(128, (3, 10), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
#     pool3 = MaxPooling2D(pool_size=(1, 2))(conv3)
#     conv4 = Conv2D(256, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
#     conv4 = Conv2D(256, (3, 10), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
#     drop4 = Dropout(0.5)(conv4)
#     # drop4 为 9*25
#     # pool4 = MaxPooling2D(pool_size=(1, 2))(drop4) # 这里pool之后，是12.5，也就是12了，之后上采样变成24就会对不齐
#     # 可以省略这个pool，并相应省略之后的上采样部分
#     conv5 = Conv2D(512, (3, 10), activation='relu', padding='same', kernel_initializer='he_normal')(drop4)
#     conv5 = Conv2D(512, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
#     drop5 = Dropout(0.5)(conv5)
#     print('drop5.shape: ', drop5.shape)
#     up6 = Conv2D(256, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
#         drop5)  # UpSampling2D(size = (1,2))(drop5))
#     merge6 = concatenate([drop4, up6], axis=3)
#     conv6 = Conv2D(256, (3, 10), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
#     conv6 = Conv2D(256, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
#
#     up7 = Conv2D(128, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(1, 2))(conv6))
#     merge7 = concatenate([conv3, up7], axis=3)
#     conv7 = Conv2D(128, (3, 10), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
#     conv7 = Conv2D(128, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
#
#     up8 = Conv2D(64, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(1, 2))(conv7))
#     merge8 = concatenate([conv2, up8], axis=3)
#     conv8 = Conv2D(64, (3, 10), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
#     conv8 = Conv2D(64, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
#
#     up9 = Conv2D(32, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(1, 2))(conv8))
#     merge9 = concatenate([conv1, up9], axis=3)
#     conv9 = Conv2D(32, (3, 10), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
#     conv9 = Conv2D(32, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
#     conv9 = Conv2D(2, (1, 10), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
#     conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
#
#     model = Model(input=inputs, output=conv10)
#
#     model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=['mse'])
#
#     # model.summary()
#
#     if (pretrained_weights):
#         model.load_weights(pretrained_weights)
#
#     print('In function: Unet (the way to create Unet)')
#     model.summary()
#
#     return model
