# coding=utf-8
import numpy as np
import keras.backend as K
from keras import layers

from keras import backend as K
from keras.layers import Conv1D, GlobalAveragePooling1D, Permute
from keras.layers import Input, Conv2D, MaxPooling2D, Dense,Layer
from keras.layers import Flatten, Dropout, Reshape, LSTM, Average
from keras.layers import BatchNormalization, SeparableConv2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import DepthwiseConv2D, Activation, concatenate, Lambda,Permute
from keras.models import Model
from keras import optimizers
from keras.constraints import max_norm
from data_pre import log, square
import os

Samples = 200 

'''输入为3D（1，200，chans）的模型'''
#%% 输入为3D（1，200，chans）的模型


def JNE_CNN(model_input,cfg, nb_classes=2):
    # article: Inter-subject transfer learning with an end-to-end deep convolutional neural network for EEG-based BCI
    # # remain unchanged
    data = Conv2D(60, (1, 4), strides=(1, 2), activation='relu')(model_input)
    data = MaxPooling2D(pool_size=(1, 2))(data)
    data = Conv2D(40, (1, 3), activation='relu')(data)
    data = Conv2D(20, (1, 2), activation='relu')(data)
    data = Flatten()(data)
    data = Dropout(0.2)(data)
    data = Dense(100, activation='relu')(data)
    data = Dropout(0.3)(data)
    data = Dense(nb_classes, activation='softmax')(data)
    model = Model(model_input, data)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def Proposed_Conv(model_input,cfg, nb_classes):
    dropoutRate = 0.5
    norm_rate = 0.25
    Chans = cfg.chans

    block0 = Conv2D(8, (1, 5), padding='same', use_bias=False)(model_input)
    block0 = BatchNormalization()(block0)

    block1 = DepthwiseConv2D((1, Chans), use_bias=False, depth_multiplier=2, depthwise_constraint=max_norm(1.))(block0)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = SeparableConv2D(16, (1, 5), use_bias=False, padding='same')(block1)  # it's（1，16）before
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 4))(block2)  # it's（1，8）before
    block2 = Dropout(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)
    dense = Dense(nb_classes, name='dense', kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=model_input, outputs=softmax)


def Proposed_Conv_R(model_input,cfg, nb_classes):
    dropoutRate = 0.5
    norm_rate = 0.25
    Chans = cfg.chans
    '''input1   = Input(shape = (1, Chans, Samples))'''

    block1 = Conv2D(8, (1, 5), padding='same', use_bias=False,name='conv1__R')(model_input)
    block1 = BatchNormalization(axis=-1,name='BN1__R')(block1)

    block1 = DepthwiseConv2D((1, 20), use_bias=False, depth_multiplier=2, depthwise_constraint=max_norm(1.),name='conv2__R')(block1)
    block1 = BatchNormalization(axis=-1,name='BN2__R')(block1)  # but when I use axis=1 before, it worked
    block1 = Activation('elu',name='Activation2__R')(block1)
    block1 = AveragePooling2D((1, 4),name='mean2__R')(block1)
    block1 = Dropout(dropoutRate,name='drop2__R')(block1)

    block2 = SeparableConv2D(16, (1, 16), use_bias=False, padding='same',name='conv3__R')(block1)
    block2 = BatchNormalization(axis=-1,name='BN3__R')(block2)
    block2 = Activation('elu',name='Activation3__R')(block2)
    print(block2.shape)

    block3 = Reshape((int(block2.shape[-2]), int(block2.shape[-1])),name='reshape__R')(block2)
    l_lstm_sent = LSTM(32, return_sequences=True,name='lstm1__R')(block3)
    l_lstm_sent = LSTM(8, return_sequences=True,name='lstm2__R')(l_lstm_sent)

    flatten = Flatten(name='flatten__R')(l_lstm_sent)
    preds = Dense(nb_classes,name='dense__R', activation='softmax', kernel_constraint=max_norm(norm_rate))(flatten)

    return Model(inputs=model_input, outputs=preds)


def Transfer_Proposed_Conv_R(model_input,cfg, nb_classes):
    dropoutRate = 0.5
    norm_rate = 0.25
    Chans = cfg.chans
    '''input1   = Input(shape = (1, Chans, Samples))'''
    block1 = Conv2D(8, (9, 1), use_bias=False, name='conv1__R')(model_input) # spatial
    print('block1.shape',block1.shape)
    # block1 = Conv2D(8, (1, 5), padding='same', use_bias=False,name='conv1__R')(model_input)
    block1 = BatchNormalization(axis=-1,name='BN1__R')(block1)

    block1 = DepthwiseConv2D((1, 20), use_bias=False, depth_multiplier=2, depthwise_constraint=max_norm(1.),name='conv2__R')(block1)
    print('block1.shape', block1.shape)
    block1 = BatchNormalization(axis=-1,name='BN2__R')(block1)  # but when I use axis=1 before, it worked
    block1 = Activation('elu',name='Activation2__R')(block1)
    block1 = AveragePooling2D((1, 4),name='mean2__R')(block1)
    block1 = Dropout(dropoutRate,name='drop2__R')(block1)

    block2 = SeparableConv2D(16, (1, 16), use_bias=False, padding='same',name='conv3__R')(block1)
    block2 = BatchNormalization(axis=-1,name='BN3__R')(block2)
    block2 = Activation('elu',name='Activation3__R')(block2)

    # if use LSTM after Dropout, it will be confused by the order
    # but the AveragePooling2D may be worked, then I will check
    ''' block2 = AveragePooling2D((1, 4))(block2)# it's（1，8）before
    block2 = Dropout(dropoutRate)(block2)'''
    print(block2.shape) # 12，45，16
    # block3 = Reshape((48, 16))(block2)
    block3 = Reshape((int(block2.shape[-2]), int(block2.shape[-1])),name='reshape__R')(block2)

    l_lstm_sent = LSTM(32, return_sequences=True,name='lstm1__R')(block3)
    l_lstm_sent = LSTM(8, return_sequences=True,name='lstm2__R')(l_lstm_sent)

    flatten = Flatten(name='flatten__R')(l_lstm_sent)
    preds = Dense(nb_classes,name='dense__R', activation='softmax', kernel_constraint=max_norm(norm_rate))(flatten)
    # preds = Dense(nb_classes, name='dense', activation='softmax')(flatten)

    return Model(inputs=model_input, outputs=preds)


#%% 输入为2D（chans，200，1）
def EEGNet(model_input,cfg, nb_classes,dropoutRate=0.5, kernLength=64, F1=8,D=2, F2=16, norm_rate=0.25):
    # article: EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces
    # changed as the comments
    Chans = cfg.chans
    block1 = Conv2D(F1, (1, kernLength), padding='same', input_shape=(Chans, Samples, 1), use_bias=False)(model_input)
    block1 = BatchNormalization()(block1)  # I'm not sure the axis, axis=1 before

    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D, depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(block1)  # it's（1，16）before
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(dropoutRate)(block2)

    flatten = Flatten()(block2)
    dense = Dense(nb_classes, kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=model_input, outputs=softmax)

def DeepConvNet(model_input,cfg, nb_classes,dropoutRate=0.5):
    Chans = cfg.chans
    block1 = Conv2D(25, (1, 5), input_shape=(Chans, Samples, 1), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(
        model_input)  # 因为此处数据较小，所以filter降成了5

    block1 = Conv2D(25, (Chans, 1), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block1)  # it's axis=1 before
    block1 = Activation('elu')(block1)

    block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = Conv2D(50, (1, 5), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)

    block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(100, (1, 5), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block3 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

    block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(200, (1, 5), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block4 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten()(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=model_input, outputs=softmax)


def ShallowConvNet(model_input,cfg, nb_classes, dropoutRate=0.5):
    # article: EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces
    # changed as the comments
    Chans = cfg.chans
    block1 = Conv2D(40, (1, 25),
                    input_shape=(Chans, Samples, 1),
                    kernel_constraint=max_norm(2.,axis=(0, 1, 2)))(model_input)
    block1 = Conv2D(40, (Chans, 1), use_bias=False,
                    kernel_constraint=max_norm(2.,axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 75), strides=(1, 15))(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(dropoutRate)(block1)
    flatten = Flatten()(block1)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=model_input, outputs=softmax)


def Spatial_model(model_input,cfg, nb_classes):
    dropoutRate = 0.5
    norm_rate = 0.25
    Chans = cfg.chans
    '''input1   = Input(shape = (1, Chans, Samples))'''

    block1 = Conv2D(8, (1, 5), padding='same', use_bias=False,name='conv1__R')(model_input)
    block1 = BatchNormalization(axis=-1,name='BN1__R')(block1)

    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=2, depthwise_constraint=max_norm(1.),name='conv2__R')(block1)
    block1 = BatchNormalization(axis=-1,name='BN2__R')(block1)  # but when I use axis=1 before, it worked
    block1 = Activation('elu',name='Activation2__R')(block1)
    block1 = AveragePooling2D((1, 4),name='mean2__R')(block1)
    block1 = Dropout(dropoutRate,name='drop2__R')(block1)

    block2 = SeparableConv2D(16, (1, 5), use_bias=False, padding='same',name='conv3__R')(block1)
    block2 = BatchNormalization(axis=-1,name='BN3__R')(block2)
    block2 = Activation('elu',name='Activation3__R')(block2)

    block3 = SeparableConv2D(16, (1, 5), use_bias=False, padding='same', name='conv3__R')(block2)
    block3 = BatchNormalization(axis=-1, name='BN3__R')(block3)
    block3 = Activation('elu', name='Activation3__R')(block3)

    print('block3.shape',block3.shape)
    block3 = Reshape((int(block2.shape[-2]), int(block2.shape[-1])),name='reshape__R')(block2)

    l_lstm_sent = LSTM(32, return_sequences=True,name='lstm1__R')(block3)
    l_lstm_sent = LSTM(8, return_sequences=True,name='lstm2__R')(l_lstm_sent)

    flatten = Flatten(name='flatten__R')(l_lstm_sent)
    preds = Dense(nb_classes,name='dense__R', activation='softmax', kernel_constraint=max_norm(norm_rate))(flatten)
    # preds = Dense(nb_classes, name='dense', activation='softmax')(flatten)

    return Model(inputs=model_input, outputs=preds)


def Transpose_Net(model_input,cfg, nb_classes,dropoutRate=0.5):
    Chans = cfg.chans
    block1 = Conv2D(20, (1, 5), input_shape=(Chans, Samples, 1), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(
        model_input)

    block1 = Conv2D(5, (Chans, 1), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block1)  # it's axis=1 before
    block1 = Activation('elu')(block1)
    print('block1.shape:',block1.shape)

    block1 = Reshape((int(block1.shape[-2]), int(block1.shape[-1]),1), name='reshape__R')(block1)
    block1 = Permute((2,1,3))(block1)

    block2 = Conv2D(40, (1, 5), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)

    block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(60, (1, 5), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block3 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

    block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(80, (1, 5), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block4 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    #block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten()(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=model_input, outputs=softmax)

