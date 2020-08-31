from sklearn.metrics import confusion_matrix
from mod_stru import *
from keras.layers import *
from keras.models import Model
from keras.callbacks import LearningRateScheduler
from keras.constraints import max_norm
from keras import callbacks
import numpy as np
# from GlobalAttention import AttLayer, MySelfAttention

samples = 200
def check_model_input(data_format, chan):
    model_input = None
    if data_format == '2D':
        model_input = Input(shape=(chan, samples, 1))
    elif data_format == '3D':
        model_input = Input(shape=(1, samples, chan))
    elif data_format == 'true_2D':
        model_input = Input(shape=(chan, samples))
    return model_input

def erect_single_model(tmp_input,model_name,chans, dataformat):
    model = eval(model_name)(tmp_input, chans)
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



# 如果两个parallel融合
# def erect_model(model_name, Chans,data_format):
#     chan1 = Chans[0]
#     chan2 = Chans[1]

#     inputF = check_model_input(data_format, chan1)
#     inputP = check_model_input(data_format, chan2)

#     model_input = [inputF, inputP]
#     modelF = eval(model_name)(inputF, chan1)
#     modelP = eval(model_name)(inputP, chan2)
#     '''如果在flatten层融合'''
#     if model_name=='Proposed_Conv':
#         my_concatenate = Concatenate()([modelF.layers[-3].output, modelP.layers[-3].output])
#         pre = Dense(2,activation='softmax')(my_concatenate)
#         model = Model(model_input, pre)
#         model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#         return model
#     elif model_name=='Proposed_Conv_R' :
#         my_concatenate = Concatenate()([modelF.layers[-3].output, modelP.layers[-3].output])
#         #pre = Dense(100, activation='softmax')(my_concatenate)
#         pre = Dense(2, activation='softmax')(my_concatenate)
#         model = Model(model_input, pre)
#         print(type(model))
#         model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#         return model
#     elif model_name=='Transpose_Net':
#         my_concatenate = Concatenate()([modelF.layers[-3].output, modelP.layers[-3].output])
#         #pre = Dense(100, activation='softmax')(my_concatenate)
#         my_concatenate = Dense(100,activation='elu')(my_concatenate)
#         pre = Dense(2, activation='softmax')(my_concatenate)
#         model = Model(model_input, pre)
#         print(type(model))
#         model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#         return model
#     elif model_name=='EEGNet' or 'DeepConvNet':
#         print(modelF.layers[-3].output.shape,'   ',modelP.layers[-3].output.shape )
#         my_concatenate = Concatenate()([modelF.layers[-3].output, modelP.layers[-3].output])
        

#         # 加入注意力机制
#         # 这里本来就是拉长后，再进行注意力机制
#         from GlobalAttention import AttentionLayer
#         keras.layers.Lambda(function, output_shape=None, mask=None, arguments=None)
#         def expand(x):
#             x = K.expand_dims(x, axis=-1)
#             return x
#         expand_layer = Lambda(expand)
#         print('my_concatenate.shape  after concatenate',my_concatenate.shape)
#         my_concatenate = AttentionLayer(my_concatenate)
#         print('my_concatenate.shape after reshape', my_concatenate.shape)
#         my_concatenate = MySelfAttention(1,name='attention_layer')(my_concatenate)
#         print('my_concatenate.shape after attenlayer(64)', my_concatenate.shape)
#         my_concatenate = Flatten()(my_concatenate)
        


#         my_concatenate = Dense(100,activation='elu')(my_concatenate)
#         pre = Dense(2, activation='softmax')(my_concatenate)
#         model = Model(model_input, pre)
#         print(type(model))
#         model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#         return model



#     '''如果在之前的二维层channel融合:只有Pro_R考虑'''
#     '''if model_name=='Proposed_Conv_R':
#         out1 = BatchNormalization()(modelF.layers[-6].output)
#         out2 = BatchNormalization()(modelP.layers[-6].output)
#         my_concatenate = Concatenate()([out1, out2])

#         l_lstm_sent = LSTM(32, return_sequences=True)(my_concatenate)
#         l_lstm_sent = LSTM(8, return_sequences=True)(l_lstm_sent)

#         flatten = Flatten()(l_lstm_sent)
#         dense = Dense(2, kernel_constraint=max_norm(1.0))(flatten)
#         preds = Activation('softmax')(dense)

#         model = Model(model_input, preds)
#         model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#         return model'''

def fit_model(model, cfg, X_train, Y_train, X_test, Y_test, save_model_dir,validation_split):
    hist=None
    def scheduler(epoch):
        lr=None
        if epoch >20:
            lr=0.0005
        elif epoch >10:
            lr=0.001
        else:
            lr=0.01
        return lr

    change_lr = LearningRateScheduler(scheduler,verbose=1)

    if validation_split:
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', verbose=1, patience=5,mode='auto',epsilon=0.0001)
        hist = model.fit(X_train, Y_train,validation_split=validation_split,
                         batch_size=cfg.batch_size, epochs=cfg.epochs, verbose=2,
                         callbacks=[reduce_lr])
    else:
        hist = model.fit(X_train, Y_train,
                         batch_size=cfg.batch_size, epochs=cfg.epochs, verbose=2,callbacks=[change_lr],
                         validation_data=(X_test, Y_test))
    model.save(save_model_dir)
    return hist


def evaluate_model(model, X_test, Y_test):
    probs = model.predict(X_test)
    preds = probs.argmax(axis=-1)
    true_label = Y_test.argmax(axis=-1)
    acc = np.mean(preds == true_label)
    confu_mat = confusion_matrix(true_label, preds, labels=[0, 1])
    return acc, confu_mat


# 如果三个parallel融合

def erect_model(model_name, Chans,data_format):
    chan1 = Chans[0]
    chan2 = Chans[1]
    chan3 = Chans[2]
    inputF = check_model_input(data_format, chan1)
    inputC = check_model_input(data_format, chan2)
    inputP = check_model_input(data_format, chan3)
    model_input = [inputF, inputC, inputP]
    modelF = eval(model_name)(inputF, chan1)
    modelC = eval(model_name)(inputC, chan2)
    modelP = eval(model_name)(inputP, chan3)
    if model_name=='Proposed_Conv':
        my_concatenate = Concatenate()([modelF.layers[-3].output, modelC.layers[-3].output, modelP.layers[-3].output])
        pre = Dense(2,activation='softmax')(my_concatenate)
        model = Model(model_input, pre)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    elif model_name=='Proposed_Conv_R':
        my_concatenate = Concatenate()([modelF.layers[-3].output, modelC.layers[-3].output, modelP.layers[-3].output])
        pre = Dense(100, activation='softmax')(my_concatenate)
        pre = Dense(2, activation='softmax')(pre)
        model = Model(model_input, pre)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif model_name=='EEGNet' or 'DeepConvNet':
        # 加一个var层，再连接
        my_concatenate = Concatenate()([modelF.layers[-3].output, modelC.layers[-3].output,modelP.layers[-3].output])
        #my_concatenate = Dense(100,activation='elu')(my_concatenate)

        # # 加入注意力机制
        # # 这里本来就是拉长后，再进行注意力机制
        # print('my_concatenate.shape  after concatenate',my_concatenate.shape)
        # my_concatenate = Reshape((my_concatenate.shape[-1],1))(my_concatenate)
        # print('my_concatenate.shape after reshape', my_concatenate.shape)
        # my_concatenate = AttLayer(64)(my_concatenate)
        # print('my_concatenate.shape after attenlayer(64)', my_concatenate.shape)
        # # 最后拉长
        # my_concatenate = Flatten()(my_concatenate)

        pre = Dense(2, activation='softmax')(my_concatenate)
        model = Model(model_input, pre)
        print(type(model))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model