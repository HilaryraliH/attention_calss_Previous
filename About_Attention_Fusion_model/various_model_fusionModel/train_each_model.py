from sklearn.metrics import confusion_matrix
from mod_stru import JNE_CNN, EEGNet, DeepConvNet, ShallowConvNet,Proposed_Conv, Proposed_Conv_R
from test_model_stru import generate_lstmfcn_Conv2D,generate_lstmfcn_Conv2D_nolstm,Proposed_Deeper\
    ,DeepConvNet_no_spatial,Double_path,generate_lstmfcn, EEGNet_original
from keras.callbacks import LearningRateScheduler
from keras.layers import Input
from keras import callbacks
import numpy as np

samples = 200

def erect_model(DeepNet_cfg,ProR_cfg):
    DeepNet_model_input = None
    Chans = DeepNet_cfg.chans
    if DeepNet_cfg.data_format == '2D':
        DeepNet_model_input = Input(shape=(Chans, samples, 1))
    elif DeepNet_cfg.data_format == '3D':
        DeepNet_model_input = Input(shape=(1, samples, Chans))
    elif DeepNet_cfg.data_format =='true_2D':
        DeepNet_model_input = Input(shape=(Chans,samples))

    ProR_model_input = None
    Chans = ProR_cfg.chans
    if ProR_cfg.data_format == '2D':
        ProR_model_input = Input(shape=(Chans, samples, 1))
    elif ProR_cfg.data_format == '3D':
        ProR_model_input = Input(shape=(1, samples, Chans))
    elif ProR_cfg.data_format == 'true_2D':
        ProR_model_input = Input(shape=(Chans, samples))


    model = eval(cfg.model_name)([DeepNet_model_input,ProR_model_input],ProR_cfg, nb_classes=2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model





def fit_model(model, cfg, X_train, Y_train, X_test, Y_test, save_model_dir,validation_split):
    hist=None

    def scheduler(epoch):
        lr=None
        if epoch >20:
            lr=0.0005
        elif epoch >10:
            lr=0.001
        else:
            lr=0.005
        return lr

    change_lr = LearningRateScheduler(scheduler,verbose=1)

    if validation_split:
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', verbose=1, patience=5,mode='auto',epsilon=0.0001)
        hist = model.fit(X_train, Y_train,validation_split=validation_split,
                         batch_size=cfg.batch_size, epochs=cfg.epochs, verbose=2,
                         callbacks=[reduce_lr])
    else:
        hist = model.fit(X_train, Y_train, callbacks=[change_lr],
                         batch_size=cfg.batch_size, epochs=cfg.epochs, verbose=2, shuffle=True,
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
