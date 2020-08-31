# coding=utf-8
# all the medel name
# [ 'JNE_CNN', 'EEGNet', 'DeepConvNet', 'ShallowConvNet','Proposed_Conv',
#   'Proposed_Conv_R','Proposed_Deeper','Double_path','DeepConvNet_no_spatial',
#   'generate_lstmfcn','generate_lstmfcn_Conv2D','generate_lstmfcn_Conv2D_nolstm',
#   'EEGNet_original']
# coding=utf-8
import numpy as np
from keras import layers

from keras import backend as K
from keras.models import Model
from mod_stru import AttLayer
from keras.layers import *
from keras import optimizers
from keras.constraints import max_norm
from data_pre import log, square
import os
samples = 200
###########################################################################################
CUDA_VISIBLE_DEVICES=0
total_sub_num = 8
debug = 0 # 是否在debug，若是，则所有参数都会变小来调试
is_CPU = 0

# F_C_9  之前是改变了Pro的name，但没有改变pro——R和DeepNet的name，需要重新运行
# 9  都没有改变
model_name = ['DeepConvNet','Proposed_Conv_R']  # 随着模型改变
select_chan_way = '9'
validation_split = 0  # 默认0，可以适当较小比例地分离validation
Desktop_or_GPU = 'GPU'  # 'GPU DeskTop'

def mk_save_dir(model_name,debug,validation_split,Desktop_or_GPU,once):
    save_dir = None
    root_dir = None
    if debug:
        save_dir = 'results\\' + model_name + '--' + 'debug' + '\\' + '第{}次'.format(once) + '\\'
        root_dir = 'results\\' + model_name + '--' + 'debug' + '\\'
    else:
        if validation_split:
            save_dir = 'results\\' + model_name + '--'  + '\\' + '第{}次'.format(once) + '\\'
            root_dir = 'results\\' + model_name + '--'  + '\\'
        else:
            save_dir = 'results\\' + model_name + '--'  + '_no_vali' + '\\' + '第{}次'.format(once) + '\\'
            root_dir = 'results\\' + model_name + '--' +  '_no_vali' + '\\'
    return root_dir,save_dir

###########################################################################################
# when I add a model
# I should check: config.py (2D,3D,epochs,batch_size,or other special argu)
#                 train_each_model.py(import the new model)
#                 model_name in main.py

############################################################################################

import os
import numpy as np
from config import configur
from time import time
from data_pre import load_sub
from save_information import check_path, save_training_pic, save_acc_pic, save_csv
from save_information import my_append_col,my_append_row, my_concatenate, save_txt,save_total_csv
from train_each_model import erect_model, fit_model, evaluate_model
from keras.models import load_model

DeepNet_cfg = configur(model_name[0], is_CPU, debug,select_chan_way)
ProR_cfg = configur(model_name[1], is_CPU, debug,select_chan_way)
DeepNet_cfg.set_config()
ProR_cfg.set_config()

results_dir = '9_results'
# 'F_C_9_results'

root_dir = None
total_confu_matrix = None
for once in range(DeepNet_cfg.total_times):
    start = time()
    root_dir, save_dir = mk_save_dir(model_name[0], debug, validation_split, Desktop_or_GPU,once)
    DeepNet_load_root_dir = results_dir+'\\'+model_name[0]+'\\'+'第{}次'.format(once)+'\\'
    ProR_load_root_dir = results_dir + '\\' + model_name[1] + '\\'+'第{}次'.format(once)+'\\'
    check_path(save_dir)
    acc_list = []
    confu_matrix = None  # use to save_to_csv
    confu_matri = None  # use to calculate aver_confu_mat
    numParams = None
    save_model_dir = None
    info_dict = {}

    for sub in range(total_sub_num):
        DeepNet_load_dir = DeepNet_load_root_dir+model_name[0]+'_{}'.format(sub)+'.h5'
        ProR_load_dir = ProR_load_root_dir+model_name[1]+'_{}'.format(sub)+'.h5'
        save_model_dir = save_dir + DeepNet_cfg.model_name + '_' + str(sub) + '.h5'
        confu_mat = None
        acc = None
        model = None
        print('loading sub {} data'.format(sub))

        (X_train1, Y_train), (X_test1, Y_test) = load_sub(total_sub_num, sub, DeepNet_cfg.data_format, DeepNet_cfg.select_chan_way) # load data
        (X_train2, Y_train), (X_test2, Y_test) = load_sub(total_sub_num, sub, ProR_cfg.data_format, ProR_cfg.select_chan_way)

        X_train = [X_train1,X_train2]
        X_test = [X_test1,X_test2]


        #print('{},{}     {},{}'.format(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape))
        '''erect model'''
        '''
        if not os.path.exists(save_model_dir):  # if not trained
            model = erect_model(DeepNet_cfg,ProR_cfg)
            if sub == 0:
                model.summary()
            print('------------------------- start training {}th fold ------------------------------------'.format(sub))
            hist = fit_model(model, cfg, X_train, Y_train, X_test, Y_test, save_model_dir, validation_split)
            save_training_pic(sub, hist, save_dir)  # save the training trend
            acc, confu_mat = evaluate_model(model, X_test, Y_test)  # evaluate model

        else:  # if trained
        '''
        model1 = load_model(DeepNet_load_dir)
        model2 = load_model(ProR_load_dir)

        '''建立两个input形状'''
        '''
        DeepNet_model_input = None
        Chans = DeepNet_cfg.chans
        if DeepNet_cfg.data_format == '2D':
            DeepNet_model_input = Input(shape=(Chans, samples, 1))
        elif DeepNet_cfg.data_format == '3D':
            DeepNet_model_input = Input(shape=(1, samples, Chans))
        elif DeepNet_cfg.data_format == 'true_2D':
            DeepNet_model_input = Input(shape=(Chans, samples))

        ProR_model_input = None
        Chans = ProR_cfg.chans
        if ProR_cfg.data_format == '2D':
            ProR_model_input = Input(shape=(Chans, samples, 1))
        elif ProR_cfg.data_format == '3D':
            ProR_model_input = Input(shape=(1, samples, Chans))
        elif ProR_cfg.data_format == 'true_2D':
            ProR_model_input = Input(shape=(Chans, samples))
        '''
        #model_input = [DeepNet_model_input,ProR_model_input]
        model_input = [model1.input,model2.input]
        #model2.input_names=['input_2']
        #model2._feed_input_names = ['input_2']
        #提取前一部分（flatten之前的）作为model

        #model11 = Model(model1.input, model1.layers[-3].output)
        #model22 = Model(model2.input, model2.layers[-2].output)

        #DeepNet_flatten = model1.layers[-3].output
        #Pro_R_flatten = model2.layers[-2].output
        #my_concatenate = Concatenate()([DeepNet_flatten, Pro_R_flatten])
        my_concatenate = Concatenate()([model1.layers[-3].output, model2.layers[-2].output])

        #加入注意力机制
        # 这里本来就是拉长后，再进行注意力机制
        # print('my_concatenate.shape  after concatenate',my_concatenate.shape)
        # my_concatenate = Reshape((my_concatenate.shape[-1],1))(my_concatenate)
        # print('my_concatenate.shape after reshape', my_concatenate.shape)
        # my_concatenate = AttLayer(64)(my_concatenate)
        # print('my_concatenate.shape after attenlayer(64)', my_concatenate.shape)
        # #最后拉长
        # my_concatenate = Flatten()(my_concatenate)



        print('my_concatenate.shape after flatten', my_concatenate.shape)
        pre = Dense(2,activation='softmax')(my_concatenate)
        model = Model(model_input,pre)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        print('------------------------- start training {}th fold ------------------------------------'.format(sub))


        if not os.path.exists(save_model_dir):
            hist = fit_model(model, DeepNet_cfg, X_train, Y_train, X_test, Y_test, save_model_dir, validation_split)
            save_training_pic(sub, hist, save_dir)  # save the training trend
            acc, confu_mat = evaluate_model(model, X_test, Y_test)  # evaluate model


        numParams = model.count_params()
        info_dict['numParams'] = numParams
        confu_matrix = my_append_row(confu_matrix, confu_mat)  # save confu_matrix to save_to_csv
        #confu_mat = np.expand_dims(confu_mat, axis=0)  # save confu_matri to cal aver_confu_mat
        #confu_matri = my_concatenate(confu_matri, confu_mat)
        acc_list = np.append(acc_list, acc)  # save each sub's acc
        print("Classification accuracy: {}".format(acc))

    # save to file
    end = time()
    training_time = (end - start) / 3600
    total_confu_matrix = my_append_col(total_confu_matrix,confu_matrix)
    info_dict['training_time'] = str(training_time) + ' hours'
    save_acc_pic(acc_list, save_dir)
    save_csv(confu_matrix, save_dir)
    save_txt(info_dict, DeepNet_cfg, save_dir)
save_total_csv(total_confu_matrix,root_dir)

