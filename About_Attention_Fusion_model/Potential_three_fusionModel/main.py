# coding=utf-8

'''若要固定模型的初始参数，则使用： keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None))'''

from config import *
import os
import numpy as np
from time import time
from data_pre import load_sub, mk_save_dir
from save_information import *
from train_each_model import erect_model, fit_model, evaluate_model
from keras.models import load_model

cfg = configur(model_name, is_CPU, debug, select_chan_way)
cfg.set_config()
root_dir = None
total_confu_matrix = None
for once in range(cfg.total_times):
    start = time()
    root_dir, save_dir = mk_save_dir(
        model_name, debug, validation_split, Desktop_or_GPU, once)
    check_path(save_dir)
    acc_list = []
    confu_matrix = None  # use to save_to_csv
    confu_matri = None  # use to calculate aver_confu_mat
    numParams = None
    save_model_dir = None
    info_dict = {}

    for sub in range(total_sub_num):
        save_model_dir = save_dir + cfg.model_name + '_' + str(sub) + '.h5'
        confu_mat = None
        acc = None
        model = None

        model = erect_model(cfg.model_name, cfg.chans, cfg.data_format)
        print(type(model))

        print('loading sub {} data'.format(sub))
        (X_train, Y_train), (X_test, Y_test) = ([], []), ([], [])
        for way in cfg.select_chan_way:
            (x_train, Y_train), (x_test, Y_test) = load_sub(
                total_sub_num, sub, cfg.data_format, way)  # load data
            X_train.append(x_train)
            X_test.append(x_test)
        print('Successful loading')

        if not os.path.exists(save_model_dir):  # if not trained
            model = erect_model(cfg.model_name, cfg.chans, cfg.data_format)
            if sub == 0:
                model.summary()

                # plot model
                # os.environ["PATH"] += os.pathsep + 'C:/C1_Install_package/Graphviz/Graphviz 2.44.1/bin'
                # plot_model(model, cfg.model_name+'.png',show_shapes=True)


            print('------------------------- start training \033[1;32;m {} \033[0mth fold ------------------------------------'.format(sub))
            hist = fit_model(model, cfg, X_train, Y_train,
                             X_test, Y_test, save_model_dir, validation_split)
            save_training_pic(sub, hist, save_dir)  # save the training trend
            acc, confu_mat = evaluate_model(
                model, X_test, Y_test)  # evaluate model

        else:  # if trained
            model = load_model(save_model_dir)
            acc, confu_mat = evaluate_model(model, X_test, Y_test)

        numParams = model.count_params()
        info_dict['numParams'] = numParams
        # save confu_matrix to save_to_csv
        confu_matrix = my_append_row(confu_matrix, confu_mat)
        acc_list = np.append(acc_list, acc)  # save each sub's acc
        print("Classification accuracy: %f " % (acc))

    # save to file
    end = time()
    training_time = (end - start) / 3600
    total_confu_matrix = my_append_col(total_confu_matrix, confu_matrix)
    info_dict['training_time'] = str(training_time) + ' hours'
    save_acc_pic(acc_list, save_dir)
    save_csv(confu_matrix, confu_matri, save_dir)
    save_txt(info_dict, cfg, save_dir)
save_total_csv(total_confu_matrix, root_dir)

