# coding=utf-8

from config import configur
from time import time
from data_pre import *
from save_information import *
from train_each_model import erect_model, fit_model, evaluate_model
from keras.models import load_model
from config import *


root_dir = None
total_confu_matrix = None
for once in range(cfg.total_times):
    start = time()
    root_dir, save_dir = mk_save_dir(model_name, debug, validation_split, Desktop_or_GPU,once)
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
        print('loading sub {} data'.format(sub))
        (X_train, Y_train), (X_test, Y_test) = load_sub(total_sub_num, sub, cfg.data_format, cfg.select_chan_way) # load data
        print('{},{}     {},{}'.format(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape))
        '''erect model'''
        if not os.path.exists(save_model_dir):  # if not trained
            model = erect_model(cfg)
            if sub == 0 and once == 0:
                model.summary()
            print('\n=============================== start training {}th fold ================================\n'.format(sub))
            hist = fit_model(model, cfg, X_train, Y_train, X_test, Y_test, save_model_dir, validation_split)
            save_training_pic(sub, hist, save_dir)  # save the training trend
            acc, confu_mat = evaluate_model(model, X_test, Y_test)  # evaluate model
        else:  # if trained
            model = load_model(save_model_dir)
            acc, confu_mat = evaluate_model(model, X_test, Y_test)

        numParams = model.count_params()
        info_dict['numParams'] = numParams
        confu_matrix = my_append_row(confu_matrix, confu_mat)  # save confu_matrix to save_to_csv
        acc_list = np.append(acc_list, acc)  # save each sub's acc
        print("Classification accuracy: %f " % (acc))

    # save to file
    end = time()
    training_time = (end - start) / 3600
    total_confu_matrix = my_append_col(total_confu_matrix,confu_matrix)
    info_dict['training_time'] = str(training_time) + ' hours'
    save_acc_pic(acc_list, save_dir)
    save_csv(confu_matrix, save_dir)
    save_txt(info_dict, cfg, save_dir)
save_total_csv(total_confu_matrix,root_dir)

