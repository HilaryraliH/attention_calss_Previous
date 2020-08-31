# coding=utf-8
import os
import scipy.io as sio
from keras.utils import to_categorical
from keras import backend as K
import numpy as np
from my_error import DataFormatError, ModelNameError

'''def check_data_format(model_name, cfg):
    if model_name in cfg.model_2D_list:
        data_format = '2D'
    elif model_name in cfg.model_3D_list:
        data_format = '3D'
    else:
        raise ModelNameError(model_name)
    return data_format'''

'''现在的，要提取62导里面的数据
两种提取方式: F_9: 额叶电极+原来的9导
             F_C_P: 三个脑区的数据 '''


def select_chans(data, select_chan_way):
    optimal_9_elec = [16, 24, 54, 55, 57, 58, 59, 60, 61]
    EOG_elec = [1, 6]
    F_elec = [3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    C_elec = [17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
    P_elec = [2, 5, 16, 24, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]
    chans = None
    if select_chan_way == '9':
        chans = optimal_9_elec
    if select_chan_way == 'F_C_P':
        chans = F_elec +C_elec+P_elec
    if select_chan_way == 'F_C_9':
        chans = F_elec + C_elec+optimal_9_elec
    if select_chan_way == 'F_9':
        chans = F_elec + optimal_9_elec
    if select_chan_way=='F_P':
        chans = F_elec+P_elec
    if select_chan_way == 'F':
        chans = F_elec
    if select_chan_way == 'C':
        chans = C_elec
    if select_chan_way == 'P':
        chans = P_elec
    result = data[:, chans, :]
    return result


def load_sub(total_sub_num, sub, data_format, select_chan_way):
    # '2D':eletrodes_to_chans  (1,200,9)
    # '3D':eletrodes_to_high   (9,200,1)

    # load total data
    file = sio.loadmat('new_data\\TestDataCell_62.mat')
    data = file['Data']  # (1, 8) (200, 62, 1198) 下标为[0，4]的数据：(200, 62, 886)
    label = file['Label']  # (1, 8) (1, 1198) 下标为[0，4]的数据：(1, 886)

    # extract test data
    tmp = data[0, sub]  # (200, 62, 1198)
    tmp = select_chans(tmp, select_chan_way)  # (200, chans, 1198)
    nums = tmp.shape[2]
    test_x = np.zeros((nums, 1, tmp.shape[0], tmp.shape[1]))
    for j in range(nums):
        test_x[j, 0, :, :] = tmp[:, :, j]
    test_y = label[0, sub].reshape((tmp.shape[2], 1))
    test_y = to_categorical(test_y)

    # extract training data
    total_nums = 0
    chans_num = 0
    for j in range(total_sub_num):
        tmp_train = data[0, j]
        tmp_train = select_chans(tmp_train, select_chan_way)  # (200, chans, 1198)
        chans_num = tmp_train.shape[1]
        if j != sub:
            total_nums += tmp_train.shape[2]
    train_x = np.zeros((total_nums, 1, 200, chans_num))
    train_y = np.zeros((total_nums,))
    indx = 0
    for j in range(total_sub_num):
        tmp_train = data[0, j]
        tmp_train = select_chans(tmp_train, select_chan_way)  # (200, chans, 1198)
        if j != sub:
            tmp_x = tmp_train  # (200, chans, 1198)
            tmp_y = label[0, j]  # (1, 1198)
            nums = tmp_x.shape[2]
            for k in range(nums):
                train_x[indx, 0, :, :] = tmp_x[:, :, k]
                train_y[indx] = tmp_y[0, k]
                indx += 1
    train_y = train_y.reshape((total_nums, 1))
    train_y = to_categorical(train_y)
    rand_inx = np.random.permutation(range(train_x.shape[0]))  # 打乱数据
    train_x = train_x[rand_inx]
    train_y = train_y[rand_inx]
    model_input1 = (train_x, train_y), (test_x, test_y)  # (1,200,chans_num)

    train_x = np.swapaxes(np.squeeze(train_x), 1, 2)  # change shape from (1,200,chans_num) to (chans_num，200)
    test_x = np.swapaxes(np.squeeze(test_x), 1, 2)
    model_input2 = (train_x, train_y), (test_x, test_y)

    train_x = np.expand_dims(train_x, axis=3)  # change shape from (chans_num，200) to (chans_num，200，1)
    test_x = np.expand_dims(test_x, axis=3)
    model_input3 = (train_x, train_y), (test_x, test_y)

    if data_format == '3D':
        return model_input1  # (1,200,chans_num)
    elif data_format == 'true_2D':
        return model_input2  # (chans_num，200)
    elif data_format == '2D':
        return model_input3  # (chans_num，200，1)
    else:
        raise DataFormatError(data_format)


'''原来的load_sub函数'''
'''def load_sub(total_sub_num, sub, data_format):
    # '2D':eletrodes_to_chans  (1,200,9)
    # '3D':eletrodes_to_high   (9,200,1)

    # load total data
    file = sio.loadmat('TestDataCell.mat')
    data = file['Data']  # (1, 8) (200, 9, 1198) 下标为[0，4]的数据：(200, 9, 886)
    label = file['Label']  # (1, 8) (1, 1198) 下标为[0，4]的数据：(1, 886)

    # extract test data
    print('extract test data ing')
    tmp = data[0, sub]  # (200, 9, 1198)
    nums = tmp.shape[2]
    test_x = np.zeros((nums, 1, tmp.shape[0], tmp.shape[1]))
    for j in range(nums):
        test_x[j, 0, :, :] = tmp[:, :, j]
    test_y = label[0, sub].reshape((tmp.shape[2], 1))
    test_y = to_categorical(test_y)

    # extract training data
    print('extract training data ing')
    total_nums = 0
    for j in range(total_sub_num):
        if j != sub:
            total_nums += data[0, j].shape[2]
    train_x = np.zeros((total_nums, 1, 200, 9))
    train_y = np.zeros((total_nums,))
    indx = 0
    for j in range(total_sub_num):
        if j != sub:
            tmp_x = data[0, j]  # (200, 9, 1198)
            tmp_y = label[0, j]  # (1, 1198)
            nums = tmp_x.shape[2]
            for k in range(nums):
                train_x[indx, 0, :, :] = tmp_x[:, :, k]
                train_y[indx] = tmp_y[0, k]
                indx += 1
    train_y = train_y.reshape((total_nums, 1))
    train_y = to_categorical(train_y)
    rand_inx = np.random.permutation(range(train_x.shape[0]))  # 打乱数据
    train_x = train_x[rand_inx]
    train_y = train_y[rand_inx]
    model_input1 = (train_x, train_y), (test_x, test_y) # (1,200,9)

    # change shape from (1,200,9) to (9，200)
    train_x = np.swapaxes(np.squeeze(train_x), 1, 2)
    test_x = np.swapaxes(np.squeeze(test_x), 1, 2)
    model_input2 = (train_x, train_y), (test_x, test_y)

    # change shape from (9，200) to (9，200，1)
    train_x = np.expand_dims(train_x, axis=3)
    test_x = np.expand_dims(test_x, axis=3)
    model_input3 = (train_x, train_y), (test_x, test_y)


    if data_format == '3D':
        return model_input1 # (1,200,9)
    elif data_format =='true_2D':
        return model_input2 #(9，200)
    elif data_format=='2D':
        return model_input3 #(9，200，1)
    else:
        raise DataFormatError(data_format)
'''


def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))
