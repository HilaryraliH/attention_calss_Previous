# coding=utf-8
from params import *
import os
import scipy.io as sio
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

total_sub_num = 8

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

def MaxMinNormalization(x):
    Max = np.max(x)
    Min = np.min(x)
    x = (x - Min) / (Max - Min)
    return x

def load_sub(sub):
    # '2D':eletrodes_to_chans  (1,200,9)
    # '3D':eletrodes_to_high   (9,200,1)

    # load total data
    file = sio.loadmat('TestDataCell_62.mat')
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

    '''if necessary, change y to a full label(coresspond to each time point)'''
    # def full_y(y):
    #     index = np.where(y==1) # 若为[0,1]， 则下标为1   返回本来的下标【0,1,2,3,4...】和对应为1的下标【0,1,1,0,1...】等等
    #     # 0位置为1的，为attention，1位置为1的，为nonattention
    #     index = index[1]# 即变成（9,200,1）的y之后，0为attention，1为nonattention
    #     index=np.expand_dims(index,axis=1).repeat(9,axis=1)
    #     index=np.expand_dims(index,axis=2).repeat(200,axis=2)
    #     result = np.expand_dims(index, axis=3).repeat(1, axis=2)
    #     # print('full_y.shape:',result.shape)
    #     # print(result[0])
    #     return result
    # train_y = full_y(train_y)
    # test_y=full_y(test_y)

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


import numpy as np
'''当 y 的形状为（9,200） 的全0或全1时，计算 pre_y 和 y 之间的 acc'''
def compare(pre_y,test_y):
    assert pre_y.shape[0]==test_y.shape[0]
    # 把pre_y降维 成为 1198*1
    pre_y[np.where(pre_y>0.5)]=1
    pre_y[np.where(pre_y <= 0.5)] = 0
    pre_y_simple = np.zeros(pre_y.shape[0],)
    test_y_simple = np.zeros(test_y.shape[0],)
    from collections import Counter
    for i in range(pre_y.shape[0]):
        pre_yi_vector=Counter(np.squeeze(pre_y[i],axis=2).reshape((pre_y[i].shape[0]*pre_y[i].shape[1],)))
        if pre_yi_vector[0]>pre_yi_vector[1]:
            pre_y_simple[i]=0
        else:
            pre_y_simple[i]=1

        test_yi_vector = Counter(np.squeeze(test_y[i], axis=2).reshape((test_y[i].shape[0] * test_y[i].shape[1],)))
        if test_yi_vector[0]>test_yi_vector[1]:
            test_y_simple[i]=0
        else:
            test_y_simple[i]=1


    print('pre_y_simple.shape:',pre_y_simple.shape)
    print(pre_y_simple)
    print('test_y_simple.shape:',test_y_simple.shape)
    print(test_y_simple)
    acc = np.sum(pre_y_simple==test_y_simple)/pre_y_simple.shape[0]
    return acc