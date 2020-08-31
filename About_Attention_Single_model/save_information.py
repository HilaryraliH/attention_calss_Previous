'''保存的信息：
    训练过程的趋势图           ---->8个图
    每个sub的confusion matrix ----> 1个表（含8个matrix）
    sub的平均confusion matrix ----> 1个表（含1个matrix）
    每个sub的acc              ----> 1个图、1个表
    本次的epochs，batchsize   ---->  1个表
    '''

# coding=utf-8
import os
import numpy as np
import matplotlib.pyplot as plt


def my_append_row(total, one):
    if total is None:
        total = np.array(one)
    else:
        total = np.append(total, one, axis=0)
    return total

def my_append_col(total, one):
    if total is None:
        total = np.array(one)
    else:
        total = np.append(total, one, axis=1)
    return total


def my_concatenate(total, one):
    if total is None:
        total = one
    else:
        total = np.concatenate((total, one), axis=0)
    return total


def check_path(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except:
            print('make dir error')
            return


def save_training_pic(sub, hist, save_dir):
    #creat acc directory
    save_acc_dir = save_dir + 'acc_trend\\'
    check_path(save_acc_dir)
    save_acc_dir = save_acc_dir + str(sub) + '.png'

    plt.figure()
    metric = 'accuracy'
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(save_acc_dir, bbox_inches='tight')
    plt.close()

    # create loss directory
    save_loss_dir = save_dir + 'loss_trend\\'
    check_path(save_loss_dir)
    save_loss_dir = save_loss_dir + str(sub) + '.png'

    plt.figure()
    metric = 'loss'
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(save_loss_dir, bbox_inches='tight')
    plt.close()


def save_acc_pic(acc_list, save_dir):
    plt.figure()
    plt.plot(acc_list)
    plt.title('aver acc = ' + str(np.mean(acc_list)))
    plt.ylabel('acc', fontsize='large')
    plt.xlabel('sub', fontsize='large')
    plt.savefig(save_dir + 'each_acc.png', bbox_inches='tight')
    plt.close()
    np.savetxt(save_dir + 'each_acc.csv', acc_list, delimiter=',')


def save_csv(confu_matrix, save_dir):
    np.savetxt(save_dir + '8sub_confu_matrix.csv', confu_matrix, delimiter=',')
    #aver_confu_mat = np.mean(confu_matri, axis=0)
    #np.savetxt(save_dir + 'aver_confu_matrix.csv', aver_confu_mat, delimiter=',')

def save_total_csv(confu_matrix, save_dir):
    np.savetxt(save_dir + 'total_confu_matrix.csv', confu_matrix, delimiter=',')

def save_txt(info_dict,cfg,save_dir):
    save_dir = save_dir + 'para-num__training-time.txt'  # 文件路径
    f = open(save_dir, 'w', encoding='utf-8')  # 以'w'方式打开文件
    for k, v in info_dict.items():  # 遍历字典中的键值
        s = str(v)  # 把字典的值转换成字符型
        f.write(k + '\n')  # 键和值分行放，键在单数行，值在双数行
        f.write(s + '\n')
    f.write('epochs:'+str(cfg.epochs)+'\n')
    f.write('batch_size:'+str(cfg.batch_size)+'\n')
    f.close()
