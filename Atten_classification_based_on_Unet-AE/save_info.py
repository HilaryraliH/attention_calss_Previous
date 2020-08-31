# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from params import *


def get_unet_base_plot_data(model, train_x, test_x, nums):
    train_sample = train_x[nums]
    test_sample = test_x[nums]
    if two_output:
        train_out = model.predict(np.expand_dims(train_sample, axis=0))[1]
        test_out = model.predict(np.expand_dims(test_sample, axis=0))[1]
    else:
        train_out = model.predict(np.expand_dims(train_sample, axis=0))
        test_out = model.predict(np.expand_dims(test_sample, axis=0))

    plt.figure()
    train_sample = np.squeeze(train_sample, axis=2)
    train_out = np.squeeze(np.squeeze(train_out, axis=0), axis=2)
    test_sample = np.squeeze(test_sample, axis=2)
    test_out = np.squeeze(np.squeeze(test_out, axis=0), axis=2)
    return train_sample, train_out, test_sample, test_out


def plot_result(sub, train_sample, train_out, test_sample, test_out, save_retrieve_fig_dir):
    for i in range(train_sample.shape[0]):

        plt.subplot2grid((train_sample.shape[0], 2), (i, 0))
        plt.plot(range(len(train_sample[i])), train_sample[i])
        plt.plot(range(len(train_sample[i])), train_out[i], 'r')
        if not i:
            plt.title('blue: train_sample  red: train_predict')

        plt.subplot2grid((train_sample.shape[0], 2), (i, 1))
        plt.plot(range(len(train_sample[i])), test_sample[i])
        plt.plot(range(len(train_sample[i])), test_out[i], 'r')
        if not i:
            plt.title('blue: test_sample  red: test_predict')
    plt.savefig(save_retrieve_fig_dir + '\\{}.png'.format(sub))


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


def save_training_pic(sub, hist, save_dir):
    # creat acc directory

    if two_output:
        '''画分类层的准确率'''
        save_acc_dir = save_dir + 'acc_trend\\'
        check_path(save_acc_dir)
        save_acc_dir = save_acc_dir + str(sub) + '.png'

        plt.figure()
        metric = classification_layer_name+ '_accuracy' #.format(3 * (sub + 1))
        # metric = 'activation_3_accuracy'
        plt.plot(hist.history[metric])
        plt.plot(hist.history['val_' + metric])
        plt.title('model ' + metric)
        plt.ylabel(metric, fontsize='large')
        plt.xlabel('epoch', fontsize='large')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(save_acc_dir, bbox_inches='tight')
        plt.close()

        '''画分类层的loss'''
        save_loss_dir = save_dir + 'loss_trend\\'
        check_path(save_loss_dir)
        save_loss_dir = save_loss_dir + str(sub) + '.png'

        plt.figure()
        metric = classification_layer_name+'_loss' # .format(3 * (sub + 1))
        # metric = 'activation_3_loss'
        plt.plot(hist.history[metric])
        plt.plot(hist.history['val_' + metric])
        plt.title('model ' + metric)
        plt.ylabel(metric, fontsize='large')
        plt.xlabel('epoch', fontsize='large')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(save_loss_dir, bbox_inches='tight')
        plt.close()

        '''画重建层的loss'''
        save_mse_dir = save_dir + 'retrieve_mse_trend\\'
        check_path(save_mse_dir)
        save_mse_dir = save_mse_dir + str(sub) + '.png'

        plt.figure()
        metric = rebuild_layer_name+'_mse'.format(19 + 20 * sub)
        # metric = 'conv2d_19_mse'
        plt.plot(hist.history[metric])
        plt.plot(hist.history['val_' + metric])
        plt.title('model ' + metric)
        plt.ylabel(metric, fontsize='large')
        plt.xlabel('epoch', fontsize='large')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(save_mse_dir, bbox_inches='tight')
        plt.close()
    else:
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


def saveResult(results):
    f = open('results.txt', 'w', encoding='utf-8')
    f.write(results)
    f.close()
