from model import *
from data import *
from save_info import *
from params import *
import numpy as np
from keras.layers import Input
print("我正在运行10点24分的程序")

'''明天：metrics有很多，loss也有很多，怎么弄成相应输出只有对应的loss和metrics，而不是交叉组合'''
def train_unet_one(sub):

    model = eval(unet_base_name)(Input(shape=input_size))
    model.compile(optimizer=Adam(lr=unet_base_lr), loss='mean_squared_error', metrics=['mse'])
    if sub==0:
        print('unet_base_model structure:')
        print('-'*30)
        model.summary()
        print('-'*30)
    (train_x, train_y), (test_x, test_y) = load_sub(sub)

    '''对数据做归一化'''
    for i in range(train_x.shape[0]):
        train_x[i] = MaxMinNormalization(train_x[i])
    for j in range(test_x.shape[0]):
        test_x[j] = MaxMinNormalization(test_x[j])

    '''训练'''
    model_checkpoint = ModelCheckpoint(save_unet_base_param_dir+'unet_atten{}.hdf5'.format(sub), monitor='loss',verbose=2, save_best_only=True)
    model.fit(train_x,train_x,validation_data=(test_x,test_x),batch_size=unet_base_batch_size,epochs=unet_base_epochs,callbacks=[model_checkpoint])

    '''取出输入 和 输出， 可视化, 并进行比较'''
    train_sample, train_out, test_sample, test_out = get_unet_base_plot_data(model,train_x,test_x,plot_number)
    plot_result(sub,train_sample, train_out, test_sample, test_out,save_retrieve_fig_dir)


for sub in range(sub_num):
    unet_clas_model = None
    unet_base_weight = save_unet_base_param_dir+'unet_atten{}.hdf5'.format(sub)
    if train_unet and is_pretrained:
        train_unet_one(sub)
    if train_classifier:
        '''在unet_base的基础上 加入分类器，建立新模型'''
        inputs = Input(input_size)
        unet_clas_model = unet_classifier(inputs,unet_base_weight,freeze_unet)
        if two_output:
            unet_clas_model.compile(loss={'classification':'binary_crossentropy','rebuild':'mean_squared_error'},
                                    optimizer='adam',
                                    metrics={'classification':'accuracy','rebuild':'mse'},
                                    loss_weights={'classification':5,'rebuild':1000})
        else:
            unet_clas_model.compile(optimizer=Adam(lr=unet_clas_model_lr), loss='binary_crossentropy', metrics=['accuracy'])

        '''查看模型结构'''
        if sub==0:
            print('unet_base_model structure:')
            print('-'*50)
            unet_clas_model.summary()
            print('-'*50)

        '''载入数据，并归一化'''
        (train_x, train_y), (test_x, test_y) = load_sub(sub)
        for i in range(train_x.shape[0]):
            train_x[i] = MaxMinNormalization(train_x[i])
        for j in range(test_x.shape[0]):
            test_x[j] = MaxMinNormalization(test_x[j])

        '''训练，保存训练趋势图'''
        model_checkpoint = ModelCheckpoint(save_unet_clas_param_dir+'unet_clas_model{}.hdf5'.format(sub),
                                           monitor='accuracy',verbose=2, save_best_only=True)
        if two_output:
            hist = unet_clas_model.fit(train_x, {'classification':train_y,'rebuild':train_x},
                                       validation_data=(test_x, {'classification':test_y,'rebuild':test_x}),
                                       batch_size=unet_clas_model_batch_size, epochs=unet_clas_model_epochs, verbose=2,
                                       callbacks=[model_checkpoint])
            '''取出rebuild层的输入 和 输出， 可视化, 并进行比较'''
            train_sample, train_out, test_sample, test_out = get_unet_base_plot_data(unet_clas_model, train_x, test_x,plot_number)
            plot_result(sub, train_sample, train_out, test_sample, test_out, save_retrieve_fig_dir)

        else:
            hist = unet_clas_model.fit(train_x,train_y,validation_data=(test_x,test_y),batch_size=unet_clas_model_batch_size,epochs=unet_clas_model_epochs,verbose=2,callbacks=[model_checkpoint])
        save_training_pic(sub, hist, save_trend_dir)

        '''测试，保存测试acc图'''
        acc, __ = evaluate_model(unet_clas_model, test_x, test_y)
        acc_list = np.append(acc_list, acc)
        save_acc_pic(acc_list, save_trend_dir)

