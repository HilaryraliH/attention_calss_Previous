import os

sub_num=8
data_format='2D'
select_chan_way='9'

input_size=(9, 200, 1)
acc_list = []

plot_number = 20 # 可视化第几个解码图像

is_pretrained = True # 是否对unet-base进行预训练
train_unet = False # 是否进行对unet_base的预训练

train_classifier = True # 是否训练分类器
freeze_unet = True # 是否冰冻unet_base
two_output = False

unet_base_name = 'unet3' # unet4  unet3
unet_base_batch_size = 60
unet_base_epochs = 20
unet_base_lr = 1e-3


unet_clas_model_batch_size=60
unet_clas_model_epochs =60
unet_clas_model_lr = 1e-4


save_unet_base_param_dir = 'save_unet_base_param\\'
save_unet_clas_param_dir = 'save_unet_clas_param\\'
save_trend_dir = 'save_train_trend\\'
save_retrieve_fig_dir = 'save_rebuild_fig\\'
classification_layer_name = 'classification'
rebuild_layer_name = 'rebuild'

def check_path(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except:
            print('make dir error')
            return
check_path(save_unet_base_param_dir)
check_path(save_unet_clas_param_dir)
check_path(save_trend_dir)
check_path(save_retrieve_fig_dir)