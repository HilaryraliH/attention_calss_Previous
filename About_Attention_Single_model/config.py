from my_error import ModelNameError
total_sub_num = 8
debug = 0 # 是否在debug，若是，则所有参数都会变小来调试
is_CPU = 0
model_name = 'Proposed_Conv_R'  # 随着模型改变
select_chan_way = '9'
validation_split = 0  # 默认0，可以适当较小比例地分离validation
Desktop_or_GPU = 'GPU'  # 'GPU DeskTop'


class configur():
    def __init__(self, model_name, is_CPU, debug,select_chan_way):
        self.model_name_list = ['JNE_CNN', 'EEGNet', 'ShallowConvNet', 'DeepConvNet',
                                'Proposed_Conv', 'Proposed_Conv_R', 'Proposed_Deeper',
                                'Double_path', 'DeepConvNet_no_spatial', 'generate_lstmfcn',
                                'generate_lstmfcn_Conv2D', 'generate_lstmfcn_Conv2D_nolstm',
                                'EEGNet_original','Spatial_model','Transpose_Net',
                                'Transfer_Proposed_Conv_R']
        self.model_2D_list = ['EEGNet', 'ShallowConvNet', 'DeepConvNet', 'DeepConvNet_no_spatial',
                              'generate_lstmfcn_Conv2D', 'generate_lstmfcn_Conv2D_nolstm','Spatial_model',
                              'Transpose_Net','Transfer_Proposed_Conv_R']
        self.model_true_2D_list = ['generate_lstmfcn']
        self.model_3D_list = ['JNE_CNN', 'Proposed_Conv', 'Proposed_Conv_R',
                              'EEGNet_original', 'Proposed_Deeper', 'Double_path']
        self.debug = debug
        self.is_CPU = is_CPU
        self.select_chan_way = select_chan_way
        if model_name not in self.model_name_list:
            raise ModelNameError(model_name)
        self.model_name = model_name
        if self.model_name in self.model_2D_list:
            self.data_format = '2D'  # (9，200，1)
        elif self.model_name in self.model_3D_list:
            self.data_format = '3D'  # (1,200,9)
        elif self.model_name in self.model_true_2D_list:
            self.data_format = 'true_2D'  # (9，200)

    def set_config(self):
        if self.select_chan_way == '9':
            self.chans = 9
        if self.select_chan_way == 'F_C_P':
            self.chans = 59
        if self.select_chan_way=='F_C_9':
            self.chans=45
        if self.select_chan_way =='F_9':
            self.chans = 20
        if self.select_chan_way=='F_P':
            self.chans = 34
        elif self.select_chan_way=='F':
            self.chans = 11
        elif self.select_chan_way=='C':
            self.chans = 25
        elif self.select_chan_way=='P':
            self.chans = 23


        if self.debug:
            self.epochs = 1
            self.batch_size = 1000
            self.total_times = 2
        elif self.is_CPU:
            self.epochs = 20
            self.batch_size = 100
            self.total_times = 1
        else:
            if self.model_name=='ShallowConvNet':
                self.total_times = 1
                self.epochs = 50
                self.batch_size = 128
            else:
                self.total_times = 1
                self.epochs = 30
                self.batch_size = 128


cfg = configur(model_name, is_CPU, debug,select_chan_way)
cfg.set_config()