import os
from keras.utils import plot_model
from mod_stru import *
from config import *

def check_path(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except:
            print('make dir error')
            return


save_md_stru_dir = '.\\save_model_stru_png\\'
check_path(save_md_stru_dir)


os.environ["PATH"] += os.pathsep + 'C:/C1_Install_package/Graphviz/Graphviz 2.44.1/bin'

samples = 200
model_input = Input(shape=(1,200,cfg.chans),name='input__R')
model = eval(model_name)(model_input,cfg, nb_classes=2)
plot_model(model,to_file=save_md_stru_dir+model_name+'.png',show_shapes=True)
