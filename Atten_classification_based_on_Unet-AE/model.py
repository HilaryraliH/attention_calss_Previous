
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras import regularizers
from keras.constraints import max_norm
from params import *

'''之前看起来最好的那个模型，未去掉链接'''
def unet(pretrained_weights=None, input_size=(9, 200, 1)):
    inputs = Input(input_size)
    #BN0 = BatchNormalization()(inputs)
    conv1 = Conv2D(32, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    BN1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN1)
    BN1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(1, 2))(BN1)

    conv2 = Conv2D(64, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    BN2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN2)
    BN2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(1, 2))(BN2)

    conv3 = Conv2D(128, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    BN3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN3)
    BN3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(1, 2))(BN3)

    conv4 = Conv2D(256, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    BN4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN4)
    BN4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(BN4)
    # drop4 为 9*25
    # pool4 = MaxPooling2D(pool_size=(1, 2))(drop4) # 这里pool之后，是12.5，也就是12了，之后上采样变成24就会对不齐
    # 可以省略这个pool，并相应省略之后的上采样部分
    conv5 = Conv2D(512, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(drop4)
    BN5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN5)
    BN5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(BN5)
    print('drop5.shape: ', drop5.shape)


    up6 = Conv2D(256, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(drop5)  # UpSampling2D(size = (1,2))(drop5))
    BN6 = BatchNormalization()(up6)
    merge6 = concatenate([drop4, BN6], axis=3)
    conv6 = Conv2D(256, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    BN6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN6)
    BN6 = BatchNormalization()(conv6)

    up7 = Conv2D(128, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(1, 2))(BN6))
    BN7 = BatchNormalization()(up7)
    merge7 = concatenate([conv3, BN7], axis=3)
    conv7 = Conv2D(128, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    BN7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN7)
    BN7 = BatchNormalization()(conv7)

    up8 = Conv2D(64, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(1, 2))(BN7))
    BN8 = BatchNormalization()(up8)
    merge8 = concatenate([conv2, BN8], axis=3)
    conv8 = Conv2D(64, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    BN8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN8)
    BN8 = BatchNormalization()(conv8)

    up9 = Conv2D(32, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(1, 2))(BN8))
    BN9 = BatchNormalization()(up9)
    merge9 = concatenate([conv1, BN9], axis=3)
    conv9 = Conv2D(32, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    BN9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN9)
    BN9 = BatchNormalization()(conv9)
    conv9 = Conv2D(2, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN9)
    BN9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid',name=rebuild_layer_name)(BN9) # 或者用tanh

    model = Model(input=inputs, output=conv10)
    model.summary()

    return model

'''去掉了链接'''
def unet4(inputs,pretrained_weights=None):
    conv1 = Conv2D(32, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    BN1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN1)
    BN1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(1, 2))(BN1)

    conv2 = Conv2D(64, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    BN2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN2)
    BN2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(1, 2))(BN2)

    conv3 = Conv2D(128, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    BN3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN3)
    BN3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(1, 2))(BN3)

    conv4 = Conv2D(256, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    BN4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN4)
    BN4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(BN4)
    # drop4 为 9*25
    # pool4 = MaxPooling2D(pool_size=(1, 2))(drop4) # 这里pool之后，是12.5，也就是12了，之后上采样变成24就会对不齐
    # 可以省略这个pool，并相应省略之后的上采样部分

    #######################################################################################################
    conv5 = Conv2D(512, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(drop4)
    BN5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN5)
    BN5 = BatchNormalization()(conv5) # name='middle_out_layer'
    drop5 = Dropout(0.5)(BN5)
    print('drop5.shape: ', drop5.shape)
    ########################################################################################################


    up6 = Conv2D(256, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(drop5)  # UpSampling2D(size = (1,2))(drop5))
    BN6 = BatchNormalization()(up6)
    #merge6 = concatenate([drop4, BN6], axis=3)
    conv6 = Conv2D(256, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN6)
    BN6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN6)
    BN6 = BatchNormalization()(conv6)

    up7 = Conv2D(128, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(1, 2))(BN6))
    BN7 = BatchNormalization()(up7)
    #merge7 = concatenate([conv3, BN7], axis=3)
    conv7 = Conv2D(128, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN7)
    BN7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN7)
    BN7 = BatchNormalization()(conv7)

    up8 = Conv2D(64, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(1, 2))(BN7))
    BN8 = BatchNormalization()(up8)
    #merge8 = concatenate([conv2, BN8], axis=3)
    conv8 = Conv2D(64, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN8)
    BN8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN8)
    BN8 = BatchNormalization()(conv8)

    up9 = Conv2D(32, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(1, 2))(BN8))
    BN9 = BatchNormalization()(up9)
    #merge9 = concatenate([conv1, BN9], axis=3)
    conv9 = Conv2D(32, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN9)
    BN9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN9)
    BN9 = BatchNormalization()(conv9)
    conv9 = Conv2D(2, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN9)
    BN9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid',name=rebuild_layer_name)(BN9) # 或者用tanh

    model = Model(input=inputs, output=conv10)
    return model

'''在unet4的基础上，降低了层数，降低了filter数量'''
def unet3(inputs,pretrained_weights=None):
    conv1 = Conv2D(8, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    BN1 = BatchNormalization()(conv1)
    conv1 = Conv2D(8, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN1)
    BN1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(1, 2))(BN1)

    conv2 = Conv2D(16, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    BN2 = BatchNormalization()(conv2)
    conv2 = Conv2D(16, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN2)
    BN2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(1, 2))(BN2)

    conv3 = Conv2D(32, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    BN3 = BatchNormalization()(conv3)
    conv3 = Conv2D(32, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN3)
    BN3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(1, 2))(BN3)

    #######################################################################################################
    conv5 = Conv2D(64, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    BN5 = BatchNormalization()(conv5)
    conv5 = Conv2D(64, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN5)
    BN5 = BatchNormalization()(conv5) # name='middle_out_layer'
    drop5 = Dropout(0.5)(BN5)
    print('drop5.shape: ', drop5.shape)
    ########################################################################################################


    up7 = Conv2D(32, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(1, 2))(drop5))
    BN7 = BatchNormalization()(up7)
    conv7 = Conv2D(32, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN7)
    BN7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN7)
    BN7 = BatchNormalization()(conv7)

    up8 = Conv2D(16, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(1, 2))(BN7))
    BN8 = BatchNormalization()(up8)
    conv8 = Conv2D(16, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN8)
    BN8 = BatchNormalization()(conv8)
    conv8 = Conv2D(16, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN8)
    BN8 = BatchNormalization()(conv8)

    up9 = Conv2D(8, (1, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(1, 2))(BN8))
    BN9 = BatchNormalization()(up9)

    conv9 = Conv2D(8, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN9)
    BN9 = BatchNormalization()(conv9)
    conv9 = Conv2D(8, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN9)
    BN9 = BatchNormalization()(conv9)
    conv9 = Conv2D(2, (1, 4), activation='relu', padding='same', kernel_initializer='he_normal')(BN9)
    BN9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid',name=rebuild_layer_name)(BN9) # 或者用tanh

    model = Model(input=inputs, output=conv10)

    return model

'''过拟合的分类器（接在unet4后面的）（但也可以接在unet3之后试一下，因为免得训练不够）'''
def unet_classifier(inputs, unet_base_weight,freeze_unet, is_spatial=False):
    unet_base = eval(unet_base_name)(inputs)  # 若要建立成功，在unet模型建立时，一定要有传参，也就是输入，好让keras找到链接点，不然会断掉，无法建模
    if is_pretrained:
        unet_base.load_weights(unet_base_weight)
        if freeze_unet:
            for layer in unet_base.layers:
                layer.trainable = False
    if unet_base_name == 'unet3':
        block1 = Conv2D(32, (1, 4), padding='same')(unet_base.layers[19].output)
    elif unet_base_name == 'unet4':
        block1 = Conv2D(32, (1, 4), padding='same')(unet_base.layers[24].output)
    class_1 = BatchNormalization()(block1)
    class_1 = Conv2D(16, (1, 4), activation='relu')(class_1) # 是否要加一个空间的
    class_1 = BatchNormalization()(class_1)
    class_2 = Conv2D(16,(1,4),activation='relu', padding='same')(class_1)
    class_2 = BatchNormalization()(class_2)
    class_3= Conv2D(8,(1,4))(class_2)
    class_3 = BatchNormalization()(class_3)
    flattened = Flatten()(class_3)
    predict = Dense(2,activation='softmax',name=classification_layer_name)(flattened)

    if two_output:
        unet_clas_model = Model(inputs, [predict,unet_base.layers[-1].output])
    else:
        unet_clas_model = Model(inputs,predict)
    return unet_clas_model

'''L2，dropout等防止过拟合的传统conv分类器（接在unet3后面的）  unet3中间的输出是9，25，64'''
# def unet_classifier(inputs, unet_base_weight,freeze_unet, is_spatial=False):
#     unet_base = eval(unet_base_name)(inputs) # 若要建立成功，在unet模型建立时，一定要有传参，也就是输入，好让keras找到链接点，不然会断掉，无法建模
#     if is_pretrained:
#         unet_base.load_weights(unet_base_weight)
#         if freeze_unet:
#             for layer in unet_base.layers:
#                 layer.trainable = False
#     if unet_base_name=='unet3':
#         class_1 = Conv2D(32,(1,4),activation='relu', padding='same',
#                      kernel_regularizer=regularizers.l2(0.01),
#                      bias_regularizer=regularizers.l2(0.01))(unet_base.layers[19].output)
#     elif unet_base_name=='unet4':
#         class_1 = Conv2D(32,(1,4),activation='relu', padding='same',
#                      kernel_regularizer=regularizers.l2(0.01),
#                      bias_regularizer=regularizers.l2(0.01))(unet_base.layers[24].output)
#
#     class_1 = BatchNormalization()(class_1)
#     class_1 = MaxPooling2D((1,2))(class_1)
#     class_1 = Dropout(0.3)(class_1)
#
#     class_2 = Conv2D(8,(1,4),activation='relu', padding='same',
#                      kernel_regularizer=regularizers.l2(0.01),
#                      bias_regularizer=regularizers.l2(0.01))(class_1)
#     class_2 = BatchNormalization()(class_2)
#
#     class_3= Conv2D(4,(1,4),kernel_regularizer=regularizers.l2(0.01))(class_2)
#     class_3 = BatchNormalization()(class_3)
#
#     flattened = Flatten()(class_3)
#     predict = Dense(2,activation='softmax',name=classification_layer_name)(flattened)
#
#     if two_output:
#         unet_clas_model = Model(inputs, [predict,unet_base.layers[-1].output])
#     else:
#         unet_clas_model = Model(inputs,predict)
#     return unet_clas_model

'''不用传统conv分类，改为EEGNet做分类器'''
# def unet_classifier(inputs, unet_base_weight,freeze_unet, is_spatial=False):
#     unet_base = eval(unet_base_name)(inputs)  # 若要建立成功，在unet模型建立时，一定要有传参，也就是输入，好让keras找到链接点，不然会断掉，无法建模
#
#     if is_pretrained:
#         unet_base.load_weights(unet_base_weight)
#         if freeze_unet:
#             for layer in unet_base.layers:
#                 layer.trainable = False
#
#     if unet_base_name == 'unet3':
#         block1 = Conv2D(16, (1, 8), padding='same', input_shape=(9, 200, 1))(unet_base.layers[19].output)
#     elif unet_base_name == 'unet4':
#         block1 = Conv2D(16, (1, 8), padding='same', input_shape=(9, 200, 1))(unet_base.layers[24].output)
#     block1 = BatchNormalization()(block1)  # I'm not sure the axis, axis=1 before
#
#     block1 = DepthwiseConv2D((9, 1), use_bias=False, depth_multiplier=2)(block1)
#     block1 = BatchNormalization()(block1)
#     block1 = Activation('elu')(block1)
#     block1 = AveragePooling2D((1, 2))(block1)
#     block1 = Dropout(0.3)(block1)
#
#     block2 = SeparableConv2D(8, (1, 8), use_bias=False, padding='same')(block1)  # it's（1，16）before
#     block2 = BatchNormalization()(block2)
#     block2 = Activation('elu')(block2)
#     # 这里去掉了一个pool层，因为到了这里length就有点小了
#     block2 = Dropout(0.3)(block2)
#
#     flatten = Flatten()(block2)
#     dense = Dense(2, kernel_constraint=max_norm(0.25))(flatten)
#     predict = Activation('softmax',name=classification_layer_name)(dense)
#     if two_output:
#         unet_clas_model = Model(inputs, [predict,unet_base.layers[-1].output])
#     else:
#         unet_clas_model = Model(inputs,predict)
#     return unet_clas_model

'''重新将每层命名'''
# def unet_classifier(inputs, unet_base_weight,freeze_unet, is_spatial=False):
#     unet_base = eval(unet_base_name)(inputs)  # 若要建立成功，在unet模型建立时，一定要有传参，也就是输入，好让keras找到链接点，不然会断掉，无法建模
#
#     if is_pretrained:
#         unet_base.load_weights(unet_base_weight)
#         if freeze_unet:
#             for layer in unet_base.layers:
#                 layer.trainable = False
#     if unet_base_name=='unet3':
#         block1 = Conv2D(16, (1, 8), padding='same')(unet_base.layers[19].output)
#     elif unet_base_name=='unet4':
#         block1 = Conv2D(16, (1, 8), padding='same')(unet_base.layers[24].output)
#     block1 = BatchNormalization()(block1)  # I'm not sure the axis, axis=1 before
#
#     block1 = DepthwiseConv2D((9, 1), use_bias=False, depth_multiplier=2)(block1)
#     block1 = BatchNormalization()(block1)
#     block1 = Activation('elu')(block1)
#     block1 = AveragePooling2D((1, 2))(block1)
#     block1 = Dropout(0.3)(block1)
#
#     block2 = SeparableConv2D(8, (1, 8), use_bias=False, padding='same')(block1)  # it's（1，16）before
#     block2 = BatchNormalization()(block2)
#     block2 = Activation('elu')(block2)
#     block2 = Dropout(0.3)(block2)
#
#     flatten = Flatten()(block2)
#     dense = Dense(2, kernel_constraint=max_norm(0.25))(flatten)
#     predict = Activation('softmax',name=classification_layer_name)(dense)
#
#     if two_output:
#         unet_clas_model = Model(inputs, [predict,unet_base.layers[-1].output])
#     else:
#         unet_clas_model = Model(inputs,predict)
#     return unet_clas_model

'''不用传统conv分类，改为Pro做分类器'''
# def unet_classifier(inputs, unet_base_weight,freeze_unet, is_spatial=False):
#     unet_base = eval(unet_base_name)(inputs)  # 若要建立成功，在unet模型建立时，一定要有传参，也就是输入，好让keras找到链接点，不然会断掉，无法建模
#
#     if is_pretrained:
#         unet_base.load_weights(unet_base_weight)
#         if freeze_unet:
#             for layer in unet_base.layers:
#                 layer.trainable = False
#     if unet_base_name=='unet3':
#         block1 = Conv2D(16, (1, 8), padding='same')(unet_base.layers[19].output)
#     elif unet_base_name=='unet4':
#         block1 = Conv2D(16, (1, 8), padding='same')(unet_base.layers[24].output)
#     block1 = BatchNormalization()(block1)  # I'm not sure the axis, axis=1 before
#
#     block1 = DepthwiseConv2D((1, 5), use_bias=False, depth_multiplier=2)(block1)
#     block1 = BatchNormalization()(block1)
#     block1 = Activation('elu')(block1)
#     block1 = AveragePooling2D((1, 2))(block1)
#     block1 = Dropout(0.3)(block1)
#
#     block2 = SeparableConv2D(8, (1, 8), use_bias=False, padding='same')(block1)  # it's（1，16）before
#     block2 = BatchNormalization()(block2)
#     block2 = Activation('elu')(block2)
#     block2 = Dropout(0.3)(block2)
#
#     flatten = Flatten()(block2)
#     dense = Dense(2, kernel_constraint=max_norm(0.25))(flatten)
#     predict = Activation('softmax',name=classification_layer_name)(dense)
#     if two_output:
#         unet_clas_model = Model(inputs, [predict,unet_base.layers[-1].output])
#     else:
#         unet_clas_model = Model(inputs,predict)
#     return unet_clas_model





def evaluate_model(model, X_test, Y_test):
    probs = model.predict(X_test)
    if two_output:
        probs = probs[0]
    preds = probs.argmax(axis=-1)

    true_label = Y_test.argmax(axis=-1)
    acc = np.mean(preds == true_label)
    print(true_label.shape,'----',preds.shape)
#    confu_mat = confusion_matrix(true_label, preds, labels=[0, 1])
    return acc, acc



