"""
class AttentionLayer(Layer):
https://github.com/uhauha2929/examples/blob/master/Hierarchical%20Attention%20Networks%20.ipynb

"""
import tensorflow as tf
from keras import backend as K
from keras.layers import *
from keras import initializers

class AttLayer(Layer):

    def __init__(self, attention_dim, **kwargs):
         self.init = initializers.get('normal')
         self.supports_masking = True
         self.attention_dim = attention_dim
         super(AttLayer, self).__init__()

    def get_config(self):
        config = {
            'attention_dim': self.attention_dim
        }
        base_config = super(AttLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        assert len(input_shape) == 3
        print('in bulid func:################################################################')
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.Wa = K.variable(self.init((self.attention_dim, self.attention_dim)))
        #self.trainable_weights = [self.W, self.b, self.Wa]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        print('in call func:################################################################')
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        print(x.shape)
        #uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        #xo = K.dot(uit, self.Wa)
        #print(xo.shape)
        #y = K.permute_dimensions(uit,(0, 2, 1))
        # xt = K.batch_dot(x0, y)

        y = K.permute_dimensions(x, (0, 2, 1))
        xt = K.batch_dot(x, y)

        ait = K.softmax(xt, axis=1)
        #at_sum = K.sum(ait, axis=1)
        print(ait.shape)
        #at_sum_repeated = K.repeat(at_sum, x.shape[0])
        #ait /= at_sum_repeated  # vector of size (batchsize, timesteps, attention_dim)

        xy = K.permute_dimensions(x, (0, 2, 1))
        print(xy.shape)
        output = K.batch_dot(xy, ait)
        output = K.permute_dimensions(output, (0, 2, 1))
        print(output.shape)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[-1])

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight', 
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias', 
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        # x.shape = (batch_size, seq_len, time_steps)
        a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        outputs = K.permute_dimensions(a * x, (0, 2, 1))
        outputs = K.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

"""
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[-1],1),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        #x = inputs
        # x.shape = (batch_size, seq_len, time_steps)
        # general
        a = K.softmax(K.tanh(K.dot(x, self.W)))
        #a = K.permute_dimensions(a, (0, 2, 1))
        print(a.shape)
        #print(self.W.shape)
        weighted_input = x*a
        print(weighted_input.shape)
        outputs = K.sum(weighted_input, axis=-1)
        print(outputs.shape)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]
        """