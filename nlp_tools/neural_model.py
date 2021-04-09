#!/usr/bin/python
#-*- encoding: utf-8 -*-
import numpy as np
from keras.layers import Input, Dense, LSTM, Embedding, GlobalAveragePooling1D
from keras.models import Sequential
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers
import tensorflow as tf

vocab_size = 30000
max_seq_len = 800
embedding_dim = 200
lstm_units = 200

##
# @brief batch数据生成器，当显存不够时，用于keras fit_generator，每次load一批数据
#
# @param X 样本特征
# @param Y 样本标签
#
# @return batch数据生成器
def batch_generator(X, Y, batch_size):
    start = 0
    while True:
        stop = start + batch_size
        diff = stop - X.shape[0]
        if diff <= 0:
            batch = (X[start: stop], Y[start: stop])
            start += batch_size
        else:
            batch = (np.concatenate((X[start:], X[:diff])),
                     np.concatenate((Y[start:], Y[:diff])))
            start = diff
            index = np.arange(len(Y))
            np.random.shuffle(index)
            X = X[index]
            Y = Y[index]
        yield batch


##
# @brief 对LSTM输出进行mean of time pooling, 由于GlobalAveragePooling1D不支持masking，参考Stack Overflow的实现
#
#
class MaskedMeanPooling1D(Layer):
    def __init__(self, **kwargs):
        super(MaskedMeanPooling1D, self).__init__(**kwargs)
        self.supports_masking = True
    
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.repeat(mask, x.shape[-1])
            mask = tf.transpose(mask, [0, 2, 1])
            x = x * mask
            return K.sum(x, axis=1) / K.sum(mask, axis=1)
        return K.mean(x, axis=1)
    
    def compute_output_shape(self, input_shape):
        # remove temporal dimension
        return (input_shape[0], input_shape[2])

##
# @brief 对LSTM的输出进行attention pooling，Keras没有实现attention层
#
#
class Attention(Layer):
    def __init__(self, init_stddev=0.01, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.init_stddev = init_stddev
        self.initializer = initializers.random_normal(stddev=init_stddev)
        self.supports_masking = True

    def build(self, input_shape):
        self.att_v = self.add_weight(name="att_v",
                                     shape=(input_shape[2], 1),
                                     initializer=self.initializer)
        self.att_W = self.add_weight(name="att_W",
                                     shape=(input_shape[2], input_shape[2]),
                                     initializer=self.initializer)
        self.trainable_weights = [self.att_v, self.att_W]
        self.built = True

    def call(self, inputs, mask=None):
        y = K.dot(inputs, self.att_W)
        y = K.tanh(y)
        attention = K.dot(y, self.att_v)
        attention = K.softmax(K.batch_flatten(attention))
        attention = K.permute_dimensions(K.repeat(attention, inputs.shape[2]),
                                         [0, 2, 1])
        output = inputs * attention
        output = K.sum(output, axis=1)
        return K.cast(output, K.floatx())

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

##
# @brief 返回一个基于LSTM的二分类keras模型
#
# @param embedding_dim word embedding的维度
# @param vocab_size 词汇表大小
# @param max_seq_len 最大序列长度
# @param lstm_unints LSTM单元数
# @param dense_units 全连接层单元数
#
# @return keras sequentail model
def get_LSTM_model(embedding_dim, vocab_size, max_seq_len, lstm_units, dense_units):
    model = Sequential()
    model.add(Embedding(output_dim=embedding_dim, input_dim=vocab_size + 1, mask_zero=True,
                        input_length=max_seq_len))
    model.add(LSTM(lstm_units, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(lstm_units, return_sequences=True, recurrent_dropout=0.2, dropout=0.4))
    # mean over time pooling / attention pooling
    model.add(Attention())

    model.add(Dense(dense_units, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())
    
    return model

