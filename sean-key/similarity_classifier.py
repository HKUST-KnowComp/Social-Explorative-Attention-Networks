from __future__ import print_function
import numpy as np
import tensorflow as tf
import random as rn
import os
from similarity_function import *
import util

from keras.layers import Input, Activation, Dense, concatenate, Lambda, RepeatVector, Flatten, Dropout
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.layers import Embedding
from keras import regularizers
from keras.engine.topology import Layer
# import param_tw as param

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
tf.set_random_seed(42)
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)
from keras.backend import tensorflow_backend as K
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
K.set_session(tf.Session(config=config))


class AttLayer(Layer):
    def __init__(self, **kwargs):
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.L = input_shape[-2]
        self.H = input_shape[-1]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.H, 1),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(AttLayer, self).build(input_shape)

    def call(self, x, mask=None):
        eij = K.reshape(K.dot(x, self.kernel), (-1, self.L))
        weights = K.repeat_elements(K.expand_dims(
            K.softmax(eij), -1), rep=self.H, axis=-1)
        weighted_input = K.sum(x * weights, axis=1)
        return weighted_input

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class shan():
    def __init__(
            self,
            vocab_size=100000,
            max_len_user=200,
            max_len_doc=90,
            embedding_dim=300,
            hidden_size=64,
            beta=1,
            embedding_matrix=None,
            walk_length=21):
        self.word_level_embedding = None
        self.vocab_size = vocab_size
        self.H = hidden_size
        self.La = max_len_doc
        self.Lu = max_len_user
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        self.beta = beta
        self.wl = walk_length

    def soft_attention(self, values):
        t1, t2 = values
        dim = int(t1.shape[-1])
        relevant_degree = K.repeat_elements(
            K.expand_dims(t2, axis=-1), dim, axis=2)
        relevant_t1 = t1 * relevant_degree
        result = K.sum(relevant_t1, axis=1)
        return result

    def final_pred(self, values):
        x1, x2 = values
        print("beta:", self.beta)
        x = x1 * self.beta + x2 * (1 - self.beta)
        return x

    def build_model(self):
        users_input = Input(
            shape=(
                self.wl,
                self.Lu),
            dtype='int32',
            name="users_Input")
        article_input = Input(
            shape=(
                self.La,
            ),
            dtype='int32',
            name="article_input")

        self.word_level_embedding = Embedding(input_dim=self.vocab_size,
                                              output_dim=self.embedding_dim,
                                              weights=[self.embedding_matrix],
                                              trainable=False,
                                              mask_zero=False,
                                              name="embedding")

        # word level
        Wu = Dense(self.H, activation="tanh", use_bias=True)
        Wd = Dense(self.H, activation="tanh", use_bias=True)
        Wr = Dense(self.H, activation="relu", use_bias=True)

        embed_article = self.word_level_embedding(article_input)
        embed_users = TimeDistributed(
            self.word_level_embedding)(users_input)

        h_users = TimeDistributed(Wu)(embed_users)
        h_article = Wd(embed_article)
        _encoded_article = AttLayer()(h_article)
        _encoded_users = TimeDistributed(AttLayer())(h_users)
        _encoded_articles = RepeatVector(self.wl)(_encoded_article)

        # user level
        encoded_users = Wr(concatenate([_encoded_users, _encoded_articles]))

        h_user_star = Lambda(lambda xin: xin[:, 0, :])(encoded_users)
        h_friends_star = Lambda(lambda xin: xin[:, 1:, :])(encoded_users)

        u2f_weight = TimeDistributed(Lambda(similarity))(concatenate([RepeatVector(self.wl-1)(h_user_star), h_friends_star], axis=-1))
        u2f_weight = Flatten()(u2f_weight)
        u2f_weight = Activation("softmax")(u2f_weight)
        h_u2f = Lambda(self.soft_attention)(
            [h_friends_star, u2f_weight])
        u_aggregate = Lambda(self.final_pred)([h_user_star, h_u2f])

        preds = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0))(u_aggregate)
        # preds = Dropout(0.2)(preds)
        # ========================== Matching End =============================

        # ======================== Model Building =============================
        # optimizer
        adam = Adam(
            lr=0.0005,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-06,
            decay=0.0,
            clipvalue=5.)

        self.model = Model(
            [users_input, article_input], preds)
        self.word_level_embedding.trainable = False
        self.model.summary()
        self.model.compile(
            optimizer=adam,
            loss='binary_crossentropy',
            metrics=['accuracy'])
        # ======================== Model Building End =========================
