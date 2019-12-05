from __future__ import print_function
import numpy as np
import tensorflow as tf
import random as rn
import os

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
tf.set_random_seed(42)
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)
from keras.backend import tensorflow_backend as K
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 1
K.set_session(tf.Session(config=config))

from keras.layers import Input, Dense, Lambda,  Conv1D, Dropout, LeakyReLU, Reshape
from keras.layers import dot, multiply, concatenate
from keras.models import Model, Sequential
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.optimizers import Adam
from keras.layers import Embedding
from keras.layers.recurrent import GRU


class shan():
    def __init__(
            self,
            vocab_size=100000,
            max_num_sent=30,
            max_len_sent=100,
            embedding_dim=200,
            max_user=7242,
            hidden_size=128,
            walk_length=21,
            dropout=0.2,
            num_kernels=7,
            filter_size=50,
            embedding_matrix=None):
        self.vocab_size = vocab_size
        self.max_num_sent = max_num_sent
        self.max_len_sent = max_len_sent
        self.embedding_dim = embedding_dim
        self.max_user = max_user
        self.H = hidden_size
        self.walk_length = walk_length
        self.dropout = dropout
        self.embedding_matrix = embedding_matrix
        self.num_kernels = num_kernels
        self.filter_size = filter_size
        self.use_social = use_social

    def bi_gru_encode(self, dim):
        seq = Sequential()
        seq.add(Bidirectional(
            GRU(
                units=self.H,
                return_sequences=True,
                recurrent_dropout=self.dropout,
                dropout=self.dropout,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='glorot_uniform',
                bias_initializer="zeros"
            ),
            merge_mode='concat',
            input_shape=(1, dim))
        )
        return seq

    def social_attention(self, x):
        self.user_embedding = Embedding(
            input_dim=self.max_user + 1,
            output_dim=self.H,
            trainable=True,
            mask_zero=True)
        all_u_i = self.user_embedding(x)

        w_v = Dense(1, use_bias=False)
        w_x = Dense(self.H, use_bias=False)

        u_i = Lambda(lambda xin: xin[:, 0, :])(all_u_i)
        
        e = []
        w_all_u_i = []
        u_i = w_x(u_i)
        for j in range(self.walk_length):
            u_j = Lambda(lambda xin: xin[:, j, :])(all_u_i)
            u_j = w_x(u_j)
            e_i_j = LeakyReLU(alpha=0.3)(w_v(concatenate([u_i, u_j])))
            e.append(e_i_j)
            w_all_u_i.append(u_j)
        e = concatenate(e, axis=-1)
        w_all_u_i = Reshape((self.walk_length, self.H))(concatenate(w_all_u_i, axis=-1))
        alpha = Lambda(lambda xin: K.repeat_elements(K.expand_dims(
            K.softmax(xin), -1), rep=self.H, axis=-1))(e)
        u_f_i = Lambda(lambda xin: K.sum(xin, axis=1))(multiply([alpha, w_all_u_i]))
        return u_f_i

    def multi_sequence_encode(self, input_dim, t1, t2):
        kernels = [i for i in range(1, self.num_kernels+1)]
        filters = [self.filter_size for _ in range(self.num_kernels)]
        multi_cnn_encoded = []
        for k in range(len(kernels)):
            s_i = []
            for i in range(self.max_num_sent):
                conv1d = Sequential()
                conv1d.add(
                    Conv1D(
                        filters=filters[k],
                        kernel_size=kernels[k],
                        input_shape=(None, input_dim),
                        activation='relu',
                        padding='valid'
                    ))
                conv1d.add(Dropout(self.dropout))
                h_i_t = Lambda(lambda xin: xin[:, i, :])(t1)
                h_i_t = conv1d(h_i_t)
                s_i_t = Lambda(lambda xin: K.expand_dims(xin, axis=1))(self.personalized_attention(t2, h_i_t))
                s_i.append(s_i_t)
            s_i = concatenate(s_i, axis=1)
            print("s_i:", s_i)
            multi_cnn_encoded.append(s_i)
        t3 = concatenate(multi_cnn_encoded, axis=-1)
        return t3

    def personalized_attention(self, u_s, h_i):
        dim = h_i.shape[-1]
        u_i = Dense(self.H, activation="tanh", use_bias=True)(h_i)
        dot_i_s = dot([u_i, u_s], axes=(2, 1))
        alpha = Lambda(lambda xin: K.repeat_elements(K.expand_dims(
            K.softmax(xin), -1), rep=dim, axis=-1))(dot_i_s)
        s = Lambda(lambda xin: K.sum(xin, axis=1))(multiply([alpha, h_i]))
        return s

    def build_model(self):
        # ================================ Input ==============================
        user_input = Input(
            shape=(
                self.walk_length,
            ),
            dtype='float32',
            name="friends_input")
        article_input = Input(
            shape=(
                self.max_num_sent,
                self.max_len_sent
            ),
            dtype='float32',
            name="article_input")

        # ======================== GRU-based sequence encoder =================
        word_embedding = Embedding(input_dim=self.vocab_size,
                                   output_dim=self.embedding_dim,
                                   weights=[self.embedding_matrix],
                                   mask_zero=False,
                                   name="word_embedding")
        embed_article = TimeDistributed(word_embedding)(article_input)

        # ============================== Word Level Attention =================
        u_w = self.social_attention(user_input)
        s_i = self.multi_sequence_encode(self.embedding_dim, embed_article, u_w)
        print("encoded_article:", s_i._keras_shape)
        # ============================== Sentence Level Attention =============
        s_i_dim = s_i._keras_shape[-1]
        s_encoding = self.bi_gru_encode(s_i_dim)
        h_i = s_encoding(s_i)
        u_s = self.social_attention(user_input)
        v = self.personalized_attention(u_s, h_i)

        # ============================ Prediction =============================

        preds = Dense(1, activation="sigmoid")(v)

        adam = Adam(
            lr=0.0005,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-06,
            decay=0.0,
            clipvalue=5.)
        self.model = Model(
            [user_input, article_input], preds)
        word_embedding.trainable = False
        self.model.compile(
            optimizer=adam,
            loss='binary_crossentropy',
            metrics=['accuracy'])

