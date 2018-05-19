"""Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils import tf_utils
from models import unit_cube
import tensorflow as tf

my_seed = 20180112
tf.set_random_seed(my_seed)

class tf_model(object):
    def __init__(self, data, placeholder, FLAGS):
        self.optimizer = FLAGS.optimizer
        self.opti_epsilon = FLAGS.epsilon
        self.lr = FLAGS.learning_rate
        self.vocab_size = data.vocab_size
        self.measure = FLAGS.measure
        self.embed_dim = FLAGS.embed_dim
        self.batch_size = FLAGS.batch_size
        self.rel_size = FLAGS.rel_size
        self.tuple_model = FLAGS.tuple_model
        self.init_embedding = FLAGS.init_embedding
        self.rang=tf.range(0,FLAGS.batch_size,1)
        # LSTM Params
        self.term = FLAGS.term
        self.hidden_dim = FLAGS.hidden_dim
        self.peephole = FLAGS.peephole
        self.freeze_grad = FLAGS.freeze_grad

        self.t1x = placeholder['t1_idx_placeholder']
        self.t1mask = placeholder['t1_msk_placeholder']
        self.t1length = placeholder['t1_length_placeholder']
        self.t2x = placeholder['t2_idx_placeholder']
        self.t2mask = placeholder['t2_msk_placeholder']
        self.t2length = placeholder['t2_length_placeholder']
        self.rel = placeholder['rel_placeholder']
        self.relmsk = placeholder['rel_msk_placeholder']
        self.label = placeholder['label_placeholder']

        """Initiate box embeddings"""
        self.embed, _, self.rel_embed = self.init_word_embedding(data)

        self.t1_embed = tf.squeeze(tf.nn.embedding_lookup(self.embed, self.t1x), [1])
        self.t2_embed = tf.squeeze(tf.nn.embedding_lookup(self.embed, self.t2x), [1])
        self.r_embed = tf.nn.embedding_lookup(self.rel_embed, self.rel)

        train_pos_prob = self.error_func()
        train_neg_prob = self.error_func()

        self.eval_prob = self.error_func()

        """model marg prob loss"""
        self.marg_loss = 0.0

        """model cond prob loss"""
        self.pos = FLAGS.w1 * tf.multiply(train_pos_prob, self.label)
        self.neg = FLAGS.w1 * tf.multiply(train_neg_prob, (1 - self.label))
        self.pos = tf.Print(self.pos, [self.pos], 'pos')
        self.neg = tf.Print(self.neg, [self.neg], 'neg')
        self.cond_loss = tf.maximum(0.0, (100.0+tf.reduce_sum(self.pos) / (self.batch_size / 2) - tf.reduce_sum(self.neg) / (self.batch_size / 2)))

        """model final loss"""
        self.loss = self.cond_loss + self.marg_loss

    def error_func(self):
        result = tf.reduce_sum(tf.abs(self.t2_embed - (self.t1_embed + self.r_embed)), axis = 1)
        return result


    @property
    def init_embedding_scale(self):
        """For different measures, min and delta have different init value. """
        if self.measure == 'exp' and not self.term:
            min_lower_scale, min_higher_scale = 0.0, 0.001
            delta_lower_scale, delta_higher_scale = 10.0, 10.5
        elif self.measure == 'uniform' and not self.term:
            min_lower_scale, min_higher_scale = 1e-4, 1e-2
            delta_lower_scale, delta_higher_scale = 0.9, 0.999
        elif self.term and self.measure == 'uniform':
            min_lower_scale, min_higher_scale = 1.0, 1.1
            delta_lower_scale, delta_higher_scale = 5.0, 5.1
        else:
            raise ValueError("Expected either exp or uniform but received", self.measure)
        return min_lower_scale, min_higher_scale, delta_lower_scale, delta_higher_scale

    def init_word_embedding(self, data):
        min_lower_scale, min_higher_scale, delta_lower_scale, delta_higher_scale = self.init_embedding_scale
        if self.init_embedding == 'random':
            # random init word embedding
            min_embed = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embed_dim], 0.1 , 1.0, seed=my_seed),
                trainable=True, name='word_embed')
            delta_embed = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embed_dim], delta_lower_scale, delta_higher_scale,
                                  seed=my_seed), trainable=True, name='delta_embed')

        elif self.init_embedding == 'pre_train':
            # init min/delta word embedding with pre trained prob order embedding
            min_embed = tf.Variable(data.min_embed, trainable=True, name='word_embed')
            delta_embed = tf.Variable(data.delta_embed, trainable=True, name='delta_embed')

        else:
            raise ValueError("Expected either random or pre_train but received", self.init_embedding)

        rel_embed = tf.Variable(tf.random_uniform([self.rel_size, self.embed_dim], 0.1, 1.0, seed=my_seed),
                                trainable=True, name='rel_embed')

        return min_embed, delta_embed, rel_embed


    def training(self, loss, epsilon, learning_rate):
        tf.summary.scalar(loss.op.name, loss)
        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate, epsilon)
        elif self.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise ValueError('expected adam or sgd, got', self.optimizer)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op


    def rel_embedding(self, Rel, rel, relmsk):
        embed_rel = tf.nn.embedding_lookup(Rel, rel)
        embed_rel = embed_rel * relmsk
        return embed_rel
