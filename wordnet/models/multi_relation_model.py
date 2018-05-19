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
from models import multi_relation_unit_cube
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
        self.min_embed, self.delta_embed, self.rel_min_embed, self.rel_delta_embed = self.init_word_embedding(data)

        """get unit box representation for both term, no matter they are phrases or words"""
        if self.term:
            # if the terms are phrases, need to use either word average or lstm to compose the word embedding
            # Then transform them into unit cube.
            raw_t1_min_embed, raw_t1_delta_embed, raw_t2_min_embed, raw_t2_delta_embed = self.get_term_word_embedding(self.t1x, self.t1mask,
                                                                                                                        self.t1length, self.t2x,
                                                                                                                        self.t2mask, self.t2length,
                                                                                                                        False)
            self.t1_min_embed, self.t1_max_embed, self.t2_min_embed, self.t2_max_embed = self.transform_cube(raw_t1_min_embed, raw_t1_delta_embed,
                                                                                         raw_t2_min_embed, raw_t2_delta_embed)

        else:
            self.t1_min_embed, self.t1_max_embed = self.get_word_embedding(self.t1x, self.min_embed, self.delta_embed)
            self.t2_min_embed, self.t2_max_embed = self.get_word_embedding(self.t2x, self.min_embed, self.delta_embed)
            self.curr_rel_min_embed, self.curr_rel_max_embed = self.get_word_embedding(self.rel, self.rel_min_embed, self.rel_delta_embed)

            # self.curr_rel_min_embed = tf.Print(self.curr_rel_min_embed, [tf.exp(multi_relation_unit_cube.batch_log_prob(self.curr_rel_min_embed, self.curr_rel_max_embed))], 'relation size')
            # self.t1_min_embed = tf.Print(self.t1_min_embed, [tf.exp(multi_relation_unit_cube.batch_log_prob(self.t1_min_embed, self.t1_max_embed))], 't1 size')
            # self.t2_min_embed = tf.Print(self.t2_min_embed, [tf.exp(multi_relation_unit_cube.batch_log_prob(self.t2_min_embed, self.t2_max_embed))], 't2 size')
        """calculate box stats, we have three boxes now, we are interested in two overlaps."""
        # three way overlap
        self.join_min, self.join_max, self.meet_min, self.meet_max, self.disjoint = multi_relation_unit_cube.calc_join_and_meet(
            self.t1_min_embed, self.t1_max_embed, self.t2_min_embed, self.t2_max_embed, self.curr_rel_min_embed, self.curr_rel_max_embed)
        # two way overlap
        self.two_join_min, self.two_join_max, self.two_meet_min, self.two_meet_max, self.two_disjoint = unit_cube.calc_join_and_meet(
            self.t1_min_embed, self.t1_max_embed, self.curr_rel_min_embed, self.curr_rel_max_embed)

        """calculate -log(p(term2 | term1)) if overlap, surrogate function if not overlap"""
        # two surrogate function choice. lambda_batch_log_upper_bound or lambda_batch_disjoint_box
        if FLAGS.surrogate_bound:
            surrogate_func = multi_relation_unit_cube.lambda_batch_log_upper_bound
        else:
            surrogate_func = multi_relation_unit_cube.lambda_batch_disjoint_box

        train_pos_prob = tf_utils.slicing_where(condition=self.disjoint,
                                                full_input=tf.tuple([self.meet_min, self.meet_max,
                                                                     self.two_meet_min, self.two_meet_max,
                                                                     self.t1_min_embed, self.t1_max_embed,
                                                                     self.t2_min_embed, self.t2_max_embed,
                                                                     self.curr_rel_min_embed, self.curr_rel_max_embed]),
                                                true_branch=lambda x: 1.0 * surrogate_func(*x),
                                                false_branch=lambda x: multi_relation_unit_cube.lambda_batch_log_prob(*x))
        train_pos_prob = tf.Print(train_pos_prob, [tf.reduce_sum(tf.cast(tf.logical_and(
            self.disjoint, tf.logical_not(tf.cast(self.label, tf.bool))), tf.float32))],
                                  'neg disjoint value', summarize=3)
        train_pos_prob = tf.Print(train_pos_prob, [tf.reduce_sum(tf.cast(tf.logical_and(
            self.disjoint, tf.cast(self.label, tf.bool)), tf.float32))],
                                  'pos disjoint value', summarize=3)

        """calculate -log(1-p(term2 | term1)) if overlap, 0 if not overlap"""
        train_neg_prob = tf_utils.slicing_where(condition=self.disjoint,
                                                full_input=([self.meet_min, self.meet_max,
                                                             self.two_meet_min, self.two_meet_max,
                                                             self.t1_min_embed, self.t1_max_embed,
                                                             self.t2_min_embed, self.t2_max_embed,
                                                             self.curr_rel_min_embed, self.curr_rel_max_embed]),
                                                true_branch=lambda x: multi_relation_unit_cube.lambda_zero(*x),
                                                false_branch=lambda x: multi_relation_unit_cube.lambda_batch_log_1minus_prob(*x))
        """calculate negative log prob when evaluating pairs. The lower, the better"""
        # when return hierarchical error, we return the negative log probability, the lower, the probability higher
        # if two things are disjoint, we return -tf.log(1e-8).
        self.eval_prob = tf_utils.slicing_where(condition = self.disjoint,
                                                full_input = [self.join_min, self.join_max, self.meet_min, self.meet_max,
                                                              self.t1_min_embed, self.t1_max_embed,
                                                              self.t2_min_embed, self.t2_max_embed],
                                                true_branch = lambda x: unit_cube.lambda_hierarchical_error_upper(*x),
                                                false_branch = lambda x: unit_cube.lambda_batch_log_prob(*x))
        """model marg prob loss"""
        if FLAGS.w2 > 0.0:
            self.marg_prob = tf.constant(data.margina_prob)
            kl_difference = unit_cube.calc_marginal_prob(self.marg_prob, self.min_embed, self.delta_embed)
            kl_difference = tf.reshape(kl_difference, [-1]) / self.vocab_size
            self.marg_loss = FLAGS.w2 * (tf.reduce_sum(kl_difference))
        else:
            self.marg_loss = 0.0

        """model cond prob loss"""
        self.pos = FLAGS.w1 * tf.multiply(train_pos_prob, self.label)
        self.neg = FLAGS.w1 * tf.multiply(train_neg_prob, (1 - self.label))
        # print out analysis value
        self.pos_disjoint = tf.logical_and(tf.cast(self.label, tf.bool), self.disjoint)
        self.pos_overlap = tf.logical_and(tf.cast(self.label, tf.bool), tf.logical_not(self.disjoint))
        self.neg_disjoint = tf.logical_and(tf.logical_not(tf.cast(self.label, tf.bool)), self.disjoint)
        self.neg_overlap = tf.logical_and(tf.logical_not(tf.cast(self.label, tf.bool)), tf.logical_not(self.disjoint))
        self.pos_disjoint.set_shape([None])
        self.neg_disjoint.set_shape([None])
        self.pos_overlap.set_shape([None])
        self.neg_overlap.set_shape([None])
        self.pos = tf.Print(self.pos, [tf.reduce_mean(tf.boolean_mask(self.pos, self.pos_disjoint))], 'pos disjoint loss')
        self.pos = tf.Print(self.pos, [tf.reduce_mean(tf.boolean_mask(self.pos, self.pos_overlap))], 'pos overlap loss')
        self.neg = tf.Print(self.neg, [tf.reduce_mean(tf.boolean_mask(self.neg, self.neg_disjoint))], 'neg disjoint loss')
        self.neg = tf.Print(self.neg, [tf.reduce_mean(tf.boolean_mask(self.neg, self.neg_overlap))], 'neg overlap loss')
        # # print out analysis value
        self.cond_loss = tf.reduce_sum(self.pos) / (self.batch_size / 2) + \
                         tf.reduce_sum(self.neg) / (self.batch_size / 2)
        self.cond_loss = tf.Print(self.cond_loss, [tf.reduce_sum(train_pos_prob), tf.reduce_sum(self.neg)], 'check where nan comes from')

        """model regurlization: make box to be poe-ish"""
        # self.regularization = -FLAGS.r1 * (tf.reduce_sum(self.min_embed) + tf.reduce_sum(self.delta_embed))/ self.vocab_size
        self.regularization = -FLAGS.r1 * (tf.nn.l2_loss(self.rel_min_embed) + tf.nn.l2_loss(self.rel_delta_embed)) / self.rel_size
        # self.regularization = tf.Print(self.regularization, [self.regularization], 'regu loss')
        tf.summary.scalar("regularization loss", self.regularization)
        # tf.summary.scalar("pos disjoint loss", tf.reduce_mean(tf.boolean_mask(self.pos, self.pos_disjoint)))
        # tf.summary.scalar("pos overlap loss", tf.reduce_mean(tf.boolean_mask(self.pos, self.pos_overlap)))
        # tf.summary.scalar("neg disjoint loss", tf.reduce_mean(tf.boolean_mask(self.neg, self.neg_disjoint)))
        # tf.summary.scalar("neg overlap loss", tf.reduce_mean(tf.boolean_mask(self.neg, self.neg_overlap)))

        """model final loss"""
        self.loss = self.cond_loss + self.marg_loss + self.regularization



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
                tf.random_uniform([self.vocab_size, self.embed_dim], min_lower_scale, min_higher_scale, seed=my_seed),
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

        # rel_embed = tf.Variable(tf.random_uniform([self.rel_size, self.embed_dim], 0.1, 1.0, seed=my_seed),
        #                         trainable=True, name='rel_embed')
        rel_min_embed = tf.Variable(
            tf.random_uniform([self.rel_size, self.embed_dim], min_lower_scale, min_higher_scale, seed=my_seed),
            trainable=True, name='rel_min_embed')
        rel_delta_embed = tf.Variable(
            tf.random_uniform([self.rel_size, self.embed_dim], delta_lower_scale, delta_higher_scale, seed=my_seed),
            trainable=True, name='rel_delta_embed')
        return min_embed, delta_embed, rel_min_embed, rel_delta_embed

    def get_term_word_embedding(self, t1x, t1mask, t1length, t2x, t2mask, t2length, reuse):

        """

        Args:
            t1x, t1mask, t1length: entity one stats.
            t2x, t2mask, t2length: entity two stats.
            reuse: whether to reuse lstm parameters. Differs in training or eval

        Returns: word embedding for entity one phrase and entity two phrase.

        """
        if self.tuple_model == 'ave':
            t1_min_embed = tf_utils.tuple_embedding(t1x, t1mask, t1length, self.min_embed)
            t2_min_embed = tf_utils.tuple_embedding(t2x, t2mask, t2length, self.min_embed)
            t1_delta_embed = tf_utils.tuple_embedding(t1x, t1mask, t1length, self.delta_embed)
            t2_delta_embed = tf_utils.tuple_embedding(t2x, t2mask, t2length, self.delta_embed)

        elif self.tuple_model == 'lstm':
            term_rnn = tf.contrib.rnn.LSTMCell(self.hidden_dim, use_peepholes=self.peephole,num_proj=self.embed_dim, state_is_tuple=True)
            if reuse:
                with tf.variable_scope('term_embed', reuse=True):
                    t1_min_embed = tf_utils.tuple_lstm_embedding(t1x, t1mask, t1length, self.min_embed, term_rnn, False)
            else:
                with tf.variable_scope('term_embed'):
                    t1_min_embed = tf_utils.tuple_lstm_embedding(t1x, t1mask, t1length, self.min_embed, term_rnn, False)
            with tf.variable_scope('term_embed', reuse=True):
                t1_delta_embed = tf_utils.tuple_lstm_embedding(t1x, t1mask, t1length, self.delta_embed, term_rnn, True)
            with tf.variable_scope('term_embed', reuse=True):
                t2_min_embed = tf_utils.tuple_lstm_embedding(t2x, t2mask, t2length, self.min_embed, term_rnn, True)
            with tf.variable_scope('term_embed', reuse=True):
                t2_delta_embed = tf_utils.tuple_lstm_embedding(t2x, t2mask, t2length, self.delta_embed, term_rnn, True)
        else:
            raise ValueError("Expected either ave or lstm but received", self.tuple_model)

        return t1_min_embed, t1_delta_embed, t2_min_embed, t2_delta_embed

    def transform_cube(self, t1_min_embed, t1_delta_embed, t2_min_embed, t2_delta_embed):
        if self.cube == 'sigmoid':
            t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed = tf_utils.make_sigmoid_cube(t1_min_embed,
                                                                                                t1_delta_embed,
                                                                                                t2_min_embed,
                                                                                                t2_delta_embed)
        elif self.cube == 'softmax':
            t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed = tf_utils.make_softmax_cube(t1_min_embed,
                                                                                                t1_delta_embed,
                                                                                                t2_min_embed,
                                                                                                t2_delta_embed)
        else:
            raise ValueError("Expected either sigmoid or softmax but received", self.cube)
        return t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed

    def get_word_embedding(self, t1_idx, min_we, delta_we):
        """read word embedding from embedding table, get unit cube embeddings"""

        t1_min_embed = tf.squeeze(tf.nn.embedding_lookup(min_we, t1_idx))
        t1_delta_embed = tf.squeeze(tf.nn.embedding_lookup(delta_we, t1_idx))

        t1_max_embed = t1_min_embed + t1_delta_embed
        return t1_min_embed, t1_max_embed


    def generate_neg(self, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed):
        # randomly generate negative examples by swaping to the next examples
        nt1_min_embed = tf.nn.embedding_lookup(t1_min_embed,(self.rang+1)%self.batch_size)
        nt2_min_embed = tf.nn.embedding_lookup(t2_min_embed,(self.rang+2)%self.batch_size)
        nt1_max_embed = tf.nn.embedding_lookup(t1_max_embed,(self.rang+1)%self.batch_size)
        nt2_max_embed = tf.nn.embedding_lookup(t2_max_embed,(self.rang+2)%self.batch_size)

        return nt1_min_embed, nt1_max_embed, nt2_min_embed, nt2_max_embed

    def get_grad(self):
        self.pos_grad = tf.gradients(self.pos, [self.min_embed, self.delta_embed])
        self.neg_grad = tf.gradients(self.neg, [self.min_embed])
        self.neg_delta_grad = tf.gradients(self.neg, [self.delta_embed])
        self.kl_grad = tf.gradients(self.marg_loss, [self.min_embed, self.delta_embed])


    def training(self, loss, epsilon, learning_rate):
        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate, epsilon)
        elif self.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise ValueError('expected adam or sgd, got', self.optimizer)
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # whether to freeze the negative gradient
        if self.freeze_grad:
            self.get_grad()
            train_op = optimizer.apply_gradients(zip(self.pos_grad, [self.min_embed, self.delta_embed]))
            with tf.control_dependencies([train_op]):
                train_op_neg = optimizer.apply_gradients(zip(self.neg_grad, [self.min_embed]))
                kl_train_op = optimizer.apply_gradients(zip(self.kl_grad, [self.min_embed, self.delta_embed]))
                train_op=tf.group(train_op, train_op_neg, kl_train_op)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)

        with tf.control_dependencies([train_op]):
            if self.measure == 'exp':
                clipped_we,clipped_delta=unit_cube.exp_clip_embedding(self.min_embed, self.delta_embed)
            elif self.measure == 'uniform':
                clipped_we,clipped_delta=unit_cube.uniform_clip_embedding(self.min_embed, self.delta_embed)
                clipped_min_rel, clipped_delta_rel = unit_cube.uniform_clip_embedding(self.rel_min_embed, self.rel_delta_embed)
            else:
                raise ValueError('Expected exp or uniform, but got', self.measure)
            project=tf.group(tf.assign(self.min_embed,clipped_we),tf.assign(self.delta_embed,clipped_delta))
            project1 = tf.group(tf.assign(self.rel_min_embed, clipped_min_rel), tf.assign(self.rel_delta_embed, clipped_delta_rel))
            train_op=tf.group(train_op,project)
            train_op = tf.group(train_op,project1)
        return train_op


    def rel_embedding(self, Rel, rel, relmsk):
        embed_rel = tf.nn.embedding_lookup(Rel, rel)
        embed_rel = embed_rel * relmsk
        return embed_rel
