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

import tensorflow as tf
from utils import tf_utils

flags = tf.app.flags
FLAGS = flags.FLAGS


def calc_join_and_meet(t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed, rel_min_embed, rel_max_embed):
    """
    # three box embeddings are t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed, rel_min_embed, rel_max_embed
    Returns:
        join box, min box, and disjoint condition:
    """
    # join is min value of lower bound all boxes, max value of upper bound all boxes
    join_min = tf.minimum(tf.minimum(t1_min_embed, t2_min_embed), rel_min_embed)
    join_max = tf.maximum(tf.maximum(t1_max_embed, t2_max_embed), rel_max_embed)
    # find meet is calculate the max value of lower bound all boxes, min value of lower bound of boxes
    meet_min = tf.maximum(tf.maximum(t1_min_embed, t2_min_embed), rel_min_embed) # batchsize * embed_size
    meet_max = tf.minimum(tf.minimum(t1_max_embed, t2_max_embed), rel_max_embed) # batchsize * embed_size
    # The overlap cube's max value have to be bigger than min value in every dimension to form a valid cube
    # if it's not, then two concepts are disjoint, return none
    cond = tf.cast(tf.less_equal(meet_max, meet_min), tf.float32)  # batchsize * embed_size
    cond = tf.cast(tf.reduce_sum(cond, axis=1), tf.bool)  # batchsize. If disjoint, cond > 0; else, cond = 0
    return join_min, join_max, meet_min, meet_max, cond


"""-log(p(term2 | term1, relation)), when overlap"""

def lambda_batch_log_prob(nume_min, nume_max, domi_min, domi_max, a, b, c, d, e, f):
    # this function return the negative conditional log probability of positive examplse if they have overlap.
    # we want to minimize the return value -log(p(a|b))
    joint_log = batch_log_prob(nume_min, nume_max)
    domi_log = batch_log_prob(domi_min, domi_max)  # batch_size
    cond_log = joint_log - domi_log  # (batch_size)
    smooth_log_prob = smooth_prob(cond_log)
    neg_smooth_log_prob = -smooth_log_prob
    return neg_smooth_log_prob


"""When disjoint, but we still want to calculate log(p). Minimize the disjoint box."""


def lambda_batch_disjoint_box(nume_min, nume_max, domi_min, domi_max, a, b, c, d, e, f):
    # return log of disjoint dimension errors
    # <a, b> is the min and max embedding of specific term, <c, d> is the min and max embedding of general term
    # <e, f> is the min and max embedding of the relation term.
    cond = tf.less_equal(nume_max, nume_min)  # batchsize * embed_size
    # choose all those dimensions with true condtions
    meet_min_cond = tf.where(cond, nume_min, tf.ones_like(nume_min)) #batchsize * embed_size
    meet_max_cond = tf.where(cond, nume_max, tf.zeros_like(nume_min)) #batchsize * embed_size
    neg_distance = meet_min_cond - meet_max_cond
    disjoint_box_log = batch_log_prob(meet_max_cond, meet_min_cond)
    neg_smooth_log_prob = -smooth_prob(disjoint_box_log)
    # because input to log1mexp is positive
    cond = tf.squeeze(cond)
    cond.set_shape([None])
    nume_min = tf.squeeze(nume_min)
    nume_max = tf.squeeze(nume_max)
    onemp_neg_smooth_log_prob = -tf_utils.log1mexp(neg_smooth_log_prob)
    # onemp_neg_smooth_log_prob = tf.Print(onemp_neg_smooth_log_prob, [neg_smooth_log_prob, onemp_neg_smooth_log_prob, tf.boolean_mask(tf.squeeze(nume_min), cond),  tf.boolean_mask(tf.squeeze(nume_max), cond)], 'debug')

    return onemp_neg_smooth_log_prob


"""-log(1-p(term2 | term1)), when overlap"""


def lambda_batch_log_1minus_prob(nume_min, nume_max, domi_min, domi_max, a, b, c, d, e, f):
    # this function return the -log(1-p(a, b))
    # we want to minimize this value
    joint_log = batch_log_prob(nume_min, nume_max)
    domi_log = batch_log_prob(domi_min, domi_max)  # batch_size
    cond_log = joint_log - domi_log  # (batch_size)
    neg_smooth_log_prob = -smooth_prob(cond_log)
    # because input to log1mexp is positive
    onemp_neg_smooth_log_prob = -tf_utils.log1mexp(neg_smooth_log_prob)
    return onemp_neg_smooth_log_prob


"""return zeros vectores"""


def lambda_zero(nume_min, nume_max, domi_min, domi_max, a, b, c, d, e, f):
    # this function return 0 because in this case, two negative examples are already disjoint to each other
    result = tf.zeros_like(tf.reduce_sum(nume_min, axis=1))
    return result


"""for evaluation time disjoint log probability"""


def lambda_hierarchical_error_upper(join_min, join_max, meet_min, meet_max, a, b, c, d):
    return tf.fill([tf.shape(meet_min)[0]], -tf.log(1e-30))


"""for general calculation."""


def batch_log_prob(min_embed, max_embed):
    # min_embed: batchsize * embed_size
    # max_embed: batchsize * embed_size
    # log_prob: batch_size
    # numerically stable log probability of a cube probability
    if FLAGS.measure == 'uniform':
        if FLAGS.model == 'poe':
            log_prob = tf.reduce_sum(tf.log(tf.ones_like(max_embed) - min_embed + 1e-8), axis=1)
        elif FLAGS.model == 'cube':
            log_prob = tf.reduce_sum(tf.log((max_embed - min_embed) + 1e-8), axis=1)
        else:
            raise ValueError('Expected poe or cube, but receive', FLAGS.model)
    elif FLAGS.measure == 'exp':
        log_prob = tf.reduce_sum(-min_embed + tf_utils.log1mexp(max_embed - min_embed), axis=1)
    else:
        raise ValueError('Expected uniform or exp, but receive', FLAGS.measure)
    return log_prob


def batch_log_upper_bound(join_min, join_max, a, b, c, d):
    # join_min: batchsize * embed_size
    # join_max: batchsize * embed_size
    # log_prob: batch_size
    join_log_prob = batch_log_prob(join_min, join_max)
    join_log_prob_new = tf.reduce_logsumexp(
        tf.stack([tf.fill([tf.shape(join_log_prob)[0]], tf.log(0.1)), join_log_prob], axis=1), axis=1)
    x_log_prob = batch_log_prob(a, b)  # batchsize
    y_log_prob = batch_log_prob(c, d)  # batchsize
    log_xy = tf.reduce_logsumexp(tf.stack([x_log_prob, y_log_prob], axis=1), axis=1)
    log_upper_bound = join_log_prob_new + tf_utils.log1mexp(join_log_prob_new - log_xy)
    return log_upper_bound


def smooth_prob(input_prob):
    pos_prob1 = tf.log(1 - FLAGS.lambda_value) + input_prob  # (batch_size)
    pos_prob2 = tf.stack([pos_prob1, tf.zeros_like(input_prob) + tf.log(FLAGS.lambda_value) + tf.log(0.5)],
                         axis=1)  # (batch_size, 2)
    pos_prob = tf.reduce_logsumexp(pos_prob2, axis=1)  # (batch_size)
    return pos_prob

def calc_marginal_prob(golden_prob, min_embed, delta_embed):
    pred_prob = batch_log_prob(min_embed, min_embed + delta_embed)
    smoothed_pred_prob = smooth_prob(pred_prob)
    kl_difference, _ = tf_utils.bernoulli_kl_xent(smoothed_pred_prob, golden_prob)
    return kl_difference


"""in order to force everything in unit cube, need to apply constrains..."""
def uniform_clip_embedding(min_embed, delta_embed):
    # min_embed: batchsize * embed_size
    # delta_embed: batchsize * embed_size
    if FLAGS.model == 'cube':
        new_min_embed = tf.minimum((1.0-FLAGS.cube_eps), tf.maximum(0.0, min_embed))
        new_delta_embed = tf.minimum((1.0-new_min_embed), tf.maximum(FLAGS.cube_eps, delta_embed))
    elif FLAGS.model == 'poe':
        new_min_embed = tf.minimum(1.0, tf.maximum(0.0, min_embed))
        new_delta_embed = 1.0 - new_min_embed
    else:
        raise ValueError('expected cube or poe, but got', FLAGS.model)
    return new_min_embed, new_delta_embed


def exp_clip_embedding(min_embed, delta_embed):
    # min_embed: batchsize * embed_size
    # delta_embed: batchsize * embed_size
    if FLAGS.model == 'cube':
        new_min_embed = tf.nn.softplus(min_embed)
        new_delta_embed = tf.nn.softplus(delta_embed)
    elif FLAGS.model == 'poe':
        new_min_embed = tf.nn.softplus(min_embed)
        new_delta_embed = 200.00-tf.nn.softplus(min_embed)
    return new_min_embed, new_delta_embed




    #
    #   def reverse_hierarchical_error(self, t2x, t1x):
    #     t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed = self.get_word_embedding(t1x, t2x)
    #     _, _, meet_min, meet_max, not_have_meet = self.calc_join_and_meet(t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed)
    #     # when return hierarchical error, we return the negative log probability, the lower, the probability higher
    #     # if two things are disjoint, we return -tf.log(1e-8).
    #     # it's just evaluation, so it should be fine to use small number to represent for lower probability
    #     pos_prob = tf_utils.slicing_where(condition = not_have_meet,
    #         full_input = tf.tuple([t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed, meet_min, meet_max]),
    #         true_branch = lambda x: self.lambda_hierarchical_error_upper(*x),
    #         false_branch = lambda x: self.lambda_hierarchical_error_prob(*x))
    #     # self.not_have_meet = not_have_meet
    #     self.pos_prob = pos_prob
    #     return pos_prob

