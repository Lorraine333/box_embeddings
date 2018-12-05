import tensorflow as tf
import numpy as np
import math
from util.Layer import Layers
Layer = Layers()
from tensorflow.python.util import nest
tf.set_random_seed(20160408)

lambda_value = 1e-100
# log probability
def probability(min_embed, max_embed):
    # min_embed: batchsize * embed_size
    # max_embed: batchsize * embed_size
    # log_prob: batch_size
    # numerically stable log probability of a uniform hypercube measure:
    # log_prob = tf.reduce_sum(tf.log((max_embed - min_embed) + 1e-8) ,axis = 1)
    log_prob = tf.reduce_sum(-min_embed + log1mexp(max_embed - min_embed), axis = 1)
    log_prob = tf.clip_by_value(log_prob, -1000, -0.0001)
    return log_prob

def test_probability(min_embed, max_embed):
    # min_embed: batchsize * embed_size
    # max_embed: batchsize * embed_size
    # log_prob: batch_size
    # numerically stable log probability of a uniform hypercube measure:
    # log_prob = tf.reduce_sum(tf.log((max_embed - min_embed) + 1e-8) ,axis = 1)
    log_prob = tf.reduce_sum(-min_embed + log1mexp(max_embed - min_embed), axis = 1)
    # log_prob = tf.clip_by_value(log_prob, -1000, -0.0001)
    return log_prob


def batch_log_upper_bound(join_min, join_max, a, b, c, d):
    # join_min: batchsize * embed_size
    # join_max: batchsize * embed_size
    # log_prob: batch_size
    join_log_prob = probability(join_min, join_max)
    join_log_prob_new = tf.reduce_logsumexp(tf.stack([tf.fill([tf.shape(join_log_prob)[0]], tf.log(0.1)), join_log_prob], axis = 1), axis = 1)
    x_log_prob = probability(a, b) # batchsize
    y_log_prob = probability(c, d) # batchsize
    log_xy = tf.reduce_logsumexp(tf.stack([x_log_prob, y_log_prob], axis = 1), axis = 1)
    log_upper_bound = join_log_prob_new + log1mexp(join_log_prob_new - log_xy)
    return log_upper_bound


def calc_join_and_meet(t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed):
    # two word embeddings are a, b, c, d
    # join is min value of (a, c), max value of (b, d)
    join_min = tf.minimum(t1_min_embed, t2_min_embed)
    join_max = tf.maximum(t1_max_embed, t2_max_embed)
    # find meet is calculate the max value of (a,c), min value of (b,d)
    meet_min = tf.maximum(t1_min_embed, t2_min_embed) #batchsize * embed_size
    meet_max = tf.minimum(t1_max_embed, t2_max_embed) #batchsize * embed_size
    # The overlap cube's max value have to be bigger than min value in every dimension to form a valid cube
    # if it's not, then two concepts are disjoint, return none
    cond =  tf.cast(tf.less_equal(meet_max, meet_min), tf.float32) # batchsize * embed_size
    # cond = tf.reduce_sum(cond, axis = 1)
    cond = tf.cast(tf.reduce_sum(cond, axis = 1), tf.bool) # batchsize. If disjoint, cond > 0; else, cond = 0
    return join_min, join_max, meet_min, meet_max, cond

"""for positive examples"""
# this is for positive examples, slicing where function
def lambda_batch_log_upper_bound(join_min, join_max, meet_min, meet_max, a, b, c, d):
    # this function return the upper bound log(p(a join b) + log(0.01) - p(a) - p(b)) of positive examplse if they are disjoint
    # minus the log probability of the condionaled term
    # we want to minimize the return value too
    joint_log = batch_log_upper_bound(join_min, join_max, a, b, c, d)
    # domi_log = probability(a, b) # batch_size
    domi_log = probability(c, d)
    cond_log = joint_log - domi_log # (batch_size)
    return -cond_log

def lambda_batch_log_upper_bound_version2(join_min, join_max, meet_min, meet_max, a, b, c, d):
    # return log of disjoint dimension errors
    # <a, b> is the min and max embedding of specific term, <c, d> is the min and max embedding of general term
    single_dim = tf.maximum(0.0, 1e-8 + tf.maximum(a-d, c-b))
    result = tf.reduce_sum(tf.log(1.0 + single_dim), axis = 1) # sum along embedding dim, batch_size
    return result

def lambda_batch_log_cube_measure(join_min, join_max, meet_min, meet_max, a, b, c, d):
    # this function return the negative conditional log probability of positive examplse if they have overlap.
    # we want to minimize the return value -log(p(a|b))
    joint_log = probability(meet_min, meet_max)
    # domi_log = probability(a, b) # batch_size
    domi_log = probability(c, d) # batch_size
    cond_log = tf.subtract(joint_log, domi_log )# (batch_size)
    smooth_log_prob = smooth_prob(cond_log)
    cliped_smooth_log_prob = tf.clip_by_value(smooth_log_prob, np.log(1e-8), np.log(1.0))
    return cliped_smooth_log_prob
    # return cond_log

"""for negative examples"""
def lambda_batch_log_cond_cube_measure(join_min, join_max, meet_min, meet_max, a, b, c, d):
    # this function return the log(1-p(a, b))
    # we want to minimize this value
    neg_smooth_log_prob = -lambda_batch_log_cube_measure(join_min, join_max, meet_min, meet_max, a, b, c, d)
    # because input to log1mexp is positive
    cliped_neg_smooth_log_prob = tf.clip_by_value(neg_smooth_log_prob, 1e-10, neg_smooth_log_prob)
    onemp_smooth_log_prob = log1mexp(cliped_neg_smooth_log_prob)
    return onemp_smooth_log_prob

def lambda_zero_log_upper_bound(join_min, join_max, meet_min, meet_max, a, b, c, d):
    # this function return 0 because in this case, two negative examples are already disjoint to each other
    result = tf.zeros_like(tf.reduce_sum(join_min, axis = 1))
    return result

"""for test examples"""

def lambda_batch_test_joint_cube_measure(join_min, join_max, meet_min, meet_max, a, b, c, d):
    # this function return the negative conditional log probability of positive examplse if they have overlap.
    # we want to minimize the return value -log(p(a|b))
    joint_log = probability(meet_min, meet_max)
    # smooth_log_prob = smooth_prob(joint_log)
    # cliped_smooth_log_prob = tf.clip_by_value(smooth_log_prob, np.log(1e-8), np.log(1.0))
    return joint_log

# if not have meet, which means two disjoint events, the joint probablity is 0
# if have meet, then the joint probability is the measure of the meet cube
def test_joint_probability_log(join_min, join_max, meet_min, meet_max, a, b, c, d, not_have_meet):
    result = slicing_where(condition = not_have_meet,
      full_input = ([join_min, join_max, meet_min, meet_max, a, b, c, d]),
      true_branch = lambda x: lambda_zero_log_upper_bound(*x),
      false_branch = lambda x: lambda_batch_test_joint_cube_measure(*x))

    return result

def test_cond_probability_log(join_min, join_max, meet_min, meet_max, a, b, c, d, not_have_meet):
    result = slicing_where(condition = not_have_meet,
      full_input = ([join_min, join_max, meet_min, meet_max, a, b, c, d]),
      true_branch = lambda x: lambda_zero_log_upper_bound(*x),
      false_branch = lambda x: lambda_batch_log_cube_measure(*x))
    return result

"""helper function"""

def smooth_prob(input_prob):
    pos_prob1 = tf.log(1-lambda_value) + input_prob # (batch_size)
    pos_prob2 = tf.stack([pos_prob1, tf.zeros_like(input_prob) + tf.log(lambda_value) + tf.log(0.5)], axis = 1) #(batch_size, 2)
    pos_prob = tf.reduce_logsumexp(pos_prob2, axis = 1) #(batch_size)
    return pos_prob

# # log vector is a vector of negative log probabilities
def create_log_distribution(logits, batch_size):
    log_1_minus = log1mexp(-logits)
    # log_1 = tf.log(tf.ones([batch_size]))
    # log_1_minus = logits + tf.log(tf.exp(log_1 - logits) - tf.ones([batch_size]))
    return tf.concat([tf.expand_dims(logits, 1), tf.expand_dims(log_1_minus, 1)], 1)

# # vector is a tensor of (gold) probabilities
def create_distribution(probs, batch_size):
    one_minus = tf.ones([batch_size]) - probs
    return tf.concat([tf.expand_dims(probs, 1), tf.expand_dims(one_minus, 1)], 1)

def log1mexp(input_a):
    # input_a: positive
    # return the same shape as input
    result = slicing_where(condition = tf.less_equal(input_a, tf.log(2.0)),
      full_input = -input_a,
      true_branch = lambda x: tf.log(-tf.expm1(x)), 
      false_branch = lambda x: tf.log1p(-tf.exp(x)))
    return result

def make_positive_min(vec):
    return relu(vec)

def make_positive_delta(vec, method):
    if method =='relu':
        return relu(vec)
    elif method == 'softplus':
        return tf.nn.softplus(vec)
    else:
        print('delti_acti is wrong')
        return None

def relu(vec):
    return tf.maximum(vec, tf.zeros_like(vec))

"""helper function. took from stackoverflow."""
def slicing_where(condition, full_input, true_branch, false_branch):
  """Split 'full_input' between 'true_branch' and 'false_branch' on 'condition'.

  Args:
    condition: A boolean Tensor with shape [B_1, ..., B_N].
    full_input: A Tensor or nested tuple of Tensors of any dtype, each with
      shape [B_1, ..., B_N, ...], to be split between 'true_branch' and
      'false_branch' based on 'condition'.
    true_branch: A function taking a single argument, that argument having the
      same structure and number of batch dimensions as 'full_input'. Receives
      slices of 'full_input' corresponding to the True entries of
      'condition'. Returns a Tensor or nested tuple of Tensors, each with batch
      dimensions matching its inputs.
    false_branch: Like 'true_branch', but receives inputs corresponding to the
      false elements of 'condition'. Returns a Tensor or nested tuple of Tensors
      (with the same structure as the return value of 'true_branch'), but with
      batch dimensions matching its inputs.
  Returns:
    Interleaved outputs from 'true_branch' and 'false_branch', each Tensor
    having shape [B_1, ..., B_N, ...].
  """
  full_input_flat = nest.flatten(full_input)
  true_indices = tf.where(condition)
  false_indices = tf.where(tf.logical_not(condition))
  true_branch_inputs = nest.pack_sequence_as(
    structure=full_input,
    flat_sequence=[tf.gather_nd(params=input_tensor, indices=true_indices) for input_tensor in full_input_flat])
  false_branch_inputs = nest.pack_sequence_as(
    structure=full_input,
    flat_sequence=[tf.gather_nd(params=input_tensor, indices=false_indices) for input_tensor in full_input_flat])
  true_outputs = true_branch(true_branch_inputs)
  false_outputs = false_branch(false_branch_inputs)
  nest.assert_same_structure(true_outputs, false_outputs)
  def scatter_outputs(true_output, false_output):
    batch_shape = tf.shape(condition)
    scattered_shape = tf.concat([batch_shape, tf.shape(true_output)[tf.rank(batch_shape):]], 0)
    true_scatter = tf.scatter_nd(
      indices=tf.cast(true_indices, tf.int32),
      updates=true_output,
      shape=scattered_shape)
    false_scatter = tf.scatter_nd(
      indices=tf.cast(false_indices, tf.int32),
      updates=false_output,
      shape=scattered_shape)
    return true_scatter + false_scatter
  result = nest.pack_sequence_as(
    structure=true_outputs,
    flat_sequence=[scatter_outputs(true_single_output, false_single_output) for true_single_output, false_single_output in zip(nest.flatten(true_outputs), nest.flatten(false_outputs))])
  return result

def box_model_embed(args, fstate1, fstate2, W1, b1, W2, b2):
    # generate box embedding for each term using two feed forward.
    # logit1/2 is min embedding for both terms
    # delta_logits1/2 is delta embedding for both terms
    logits1 = tf.matmul(fstate1[0], W1) + b1
    logits2 = tf.matmul(fstate2[0], W1) + b1
    delta_logits1 = tf.matmul(fstate1[0], W2) + b2
    delta_logits2 = tf.matmul(fstate2[0], W2) + b2

    # make both min and delta embeddings positive.
    t1_min_embed = make_positive_min(logits1)
    t2_min_embed = make_positive_min(logits2)

    if args['mode'] == 'cube':
        t1_delta_embed = make_positive_delta(delta_logits1, args['delta_acti'])
        t2_delta_embed = make_positive_delta(delta_logits2, args['delta_acti'])

    elif args['mode'] == 'cube_fix':
        t1_delta_embed = 200.00-tf.nn.softplus(delta_logits1)
        t2_delta_embed = 200.00-tf.nn.softplus(delta_logits2)
    else:
        print('invalid mode parameter')

    # get max embedding
    t1_max_embed = t1_min_embed + t1_delta_embed
    t2_max_embed = t2_min_embed + t2_delta_embed

    return t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed

def box_model_params(t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed):
    # get join, meet, and condition
    join_min, join_max, meet_min, meet_max, not_have_meet = calc_join_and_meet(t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed)
    return join_min, join_max, meet_min, meet_max, not_have_meet

def box_prob(join_min, join_max, meet_min, meet_max, not_have_meet, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed):
    # for evaluation
    joint_predicted = test_joint_probability_log(join_min, join_max, meet_min, meet_max, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed, not_have_meet)
    cpr_predicted = test_cond_probability_log(join_min, join_max, meet_min, meet_max, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed, not_have_meet)
    cpr_predicted_reverse = test_cond_probability_log(join_min, join_max, meet_min, meet_max, t2_min_embed, t2_max_embed, t1_min_embed, t1_max_embed, not_have_meet)
    # for training
    x_predicted = probability(t1_min_embed, t1_max_embed)
    y_predicted = probability(t2_min_embed, t2_max_embed)
    return joint_predicted, x_predicted, y_predicted, cpr_predicted, cpr_predicted_reverse

