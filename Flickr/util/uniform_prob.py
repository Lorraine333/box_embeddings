import tensorflow as tf
import numpy as np
import math
from tensorflow.python.util import nest
tf.set_random_seed(20160408)

# log probability
# sum(-x)
def probability(m, max_logits):
    m = tf.sigmoid(m)
    log_prob = tf.reduce_sum(tf.log((max_logits - m) + 1e-8) ,axis = 1)
    return log_prob


def joint_probability_log(m1, m2, max_logits1, max_logits2):
    intersect_min, intersect_max = intersection_point_log(m1, m2, max_logits1, max_logits2)
    return probability(intersect_min, intersect_max)


def cond_probability_log(m1, m2, max_logits1, max_logits2):
    # pair
    intersect_min, intersect_max = intersection_point_log(m1, m2, max_logits1, max_logits2)
    joint_prob_pair = probability(intersect_min, intersect_max)
    y_prob_pair = probability(m2, max_logits2)
    cond_prob_pair = joint_prob_pair - y_prob_pair  # log probabilities
    cond_prob_pair = tf.clip_by_value(cond_prob_pair, math.log(1e-10), -0.0001)
    return cond_prob_pair


def intersection_point_log(m1, m2, max_logits1, max_logits2):
    meet_min = tf.maximum(m1, m2) #batchsize * embed_size
    meet_max = tf.minimum(max_logits1, max_logits2) #batchsize * embed_size
    return meet_min, meet_max


# log vector is a vector of negative log probabilities
def create_log_distribution(logits, batch_size):
    log_1_minus = log1mexp(-logits)
    # log_1 = tf.log(tf.ones([batch_size]))
    # log_1_minus = logits + tf.log(tf.exp(log_1 - logits) - tf.ones([batch_size]))
    return tf.concat([tf.expand_dims(logits, 1), tf.expand_dims(log_1_minus, 1)], 1)

# vector is a tensor of (gold) probabilities
def create_distribution(probs, batch_size):
    one_minus = tf.ones([batch_size]) - probs
    return tf.concat([tf.expand_dims(probs, 1), tf.expand_dims(one_minus, 1)], 1)


# assume input is 2 vectors:
# vec1 : predicted cpr values (negative log prob)
# vec2 : gold cpr values
def kl_divergence_batch(pred_vec, gold_vec):
    pred_cpr = np.exp(pred_vec)
    pred_cpr = np.clip(pred_cpr, 0, 1)
    vals = []
    for index, pred_val in enumerate(pred_cpr):
        gold_val = gold_vec[index]
        vals.append(kl_divergence(pred_val, gold_val))
    return np.mean(vals)


def kl_divergence(pred_prob, gold_prob):
    val = 0
    if gold_prob > 0 and pred_prob > 0:
        try:
            val += gold_prob * (math.log(gold_prob / pred_prob))
        except ValueError:
            print(gold_prob, pred_prob)
    if gold_prob < 1 and pred_prob < 1:
        try:
            val += (1-gold_prob) * (math.log((1-gold_prob)/(1-pred_prob)))
        except ValueError:
            print(gold_prob, pred_prob)
    return val


def log1mexp(input_a):
    # input_a: positive
    # return the same shape as input
    result = slicing_where(condition = tf.less_equal(input_a, tf.log(2.0)),
      full_input = -input_a,
      true_branch = lambda x: tf.log(-tf.expm1(x)), 
      false_branch = lambda x: tf.log1p(-tf.exp(x)))
    # result = tf.log1p(-tf.exp(-input_a)))
    return result


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