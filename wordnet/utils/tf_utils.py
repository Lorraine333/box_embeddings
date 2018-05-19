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
import numpy as np
from tensorflow.python.util import nest

flags = tf.app.flags
FLAGS = flags.FLAGS

# tf.set_random_seed(FLAGS.random_seed)
# my_seed = FLAGS.random_seed
my_seed = 20180112

def make_sigmoid_cube(t1_min_embed, t1_delta_embed, t2_min_embed, t2_delta_embed):
    """given any word embeddings, make it into uniform cube. This function is done by apply sigmoid function."""
    t1_min_embed = tf.sigmoid(t1_min_embed)
    t2_min_embed = tf.sigmoid(t2_min_embed)
    t1_delta_embed = t1_min_embed + (1-t1_min_embed) * tf.sigmoid(t1_delta_embed)
    t2_delta_embed = t2_min_embed + (1-t2_min_embed) * tf.sigmoid(t2_delta_embed)
    t1_max_embed = t1_min_embed + t1_delta_embed
    t2_max_embed = t2_min_embed + t2_delta_embed
    return t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed

def make_softmax_cube(t1_min_embed, t1_delta_embed, t2_min_embed, t2_delta_embed):
    """given any word embeddings, make it into uniform cube. This function is done by apply softmax function."""

    # init feed forward network
    input_dim = int(t1_min_embed.get_shape()[1])
    output_dim = 100
    w1 = tf.Variable(tf.random_uniform([input_dim, output_dim], minval=-0.05, maxval=0.05, seed=my_seed), name='Output1', trainable=True)
    b1  = tf.Variable(tf.random_uniform([output_dim], minval=-0.05, maxval=0.05, seed=my_seed), name='OutputBias1', trainable=True)

    w2 = tf.Variable(tf.random_uniform([input_dim, output_dim], minval=-0.05, maxval=0.05, seed=my_seed), name='Output2', trainable=True)
    b2 = tf.Variable(tf.random_uniform([output_dim], minval=5.00, maxval=5.10, seed=my_seed), name='OutputBias2', trainable=True)

    # get the min value, softmax[0]
    s01 = tf.matmul(t1_min_embed, w1) + b1
    s02 = tf.matmul(t2_min_embed, w1) + b1
    # get the delta value, softmax[1]
    s11 = tf.matmul(t1_delta_embed, w2) + b2
    s12 = tf.matmul(t2_delta_embed, w2) + b2

    # fix third one to be 0, softmax[2]
    s21 = s22 =tf.zeros_like(s11)
    t1_embed = tf.stack([s01, s11, s21], axis = 2)
    t2_embed = tf.stack([s02, s12, s22], axis = 2)

    # use softmax to get embedding
    soft_t1_embed =tf.nn.softmax(t1_embed)
    soft_t1_min_embed=soft_t1_embed[:,:,0]
    soft_t1_delta_embed = soft_t1_embed[:,:,1]

    soft_t2_embed =tf.nn.softmax(t2_embed)
    soft_t2_min_embed=soft_t2_embed[:,:,0]
    soft_t2_delta_embed = soft_t2_embed[:,:,1]

    soft_t1_max_embed = soft_t1_min_embed + soft_t1_delta_embed
    soft_t2_max_embed = soft_t2_min_embed + soft_t2_delta_embed

    return soft_t1_min_embed, soft_t1_max_embed, soft_t2_min_embed, soft_t2_max_embed

# for compose phrase embeddings
def tuple_lstm_embedding(x, x_mask, x_length, We, term_rnn, reuse):
    print('Using LSTM to compose term vectors')

    embed = tf.nn.embedding_lookup(We, x)  # batchsize * maxlength *  embed_size
    x_length = tf.reshape(x_length, [-1])
    if reuse:
        tf.get_variable_scope().reuse_variables()
    output, state = tf.nn.dynamic_rnn(term_rnn, embed, dtype=tf.float32, sequence_length=x_length)

    # select relevant vectors
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + x_length - 1
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant

def tuple_embedding(x, x_mask, x_length, We):
    print('Averaging vectors to get term vectos')
    embed = tf.nn.embedding_lookup(We, x)  # batchsize * length *  embed_size
    x_mask = tf.cast(x_mask, tf.float32)
    embed = embed * x_mask  # x_mask: batchsize * length * 1
    embed = tf.reduce_sum(embed, 1)
    x_length = tf.cast(x_length, tf.float32)
    embed = embed / x_length
    return embed

def log1mexp(input_a):
    # input_a: positive
    # return the same shape as input
    result = slicing_where(condition = tf.less_equal(input_a, tf.log(2.0)),
      full_input = -input_a,
      true_branch = lambda x: tf.log(-tf.expm1(x)),
      false_branch = lambda x: tf.log1p(-tf.exp(x)))
    return result

def clip(x,min,max):
    return tf.minimum(tf.maximum(x,min),max)

def bernoulli_kl_xent(log_p, q_true):
    """p and q are vectors of bernoulli log probabilities of "true" outcome"""
    log_q = tf.log(q_true+1e-8)
    # clip for numerical stability
    log_min = -18.420680744 # log(1e-8)
    log_max = -1.00000001002e-08 # log(1-1e-8)
    log_p_true = clip(log_p,log_min,log_max)
    log_q_true = clip(log_q,log_min,log_max)

    p_true = tf.exp(log_p)
    log_p_false = log1mexp(-log_p_true)
    log_q_false = log1mexp(-log_q_true)

    p_true = tf.reshape(p_true, [-1])
    q_true = tf.reshape(q_true, [-1])
    log_p_true = tf.reshape(log_p_true, [-1])
    log_q_true = tf.reshape(log_q_true, [-1])
    log_p_false = tf.reshape(log_p_false, [-1])
    log_q_false = tf.reshape(log_q_false, [-1])
    # kl = p_true*(log_p_true-log_q_true)+(1-p_true)*(log_p_false-log_q_false)
    # ok, so this is right. feb 19
    kl = q_true*(log_q_true-log_p_true)+(1-q_true)*(log_q_false-log_p_false)

    # this is a constant if this is the label so optimizing KL or xent doesnt matter
    # if we integrate wrt label, it just doesnt give likelihood value
    h_p = -(p_true*log_p_true+(1-p_true)*log_p_false)
    xent = kl + h_p
    xent_fast = -(p_true*log_q_true+(1-p_true)*log_q_false)

    return kl, xent_fast

def robust_norm(x):
    x = x + 1e-8
    a = tf.reduce_max(tf.abs(x), axis = 2, keep_dims = True)
    return tf.squeeze(a, [2]) * tf.norm(x / a, axis = 2)

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
