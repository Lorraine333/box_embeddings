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
from utils import data_loader
from utils import loader_utils
# import numpy as np
# from itertools import *


def define_placeholder():
    placeholder = {}
    # positive example term1
    placeholder['t1_idx_placeholder'] = tf.placeholder(tf.int32, shape=(None, None))
    placeholder['t1_msk_placeholder'] = tf.placeholder(tf.int32, shape=(None, None, 1))
    placeholder['t1_length_placeholder'] = tf.placeholder(tf.int32, shape=(None, 1))
    # positive example term2
    placeholder['t2_idx_placeholder'] = tf.placeholder(tf.int32, shape=(None, None))
    placeholder['t2_msk_placeholder'] = tf.placeholder(tf.int32, shape=(None, None, 1))
    placeholder['t2_length_placeholder'] = tf.placeholder(tf.int32, shape=(None, 1))
    # positive relation
    placeholder['rel_placeholder'] = tf.placeholder(tf.int32, shape=[None])
    placeholder['rel_msk_placeholder'] = tf.placeholder(tf.float32, shape=[None, 1])
    # positive label
    placeholder['label_placeholder'] = tf.placeholder(tf.float32, shape=[None])

    return placeholder


def fill_feed_dict(data_set, placeholder, rel, batch_size):
    r_idx, t1_idx, t2_idx, labels = data_set.next_batch(batch_size)
    t1x, t1mask, t1length = data_loader.prepare_data(t1_idx)
    t2x, t2mask, t2length = data_loader.prepare_data(t2_idx)
    relmsk = data_loader.rel_msk(r_idx, rel)
    feed_dict = {
        placeholder['t1_idx_placeholder']: t1x,
        placeholder['t1_msk_placeholder']: t1mask,
        placeholder['t1_length_placeholder']: t1length,
        placeholder['t2_idx_placeholder']: t2x,
        placeholder['t2_msk_placeholder']: t2mask,
        placeholder['t2_length_placeholder']: t2length,
        placeholder['rel_placeholder']: r_idx,
        placeholder['label_placeholder']: labels,
        placeholder['rel_msk_placeholder']: relmsk,
    }

    return feed_dict


def fill_taxo_feed_dict(words, idx2word, placeholder, input_dict=True):
    # construct samples
    # t1x is [1,1,1,1,2,2,2,2,3,3,3,3] if vocab size is 4
    # t2x is [1,2,3,4,1,2,3,4,1,2,3,4] if vocab size is 4
    if input_dict:
        idx2word = list(idx2word.keys())
    t1x = list(chain(*([i] * len(idx2word) for i in words)))
    t2x = list(chain(*(idx2word for i in words)))
    # print('t1x', len(t1x))
    # print('t2x', len(t2x))

    feed_dict = {
        placeholder['t1_idx_placeholder']: np.asarray(t1x).reshape(len(t1x), 1),
        placeholder['t2_idx_placeholder']: np.asarray(t2x).reshape(len(t1x), 1),
    }
    return feed_dict
