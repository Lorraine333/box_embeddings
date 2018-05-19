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

from __future__ import division
from __future__ import print_function
import pickle
import loader_utils
import tensorflow as tf
import numpy as np
from random import *
from taxo_eval_prep import *

# flags = tf.app.flags
# FLAGS = flags.FLAGS
#
# random_seed = FLAGS.random_seed
random_seed = 20180112

class DataSet(object):
    def __init__(self, input_tuples):
        """Construct a DataSet"""
        self._num_examples = len(input_tuples)
        self._tuples = input_tuples
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        if batch_size == 0:
            """Return the next whole examples from this eval data set."""
            end = self._num_examples
            next_batch = self._tuples[0:end]
        else:
            start = self._index_in_epoch
            self._index_in_epoch += batch_size
            if self._index_in_epoch > self._num_examples:
                self._epochs_completed += 1
                seed(random_seed)
                shuffle(self._tuples)
                start = 0
                self._index_in_epoch = batch_size
                assert batch_size <= self._num_examples
            end = self._index_in_epoch
            next_batch = self._tuples[start:end]
        batch_idx = [i for i in next_batch]
        r_idx, t1_idx, t2_idx, s = loader_utils.sep_idx(batch_idx)
        l = np.ones(len(s))
        for i in range(len(s)):
            if s[i] <= 0:
                l[i] = 0.
        return np.asarray(r_idx), t1_idx, t2_idx, l


def read_data_sets(FLAGS, dtype=tf.float32):
    train_dir = FLAGS.train_dir

    class DataSets(object):
        pass

    data_sets = DataSets()

    # define file names
    TRAIN_FILE = train_dir + '/' + FLAGS.train_file
    TRAIN_TEST_FILE = train_dir + '/' + FLAGS.train_test_file
    DEV_FILE = train_dir + '/' + FLAGS.dev_file
    DEVTEST_FILE = train_dir + '/' + FLAGS.test_file
    DICT_FILE = train_dir + '/dict.txt'
    REL_FILE = train_dir + '/rel.txt'

    # read in all files using loader_utils functions
    word2idx, idx2word = loader_utils.get_vocab(DICT_FILE)
    rel2idx = loader_utils.get_relation(REL_FILE)
    train_data = loader_utils.get_data(TRAIN_FILE, word2idx, rel2idx)
    train_test_data = loader_utils.get_data(TRAIN_TEST_FILE, word2idx, rel2idx)
    dev_data = loader_utils.get_data(DEV_FILE, word2idx, rel2idx)
    devtest_data = loader_utils.get_data(DEVTEST_FILE, word2idx, rel2idx)

    data_sets.train = DataSet(train_data)
    data_sets.train_test = DataSet(train_test_data)
    data_sets.dev = DataSet(dev_data)
    data_sets.devtest = DataSet(devtest_data)
    data_sets.word2idx = word2idx
    data_sets.idx2word = idx2word
    data_sets.rel2idx = rel2idx
    data_sets.vocab_size = len(word2idx)

    if FLAGS.w2 > 0.0:
        # if the loss term minimize marginal prob as well, then read in the marginal file
        MARG_FILE = train_dir + '/' + FLAGS.marg_prob_file
        marginal_prob = loader_utils.get_count(MARG_FILE)
        data_sets.margina_prob = marginal_prob

    # if embeddings are initialized using pre trained model, then read them in
    if FLAGS.init_embedding == 'pre_train':
        trained_model = pickle.load(open(FLAGS.init_embedding_file, "rb"))
        min_embed = trained_model['embeddings']
        delta_embed = trained_model['imag_embeddings']
        data_sets.min_embed = min_embed
        data_sets.delta_embed = delta_embed

    # if evaluation method is taxonomy evaluation which may be used when evaluate wordnet data, then read all taxo eval related stuff in.
    if FLAGS.eval == 'taxo':
        TAXO_DEV_FILE = train_dir + '/wordnet_taxo_dev_single.txt'
        TAXO_TEST_FILE = train_dir + '/wordnet_taxo_dev_single.txt'
        taxo_dev = loader_utils.get_taxo_words(TAXO_DEV_FILE, word2idx)
        taxo_test = loader_utils.get_taxo_words(TAXO_TEST_FILE, word2idx)

        taxo_dev_parents = get_wordnet_synset_parents(taxo_dev, word2idx, idx2word)
        taxo_test_parents = get_wordnet_synset_parents(taxo_test, word2idx, idx2word)
        data_sets.taxo_dev = taxo_dev
        data_sets.taxo_test = taxo_test
        data_sets.taxo_dev_parents = taxo_dev_parents
        data_sets.taxo_test_parents = taxo_test_parents

    return data_sets

def prepare_data(list_of_seqs):
    lengths = [len(s) for s in list_of_seqs]
    n_samples = len(list_of_seqs)
    maxlen = np.max(lengths)
    x = np.zeros((n_samples, maxlen)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen, 1)).astype('int32')
    x_len = np.zeros((n_samples, 1)).astype('int32')
    for idx, s in enumerate(list_of_seqs):
        x[idx, :lengths[idx]] = s
        x[idx, lengths[idx]:] = 0.
        x_len[idx, :] = lengths[idx]
        x_mask[idx, :lengths[idx]] = 1.
    return x, x_mask, x_len

def rel_msk(rel_idx, rel):
    relmsk = np.ones((rel_idx.shape[0], 1)).astype('float32')
    for i in range(len(rel_idx)):
        if rel['isa'] == int(rel_idx[i]):
            relmsk[i] = 0
    return relmsk.reshape(relmsk.shape[0], 1)