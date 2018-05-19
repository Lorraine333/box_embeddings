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

import numpy as np
from random import *
import collections
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

random_seed = 20180112


def get_vocab(filename):
    """read vocab file, only return word2idx or idx2word. Not including word embeddings"""
    word2idx = {}
    f = open(filename, 'r')
    lines = f.readlines()
    for (n, word) in enumerate(lines):
        # n is the line number-1 (start from 0), word is the actually word
        word = word.strip()
        word2idx[word] = n
    idx2word = {word2idx[j]: j for j in word2idx}
    f.close()
    return word2idx, idx2word


def get_relation(filename):
    """read relation file, return rel2idx"""
    rel2idx = {}
    f = open(filename, 'r')
    lines = f.readlines()
    for (n, rel) in enumerate(lines):
        rel = rel.strip().lower()
        rel2idx[rel] = n
    f.close()
    return rel2idx


def get_data(filename, word2idx, rel2idx):
    """Read data: relation \t term1 \t term2 \t score or label"""
    f = open(filename, 'r')
    lines = f.readlines()
    examples = []
    for i in lines:
        i = i.strip()
        if (len(i) > 0):
            i = i.split('\t')
            e = (i[0], i[1], i[2], float(i[3]))
            e_idx = convertToIndex(e, word2idx, rel2idx)
            examples.append(e_idx)
    seed(random_seed)
    shuffle(examples)
    f.close()
    print('read data from', filename, 'of length', len(examples))
    return examples


def get_count(filename):
    """Read in marginal prob, to form a matrix of size: vocab * 1"""
    count = []
    with open(filename) as inputfile:
        lines = inputfile.read().splitlines()
        for i in range(len(lines)):
            line = lines[i]
            line = line.strip()
            count.append(line)
    print('read data from ', filename, 'of length', len(count))
    return np.reshape(np.matrix(count, dtype=np.float32), (len(count), 1))

def convertToIndex(e, word2idx, rel2idx):
    if len(e) > 1:
        (r, t1, t2, s) = e
        # we want to seperate them because sometimes words are phrases, need multiple lookups in one term
        return (lookupRelIDX(rel2idx, r), lookupwordID(word2idx, t1), lookupwordID(word2idx, t2), float(s))
    else:
        return (lookupwordID(word2idx, e))


def get_taxo_words(filename, word2idx):
    result = []
    with open(filename) as inputfile:
        lines = inputfile.readlines()
        for line in lines:
            line = line.strip()
            result.append(word2idx[line])
    print('Read taxonomy data from ', filename, 'of length ', len(result))
    return result


def lookupwordID(words, w):
    result = []
    array = w.split(' ')
    for i in range(len(array)):
        if (array[i] in words):
            result.append(words[array[i]])
        else:
            result.append(words['UUUNKKK'])
    return result


def lookupRelIDX(rel_dict, r):
    r = r.lower()
    if r in rel_dict:
        return rel_dict[r]
    else:
        return rel_dict['UUUNKKK']


def sep_idx(e_idx):
    g1 = [];
    g2 = [];
    R = [];
    labels = []
    for e in e_idx:
        (r, t1, t2, s) = e
        g1.append(t1)
        g2.append(t2)
        R.append(r)
        labels.append(s)
    return R, g1, g2, labels
