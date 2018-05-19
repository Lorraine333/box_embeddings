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

import random
import numpy as np
from nltk.corpus import wordnet as wn
import collections

def get_wordnet_synset_parents(words, word2idx, idx2word, use_lemma = False, use_top = True):
  # input: list of idx of test words
  # output: parents list for every word in words, including recurrent parents happended in word list
  print('Getting wordnet parents now')
  result = []
  mydict = list(word2idx.keys())
  top_concepts = get_top_concept(3, use_lemma)
  for word_idx in words:
    parent_list = []
    base_node = collections.defaultdict(list)
    # get parents list for the single word
    synset = wn.synset(idx2word[word_idx])
    if synset.hypernyms():
      for h_synset in synset.closure(lambda s: s.hypernyms()):
        parent_list.append(word2idx[h_synset.name()])
    else:
      for first_synset in synset.instance_hypernyms():
        for h_synset in first_synset.closure(lambda s: s.hypernyms()):
          parent_list.append(word2idx[h_synset.name()])
    base_node[word_idx] = parent_list
    # for each parent in the parents list, find it's parents too.
    # special case: 1) node doesn't have parents anymore
    for p in parent_list:
      sub_parent_list = []
      # get sub parents list for the word in the parents list
      synset = wn.synset(idx2word[p])
      if synset.hypernyms():
        for h_synset in synset.closure(lambda s: s.hypernyms()):
          sub_parent_list.append(word2idx[h_synset.name()])
      else:
        for first_synset in synset.instance_hypernyms():
          for h_synset in first_synset.closure(lambda s: s.hypernyms()):
            sub_parent_list.append(word2idx[h_synset.name()])
      base_node[p] = sub_parent_list
    # for k in base_node:
    #   print('key',idx2word[k])
    #   for value in base_node[k]:
    #     print('value',idx2word[value],)
    result.append(base_node)
  return result

def get_top_concept(level, use_lemma):
    result = []
    hypo = lambda s: s.hyponyms()
    for synset in (wn.synset('entity.n.01').closure(hypo, depth=level)):
        if use_lemma:
            for lemma in get_lemma(synset):
                if lemma not in result:
                    result.append(lemma)
        else:
            result.append(synset)
    # print('Number of Top Concept Lemma', len(result))
    # print('Example Concept Lemma', result[-10:])
    return result

def get_lemma(synsets):
	return [lemma.name() for lemma in synsets.lemmas()]
