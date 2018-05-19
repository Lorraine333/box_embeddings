# """Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License."""

# """Evaluates the network."""
from __future__ import division
from __future__ import print_function
from utils import feeder
import numpy as np
from collections import defaultdict


def best_threshold(errs, target):
  indices = np.argsort(errs)
  sortedErrors = errs[indices]
  sortedTarget = target[indices]
  tp = np.cumsum(sortedTarget)
  invSortedTarget = (sortedTarget == 0).astype('float32')
  Nneg = invSortedTarget.sum()
  fp = np.cumsum(invSortedTarget)
  tn = fp * -1 + Nneg
  accuracies = (tp + tn) / sortedTarget.shape[0]
  i = accuracies.argmax()
  # calculate recall precision and F1
  Npos = sortedTarget.sum()
  fn = tp * -1 + Npos
  precision = tp/(tp + fp)
  recall = tp/(tp + fn)
  f1 = (2*precision[i]*recall[i])/(precision[i]+recall[i])
  print("Best threshold", sortedErrors[i], "Accuracy:", accuracies[i], "Precision, Recall and F1 are %.5f %.5f %.5f" % (precision[i], recall[i], f1))
  print("TP, FP, TN, FN are %.5f %.5f %.5f %.5f" % (tp[i], fp[i], tn[i], fn[i]))
  return sortedErrors[i], accuracies[i]


def single_eval(sess, error, placeholder, data_set, rel2idx, FLAGS, error_file_name):
  feed_dict = feeder.fill_feed_dict(data_set, placeholder, rel2idx, 0)
  true_label = feed_dict[placeholder['label_placeholder']]
  pred_error = sess.run(error, feed_dict = feed_dict)
  _, acc = best_threshold(pred_error, true_label)
  return acc


def do_eval(sess, error, placeholder,dev, devtest, curr_best, FLAGS,error_file_name, rel2idx, word2idx):
  feed_dict_dev = feeder.fill_feed_dict(dev, placeholder, rel2idx, 0)
  true_label = feed_dict_dev[placeholder['label_placeholder']]
  pred_error = sess.run(error, feed_dict = feed_dict_dev)
  print('Dev Stats:', end = '')
  thresh, _ = best_threshold(pred_error, true_label)

  #evaluat devtest
  feed_dict_devtest = feeder.fill_feed_dict(devtest, placeholder, rel2idx, 0)
  true_label_devtest = feed_dict_devtest[placeholder['label_placeholder']]
  devtest_he_error = sess.run(error, feed_dict = feed_dict_devtest)

  pred = devtest_he_error <= thresh
  correct = (pred == true_label_devtest)
  accuracy = float(correct.astype('float32').mean())
  wrong_indices = np.logical_not(correct).nonzero()[0]
  wrong_preds = pred[wrong_indices]

  if accuracy>curr_best:
  # #evaluat devtest
    error_file = open(error_file_name+"_test.txt",'wt')
    if FLAGS.rel_acc:
      rel_acc_checker(feed_dict_devtest, placeholder, correct, dev, error_file, rel2idx)

    if FLAGS.error_analysis:
      err_analysis(dev, wrong_indices, feed_dict_devtest, placeholder, error_file, rel2idx, word2idx, devtest_he_error)

  return accuracy


def err_analysis(data_set, wrong_indices, feed_dict, placeholder, error_file, rel, words, errors):
  temp,temp1, temp2 = {}, {}, {}
  for w in words:
    temp[words[w]] = w
  for w1 in rel:
    temp1[rel[w1]] = w1

  # print(wrong_indices)
  # outputfile = open('result/train_test'+str(num)+'.txt','wt') 
  for i in wrong_indices:
    wrong_t1 = feed_dict[placeholder['t1_idx_placeholder']][i]
    wrong_t2 = feed_dict[placeholder['t2_idx_placeholder']][i]
    wrong_rel = feed_dict[placeholder['rel_placeholder']][i]
    wrong_lab = feed_dict[placeholder['label_placeholder']][i]
    # print(i)

    for t in wrong_t1:
      if "</s>" not in temp[t]:
        print(temp[t]+"|",end=''),
        # print("\t"),
        # outputfile.write(temp[t]+"_")
        # outputfile.write("\t")
    for t2 in wrong_t2:
      if "</s>" not in temp[t2]:
        print(temp[t2]+"|",end='')
        # print("\t"),
        # outputfile.write(temp[t2]+"_")
        # outputfile.write("\t")
    print(temp1[wrong_rel]+'\t',end='')
    print(str(wrong_lab))
    # print(errors[i])
  #check different relation wrong numbers
    if wrong_rel in temp2:
      temp2[wrong_rel] += 1
    else:
      temp2[wrong_rel] = 1
    # outputfile.write(temp1[wrong_rel]+"\t")
    # outputfile.write(str(wrong_lab)+"\n")
  print('relation analysis', file = error_file)
  for key in temp2:
    print(str(temp1[key]) + ":" +str(temp2[key]), file = error_file)
    # outputfile.write(str(temp1[key]) + ":" +str(temp2[key])+"\n")



def rel_acc_checker(feed_dict_devtest, placeholder, correct, data_set, error_file, rel):
  print('Relation Accurancy','*'*50, file = error_file)
  #check the different relation accurancy
  test_rel_id = feed_dict_devtest[placeholder['rel_placeholder']]

  # count the relation 
  cnt = defaultdict(int)
  for t in test_rel_id:
    cnt[t] += 1
  print('Relation Count', '*'*50, file = error_file)
  for c in cnt:
    print(c, cnt[c], file = error_file)

  # count the correct prediction for each relation
  right = {}
  for i in range(len(correct)):
    if test_rel_id[i] in right and correct[i]:
      right[test_rel_id[i]] += 1
    elif test_rel_id[i] not in right and correct[i]:
      right[test_rel_id[i]] = 1
    elif test_rel_id[i] not in right and not correct[i]:
      right[test_rel_id[i]] = 0

  # calculate the accurancy for different relation
  result = defaultdict(int)
  for j in cnt:
    result[j] = float(right[j])/float(cnt[j])

  # print out the result
  rel_dict = {}
  for w1 in rel:
    rel_dict[rel[w1]] = w1
    # print(rel_dict)
  for rel in result:
    acc = result[rel]
  #  print(rel)
    print(rel_dict[rel],rel, acc, file = error_file)