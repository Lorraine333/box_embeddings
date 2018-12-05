from __future__ import division
import random

import numpy as np
import Probability
import corr_prob as corr_prob
from util.DataLoader import DataLoader
Sparse = DataLoader()
random.seed(20160408)

class Training:

    def __init__(self, sess, optimizer, mean_loss, x_loss, cpr_loss, x_predicted, y_predicted, xy_predicted, cpr_xy_predicted, cpr_xy_predicted_rev, dataset1, dataset2, x_labels, y_labels, xy_labels, cpr_xy_labels, lengths1, lengths2, batch_size, maxlength, dropout_ph, dropout, gradient):
        self.sess = sess
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.mean_loss = mean_loss
        self.x_loss = x_loss
        self.cpr_loss = cpr_loss
        self.x_predicted = x_predicted
        self.y_predicted = y_predicted
        self.xy_predicted = xy_predicted
        self.cpr_xy_predicted = cpr_xy_predicted
        self.cpr_xy_predicted_rev = cpr_xy_predicted_rev
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.x_labels = x_labels
        self.y_labels = y_labels
        self.xy_labels = xy_labels
        self.cpr_xy_labels = cpr_xy_labels
        self.lengths1 = lengths1
        self.lengths2 = lengths2
        self.maxlength = maxlength
        self.dropout_ph = dropout_ph
        self.dropout = dropout
        self.gradient = gradient

    def eval_analysis(self, data1, data2, x_label, y_label, xy_label, cpr_xy_label, lens1, lens2):
        ind = range(0, self.batch_size)
        data1_sub = []
        data2_sub = []
        for idx in ind:
            data1_sub.append(data1[idx])
            data2_sub.append(data2[idx])
        data1_pair = Sparse.pad_tensor(data1_sub, self.maxlength)
        data2_pair = Sparse.pad_tensor(data2_sub, self.maxlength)
        l1 = lens1[ind]
        l2 = lens2[ind]
        batch_xlabels = x_label[ind]
        batch_ylabels = y_label[ind]
        batch_cpr_xylabels = cpr_xy_label[ind]
        loss, x_predictions, y_predictions, xy_predictions, cpr_predictions, cpr_predictions_rev = self.sess.run([self.mean_loss, self.x_predicted, self.y_predicted, self.xy_predicted, self.cpr_xy_predicted, self.cpr_xy_predicted_rev], feed_dict={ self.x_labels: batch_xlabels, self.y_labels: batch_ylabels, self.cpr_xy_labels: batch_cpr_xylabels, self.dataset1:data1_pair, self.dataset2:data2_pair, self.lengths1:l1, self.lengths2:l2, self.dropout_ph: 1.0})
        return ind, np.exp(x_predictions), np.corrcoef(np.exp(x_predictions), x_label[ind])[0, 1], Probability.kl_divergence_batch(x_predictions, x_label[ind]), np.exp(y_predictions), np.corrcoef(np.exp(y_predictions), y_label[ind])[0, 1], Probability.kl_divergence_batch(y_predictions, y_label[ind]), np.exp(xy_predictions), np.corrcoef(np.exp(xy_predictions), xy_label[ind])[0, 1], np.exp(cpr_predictions), np.corrcoef(np.exp(cpr_predictions), cpr_xy_label[ind])[0, 1], Probability.kl_divergence_batch(cpr_predictions, cpr_xy_label[ind]), cpr_predictions_rev
	# return ind, np.exp(x_predictions), np.corrcoef(np.exp(x_predictions), x_label[ind])[0, 1], np.exp(y_predictions), np.corrcoef(np.exp(y_predictions), y_label[ind])[0, 1], np.exp(xy_predictions), np.corrcoef(np.exp(xy_predictions), xy_label[ind])[0, 1], np.exp(cpr_predictions), np.corrcoef(np.exp(cpr_predictions), cpr_xy_label[ind])[0, 1], Probability.kl_divergence_batch(cpr_predictions, cpr_xy_label[ind]), cpr_predictions_rev        

    def eval(self, data1, data2, x_label, y_label, xy_label, cpr_xy_label, lens1, lens2):
        x_predictions = []
        y_predictions = []
        xy_predictions = []
        cpr_predictions = [] 
        cpr_predictions_rev = []
        gold_corr = []
        pred_corr = []
        indices = range(len(data1))
        padded = False
        total_loss = 0
        for i in range(int((len(data1)/self.batch_size) + 1)):
            while self.batch_size*(i+1) > len(indices):
                indices.append(indices[-1])
                padded = True
            ind = indices[self.batch_size*i:self.batch_size*(i+1)]
            data1_sub = []
            data2_sub = []
            for idx in ind:
                data1_sub.append(data1[idx])
                data2_sub.append(data2[idx])
            data1_pair = Sparse.pad_tensor(data1_sub, self.maxlength)
            data2_pair = Sparse.pad_tensor(data2_sub, self.maxlength)
            l1 = lens1[ind]
            l2 = lens2[ind]
            batch_xlabels = x_label[ind]
            batch_ylabels = y_label[ind]
            batch_xylabels = xy_label[ind]
            batch_cpr_xylabels = cpr_xy_label[ind]
            gold_corr_single = corr_prob.np_correlation(np.log(corr_prob.np_clip_prob(batch_xlabels)), np.log(corr_prob.np_clip_prob(batch_ylabels)), np.log(corr_prob.np_clip_prob(batch_xylabels)))
            loss, x_pred, y_pred, xy_pred, cpr_xy_pred, cpr_xy_pred_reverse = self.sess.run([self.mean_loss, self.x_predicted, self.y_predicted, self.xy_predicted, self.cpr_xy_predicted, self.cpr_xy_predicted_rev], feed_dict={ self.x_labels: batch_xlabels, self.y_labels: batch_ylabels, self.xy_labels:batch_xylabels, self.cpr_xy_labels: batch_cpr_xylabels, self.dataset1:data1_pair, self.dataset2:data2_pair, self.lengths1:l1, self.lengths2:l2, self.dropout_ph: 1.0})
            pred_corr_single = corr_prob.np_correlation(x_pred, y_pred, xy_pred)
            gold_corr.extend(gold_corr_single)
            pred_corr.extend(pred_corr_single)
            total_loss += loss
            x_predictions.extend(x_pred)
            y_predictions.extend(y_pred)
            xy_predictions.extend(xy_pred)
            cpr_predictions.extend(cpr_xy_pred)
            cpr_predictions_rev.extend(cpr_xy_pred_reverse)
        if padded:
            while len(cpr_predictions) > len(data1):
                x_predictions = x_predictions[:-1]
                y_predictions = y_predictions[:-1]
                xy_predictions = xy_predictions[:-1]
                cpr_predictions = cpr_predictions[:-1]
                cpr_predictions_rev = cpr_predictions_rev[:-1]
                gold_corr = gold_corr[:-1]
                pred_corr = pred_corr[:-1]
        corr_loss = np.mean((np.abs(np.asarray(pred_corr)-np.asarray(gold_corr))))
        pos_sign_matching_bool = np.logical_and(np.sign(gold_corr) == np.sign(pred_corr), np.sign(gold_corr))
        neg_sign_matching_bool = np.logical_and(np.sign(gold_corr) == np.sign(pred_corr), np.logical_not(np.sign(gold_corr)))
        pos_not_matching = np.logical_and(np.sign(gold_corr) != np.sign(pred_corr), np.sign(gold_corr))
        neg_not_matching = np.logical_and(np.sign(gold_corr) != np.sign(pred_corr), np.logical_not(np.sign(gold_corr)))
        tp = np.sum(pos_sign_matching_bool)
        tn = np.sum(neg_sign_matching_bool)
        fp = np.sum(pos_not_matching)
        fn = np.sum(neg_not_matching)
	return total_loss, x_predictions, np.corrcoef(np.exp(x_predictions), x_label)[0, 1], y_predictions, np.corrcoef(np.exp(y_predictions), y_label)[0, 1], xy_predictions, np.corrcoef(np.exp(xy_predictions), xy_label)[0, 1], cpr_predictions, np.corrcoef(np.exp(cpr_predictions), cpr_xy_label)[0, 1], Probability.kl_divergence_batch(cpr_predictions, cpr_xy_label), cpr_predictions_rev, corr_loss, tp, tn, fp, fn

    def train(self, train1, train2, train_xlabels, train_ylabels, train_xylabels, train_cpr_xylabels, dev1, dev2, dev_xlabels, dev_ylabels, dev_xylabels, dev_cpr_xylabels, train_lens1=None, train_lens2=None, dev_lens1=None, dev_lens2=None):
        total_loss = 0.0
        count = 0
        data_size = len(train1)
        indices = range(data_size)
        random.seed(20160408)
        random.shuffle(indices)
        # print(indices)
        print_step = data_size / 10
        for step in range(int(data_size/self.batch_size)):
            # print('step', step)
            # while (step + 1) * self.batch_size >= count:
            #     print(str(count) + " "),
            #     count += print_step
            r_ind = indices[(step * self.batch_size):((step + 1) * self.batch_size)]
            batch_xlabels = train_xlabels[r_ind]
            batch_ylabels = train_ylabels[r_ind]
            batch_xylabels = train_xylabels[r_ind]
            batch_cpr_xylabels = train_cpr_xylabels[r_ind]
            batch_lens1 = train_lens1[r_ind]
            batch_lens2 = train_lens2[r_ind]
            train1_sub = []
            train2_sub = []
            for i in r_ind:
                train1_sub.append(train1[i])
                train2_sub.append(train2[i])
            training_1_pair = Sparse.pad_tensor(train1_sub, self.maxlength)
            training_2_pair = Sparse.pad_tensor(train2_sub, self.maxlength)
            feed_dict = {self.dataset1: training_1_pair, self.dataset2: training_2_pair, self.x_labels: batch_xlabels, self.y_labels: batch_ylabels, self.xy_labels: batch_xylabels, self.cpr_xy_labels: batch_cpr_xylabels, self.lengths1:batch_lens1, self.lengths2:batch_lens2, self.dropout_ph: self.dropout}
            _, l, x_pred, y_pred, xy_pred, cpr_xy_pred, x_loss, cpr_loss, grad = self.sess.run([self.optimizer, self.mean_loss, self.x_predicted, self.y_predicted, self.xy_predicted, self.cpr_xy_predicted, self.x_loss, self.cpr_loss, self.gradient], feed_dict=feed_dict)
            # l, x_pred, y_pred, xy_pred, cpr_xy_pred, x_loss, cpr_loss = self.sess.run([self.mean_loss, self.x_predicted, self.y_predicted, self.xy_predicted, self.cpr_xy_predicted, self.x_loss, self.cpr_loss], feed_dict=feed_dict)
            # print('finished')
            # print('l', l)
            # for gv, v in grad:
            #     print('*'*50)
            #     if len(gv.shape)>1:
            #         print(gv[:3,:3], v[:3,:3])
            #     else:
            #         print(gv[:3], v[:3])
            # print('x_pred', x_pred)
            # print('y_pred', y_pred)
            # print('xy_pred', xy_pred)
            # print('cpr_xy_pred', cpr_xy_pred)
            total_loss += l
            # if (step % 20000 == 0):
            #     print(step)
            #     dev_loss, x_pred, x_corr, y_pred, y_corr, xy_pred, xy_corr, cpr_xy_pred, cpr_xy_corr, dev_cpr_kl, cpr_xy_pred_rev, square_loss, sign_matching= self.eval(dev1, dev2, dev_xlabels, dev_ylabels, dev_xylabels, dev_cpr_xylabels, dev_lens1, dev_lens2)
            #     print('\nTRAIN: loss %f  DEV: loss: %f  X corr %f  Y corr %f  XY corr %f  CPR corr %f  Dev CPR KL %f Dev Square Loss %f Dev Sign Matching %f' % (l, dev_loss, x_corr, y_corr, xy_corr, cpr_xy_corr, dev_cpr_kl, square_loss, sign_matching))
        
        dev_loss, x_pred, x_corr, y_pred, y_corr, xy_pred, xy_corr, cpr_xy_pred, cpr_xy_corr, dev_cpr_kl, cpr_xy_pred_rev, square_loss, tp, tn, fp, fn = self.eval(dev1, dev2, dev_xlabels, dev_ylabels, dev_xylabels, dev_cpr_xylabels, dev_lens1, dev_lens2)
        print('Finished One Epoch \nTRAIN: loss %f  DEV: loss: %f  X corr %f  Y corr %f  XY corr %f  CPR corr %f  Dev CPR KL %f ' % (total_loss, dev_loss, x_corr, y_corr, xy_corr, cpr_xy_corr, dev_cpr_kl))
        print('Dev Square Loss %f Dev Sign Matching tp %f, tn %f, fp %f, fn %f' % (square_loss, tp, tn, fp, fn))
        return dev_cpr_kl
