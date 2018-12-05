import random
import numpy as np
random.seed(20160408)


class Training:

    def __init__(self, sess, optimizer, loss, prediction, prob_feat, sent1, sent2, labels, lengths1, lengths2, batch_size, dropout_lstm_ph, dropout_lstm, dropout_entail_ph, dropout_entail):
        self.sess = sess
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.prediction = prediction
        self.prob_feat = prob_feat
        self.sent1 = sent1
        self.sent2 = sent2
        self.labels = labels
        self.lengths1 = lengths1
        self.lengths2 = lengths2
        self.dropout_lstm_ph = dropout_lstm_ph
        self.dropout_lstm = dropout_lstm
        self.dropout_entail_ph = dropout_entail_ph
        self.dropout_entail = dropout_entail

    def eval(self, data_feat, data1, data2, label, lens1, lens2):
        total_loss = 0
        predictions = []
        indices = range(data_feat.shape[0])
        random.shuffle(indices)
        padded = False
        for i in range(data_feat.shape[0]/self.batch_size + 1):
            while self.batch_size*(i+1) > len(indices):
                indices.append(indices[-1])
                padded = True
            ind = indices[self.batch_size*i:self.batch_size*(i+1)]
            feat_batch = data_feat[ind]
            d1_batch = data1[ind]
            d2_batch = data2[ind]
            l = label[ind]
            l1_entail = lens1[ind]
            l2_entail = lens2[ind]
            loss, p = self.sess.run([self.loss, self.prediction], feed_dict={self.prob_feat:feat_batch, self.sent1:d1_batch, self.sent2:d2_batch, self.labels:l, self.lengths1:l1_entail, self.lengths2:l2_entail, self.dropout_lstm_ph: 1.0, self.dropout_entail_ph: 1.0})
            total_loss += loss
            predictions.extend(p)
        if padded:
            while len(predictions) > data_feat.shape[0]:
                predictions = predictions[:-1]
        return total_loss, 100.0*sum(predictions)/len(predictions), predictions

    def train(self, train_feat, train_sent1, train_sent2, train_labels, dev_feat, dev_sent1, dev_sent2, dev_labels, train_lens1, train_lens2, dev_lens1, dev_lens2):
        total_loss = 0.0
        count = 0
        data_size = len(train_feat)
        indices = range(data_size)
        random.shuffle(indices)
        print_step = data_size / 10
        for step in range(data_size/self.batch_size):
            while (step + 1) * self.batch_size >= count:
                print(str(count) + " "),
                count += print_step
            r_ind = indices[(step * self.batch_size):((step + 1) * self.batch_size)]
            feed_dict = {self.prob_feat: train_feat[r_ind], self.sent1: train_sent1[r_ind], self.sent2: train_sent2[r_ind], self.labels: train_labels[r_ind], self.lengths1:train_lens1[r_ind], self.lengths2:train_lens2[r_ind], self.dropout_entail_ph: self.dropout_entail, self.dropout_lstm_ph: self.dropout_lstm}
            _, l = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
            total_loss += l
        loss, dev_correct, _ = self.eval(dev_feat, dev_sent1, dev_sent2, dev_labels, dev_lens1, dev_lens2)
        print('\nTrain: loss %f  Dev: loss %f  accuracy %f' % (total_loss, loss, dev_correct))
        return dev_correct
