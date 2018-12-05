import random
import numpy as np

random.seed(20160408)

class Training:

    def __init__(self, sess, optimizer, loss, prediction, sent1, sent2, labels, lengths1, lengths2, batch_size, dropout_ph, dropout):
        self.sess = sess
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.prediction = prediction
        self.sent1 = sent1
        self.sent2 = sent2
        self.labels = labels
        self.lengths1 = lengths1
        self.lengths2 = lengths2
        self.dropout_ph = dropout_ph
        self.dropout = dropout

    def eval(self, data1, data2, labels, lens1, lens2):
        predictions = []
        total_loss = 0
        indices = range(data1.shape[0])
        padded = False
        for i in range((data1.shape[0]/self.batch_size) + 1):
            while self.batch_size*(i+1) > len(indices):
                indices.append(indices[-1])
                padded = True
            ind = indices[self.batch_size*i:self.batch_size*(i+1)]
            D1 = data1[ind]
            D2 = data2[ind]
            L = labels[ind]
            l1 = lens1[ind]
            l2 = lens2[ind]
            feed_dict = {self.sent1: D1, self.sent2: D2, self.lengths1: l1,
                         self.lengths2: l2, self.dropout_ph: 1.0, self.labels: L}
            loss, p = self.sess.run([self.loss, self.prediction], feed_dict=feed_dict)
            predictions.extend(p)
            total_loss += loss
        if padded:
            while len(predictions) > data1.shape[0]:
                predictions = predictions[:-1]
        correct = np.equal(predictions, np.argmax(labels, 1))
        return total_loss, predictions, correct, 100.0*sum(correct)/len(correct)

    def train(self, train_sent1, train_sent2, train_labels, dev_sent1, dev_sent2, dev_labels, train_lens1, train_lens2, dev_lens1, dev_lens2):
        total_loss = 0.0
        count = 0
        data_size = train_sent1.shape[0]
        indices = range(data_size)
        random.shuffle(indices)
        print_step = data_size / 10
        for step in range(data_size/self.batch_size):
            while (step + 1) * self.batch_size >= count:
                print(str(count) + " "),
                count += print_step
            r_ind = indices[(step * self.batch_size):((step + 1) * self.batch_size)]
            batch_labels = train_labels[r_ind]
            batch_lens1 = train_lens1[r_ind]
            batch_lens2 = train_lens2[r_ind]
            feed_dict = {self.sent1: train_sent1[r_ind], self.sent2: train_sent2[r_ind], self.labels: batch_labels, self.lengths1:batch_lens1, self.lengths2:batch_lens2, self.dropout_ph: self.dropout}
            _, l = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
            total_loss += l
        dev_loss, _, dev_correct, dev_accuracy = self.eval(dev_sent1, dev_sent2, dev_labels, dev_lens1,
                                                           dev_lens2)
        print('\nTotal train loss %f  Dev loss %f  Dev accuracy %f' % (total_loss, dev_loss, dev_accuracy))
        return dev_accuracy