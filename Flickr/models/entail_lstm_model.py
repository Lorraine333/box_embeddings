# Train LSTM to predict entailment from SNLI data

import random
import sys
import time
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.nan)
from util.Layer import Layers
from util.Train_entail_lstm import Training
from util.DataLoader import DataLoader
Data = DataLoader()
Layer = Layers()
sys.path.append(".")


def run(**args):

    tf.reset_default_graph()
    random.seed(20160408)
    tf.set_random_seed(20160408)
    start_time = time.time()
    graph = tf.get_default_graph()

    # Read Training/Dev/Test data
    data_dir = 'data/' + args['data_dir'] + '/'
    np_matrix, index = Data.read_glove_vectors('data/' + args['data_dir'] + '/' + args['vector_file'])
    num_classes = 3

    # entailment data
    train_1, train_2, train_labels, train_lens1, train_lens2, maxlength = Data.read_entail(data_dir + args['train_entail_data'], index)
    dev_1, dev_2, dev_labels, dev_lens1, dev_lens2, _ = Data.read_entail(data_dir + args['dev_entail_data'], index)
    test_1, test_2, test_labels, test_lens1, test_lens2, _ = Data.read_entail(data_dir + args['test_entail_data'], index)

    training_1 = Data.pad_tensor(train_1, maxlength)
    development_1 = Data.pad_tensor(dev_1, maxlength)
    testing_1 = Data.pad_tensor(test_1, maxlength)
    training_2 = Data.pad_tensor(train_2, maxlength)
    development_2 = Data.pad_tensor(dev_2, maxlength)
    testing_2 = Data.pad_tensor(test_2, maxlength)
    training_labels = Data.pad_labels(train_labels)
    development_labels = Data.pad_labels(dev_labels)
    testing_labels = Data.pad_labels(test_labels)

    # Input -> LSTM -> Outstate
    dropout_ph = tf.placeholder(tf.float32)
    inputs1 = tf.placeholder(tf.int32, [args['batch_size'], maxlength])
    inputs2 = tf.placeholder(tf.int32, [args['batch_size'], maxlength])
    lengths1 = tf.placeholder(tf.int32, [args['batch_size']])
    lengths2 = tf.placeholder(tf.int32, [args['batch_size']])
    labels = tf.placeholder(tf.float32, [args['batch_size'], num_classes])

    with tf.variable_scope('lstm'):
        # LSTM
        embeddings = tf.Variable(np_matrix, dtype=tf.float32, trainable=False)

        output_dim = 2 * args['hidden_dim']

        lstm = tf.nn.rnn_cell.LSTMCell(args['hidden_dim'], state_is_tuple=True)
        lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=dropout_ph)

        Wemb1 = tf.nn.embedding_lookup(embeddings, inputs1)
        Wemb2 = tf.nn.embedding_lookup(embeddings, inputs2)
        outputs1, fstate1 = tf.nn.dynamic_rnn(lstm, Wemb1, sequence_length=lengths1, dtype=tf.float32)

        tf.get_variable_scope().reuse_variables()
        outputs2, fstate2 = tf.nn.dynamic_rnn(lstm, Wemb2, sequence_length=lengths2, dtype=tf.float32)

        output_layer1 = Layer.W(2 * args['hidden_dim'], output_dim, 'Output1')
        output_bias1  = Layer.b(output_dim, 'OutputBias1')
        logits1 = tf.nn.dropout(tf.nn.tanh(tf.matmul(tf.concat(1, [fstate1[0], fstate2[0]]), output_layer1) + output_bias1), dropout_ph)

        output_layer2 = Layer.W(output_dim, output_dim, 'Output2')
        output_bias2  = Layer.b(output_dim, 'OutputBias2')
        logits2 = tf.nn.dropout(tf.nn.tanh(tf.matmul(logits1, output_layer2) + output_bias2), dropout_ph)

        output_layer3 = Layer.W(output_dim, num_classes, 'Output3')
        output_bias3  = Layer.b(num_classes, 'OutputBias3')
        logits3 = tf.matmul(logits2, output_layer3) + output_bias3

        prediction = tf.argmax(logits3, 1)
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits3, labels))

    optimizer = tf.train.AdamOptimizer(0.001, 0.9)
    varlist = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='lstm')
    train_op = optimizer.minimize(loss, var_list=varlist)

    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session() as sess:
        if args['method'] == 'train':

            sess.run(tf.initialize_all_variables())
            Trainer = Training(sess, train_op, loss, prediction, inputs1, inputs2, labels, lengths1, lengths2, args['batch_size'], dropout_ph, args['dropout_lstm'])

            best_dev = 0.0
            for e in range(args['num_epochs']):
                print("Outer epoch %d" % e)
                correct = Trainer.train(training_1, training_2, training_labels, development_1, development_2, development_labels, train_lens1, train_lens2, dev_lens1, dev_lens2)
                if correct > best_dev:
                    save_path = saver.save(sess, "tmp/"+args['exp_name_lstm']+"_best.ckpt")
                    print("Best model saved in file: %s" % save_path)
                    best_dev = correct
                print("--- %s seconds ---" % (time.time() - start_time))
                save_path = saver.save(sess, "tmp/"+args['exp_name_lstm']+"_"+str(e)+".ckpt")
                print("Model saved in file: %s" % save_path)

        elif args['method'] == 'test':
            saver.restore(sess, "tmp/"+args['exp_name_lstm']+".ckpt")
            Trainer = Training(sess, train_op, loss, prediction, inputs1, inputs2, labels, lengths1, lengths2, args['batch_size'], dropout_ph, args['dropout_lstm'])
            _, _, _, accuracy = Trainer.eval(testing_1, testing_2, testing_labels, test_lens1, test_lens2)
            print("Accuracy: %f" % accuracy)
            print("--- %s seconds ---" % (time.time() - start_time))
