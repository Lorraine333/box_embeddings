# Train entailment prediction model by appending predicted probability
# features from file to output of previously trained LSTM model

import random
import sys
import time
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.nan)
from util.Layer import Layers
from util.Train_entail_prob import Training
from util.DataLoader import DataLoader
Data = DataLoader()
Layer = Layers()
sys.path.append(".")


def run(**args):

    random.seed(20160408)
    tf.set_random_seed(20160408)
    graph = tf.get_default_graph()
    start_time = time.time()

    # Read Training/Dev/Test data
    data_dir = 'data/' + args['data_dir'] + '/'
    np_matrix, index = Data.read_glove_vectors('data/' + args['data_dir'] + '/' + args['vector_file'])
    num_classes = 3

    train_features, train_cpr_labels, num_features = Data.read_feat_probabilities(data_dir + args['train_prob_data'])
    dev_features, dev_cpr_labels, _ = Data.read_feat_probabilities(data_dir + args['dev_prob_data'])
    test_features, test_cpr_labels, _ = Data.read_feat_probabilities(data_dir + args['test_prob_data'])

    # entailment data
    train_1_entail, train_2_entail, train_entail_labels, train_lens1_entail, train_lens2_entail, maxlength = Data.read_entail(
        data_dir + args['train_entail_data'], index)
    dev_1_entail, dev_2_entail, dev_entail_labels, dev_lens1_entail, dev_lens2_entail, _ = Data.read_entail(
        data_dir + args['dev_entail_data'], index)
    test_1_entail, test_2_entail, test_entail_labels, test_lens1_entail, test_lens2_entail, _ = Data.read_entail(
        data_dir + args['test_entail_data'], index)

    training_1_entail = Data.pad_tensor(train_1_entail, maxlength)
    development_1_entail = Data.pad_tensor(dev_1_entail, maxlength)
    testing_1_entail = Data.pad_tensor(test_1_entail, maxlength)
    training_2_entail = Data.pad_tensor(train_2_entail, maxlength)
    development_2_entail = Data.pad_tensor(dev_2_entail, maxlength)
    testing_2_entail = Data.pad_tensor(test_2_entail, maxlength)
    training_entail_labels = Data.pad_labels(train_entail_labels)
    development_entail_labels = Data.pad_labels(dev_entail_labels)
    testing_entail_labels = Data.pad_labels(test_entail_labels)

    # Input -> LSTM -> Outstate
    dropout_lstm_ph = tf.placeholder(tf.float32)
    dropout_entail_ph = tf.placeholder(tf.float32)
    input_feat = tf.placeholder(tf.float32, [args['batch_size'], num_features])

    inputs1_entail = tf.placeholder(tf.int32, [args['batch_size'], maxlength])
    inputs2_entail = tf.placeholder(tf.int32, [args['batch_size'], maxlength])
    lengths1_entail = tf.placeholder(tf.int32, [args['batch_size']])
    lengths2_entail = tf.placeholder(tf.int32, [args['batch_size']])
    entail_labels = tf.placeholder(tf.float32, [args['batch_size'], num_classes])

    output_dim = 2 * args['hidden_dim']
    with tf.variable_scope('lstm'):
        # LSTM
        embeddings = tf.Variable(np_matrix, dtype=tf.float32)

        lstm = tf.nn.rnn_cell.LSTMCell(args['hidden_dim'], state_is_tuple=True)
        lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=dropout_lstm_ph)

        Wemb1 = tf.nn.embedding_lookup(embeddings, inputs1_entail)
        Wemb2 = tf.nn.embedding_lookup(embeddings, inputs2_entail)
        outputs1, fstate1 = tf.nn.dynamic_rnn(lstm, Wemb1, sequence_length=lengths1_entail, dtype=tf.float32)

        tf.get_variable_scope().reuse_variables()
        outputs2, fstate2 = tf.nn.dynamic_rnn(lstm, Wemb2, sequence_length=lengths2_entail, dtype=tf.float32)

        output_layer1 = Layer.W(2 * args['hidden_dim'], output_dim, 'Output1')
        output_bias1 = Layer.b(output_dim, 'OutputBias1')
        logits1 = tf.nn.dropout(
            tf.nn.tanh(tf.matmul(tf.concat(1, [fstate1[0], fstate2[0]]), output_layer1) + output_bias1),
            dropout_lstm_ph)

        output_layer2 = Layer.W(output_dim, output_dim, 'Output2')
        output_bias2 = Layer.b(output_dim, 'OutputBias2')
        logits2 = tf.nn.dropout(tf.nn.tanh(tf.matmul(logits1, output_layer2) + output_bias2), dropout_lstm_ph)

    with tf.variable_scope('entail'):

        output_layer4 = Layer.W(num_features + output_dim, output_dim, 'Output4')
        output_bias4  = Layer.b(output_dim, 'OutputBias4')
        logits4 = tf.nn.dropout(tf.nn.tanh(tf.matmul(tf.concat(1, [input_feat, logits2]), output_layer4) + output_bias4), dropout_entail_ph)

        output_layer5 = Layer.W(output_dim, output_dim, 'Output5')
        output_bias5  = Layer.b(output_dim, 'OutputBias4')
        logits5 = tf.nn.dropout(tf.nn.tanh(tf.matmul(logits4, output_layer5) + output_bias5), dropout_entail_ph)

        output_layer6 = Layer.W(output_dim, num_classes, 'Output6')
        output_bias6  = Layer.b(num_classes, 'OutputBias2')
        logits6 = tf.matmul(logits5, output_layer6) + output_bias6

        class_loss_ff = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits6, entail_labels))
        entail_prediction_ff = tf.equal(tf.argmax(logits6, 1), tf.argmax(entail_labels, 1))


    ## Learning ##
    varlist_lstm = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='lstm')

    optimizer = tf.train.AdamOptimizer(0.001, 0.9)
    varlist_entail = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='entail')
    train_op_entail = optimizer.minimize(class_loss_ff, var_list=varlist_entail)

    # Add this to session to see cpu/gpu placement:
    saver_lstm = tf.train.Saver(var_list=varlist_lstm, max_to_keep=100)
    saver_entail = tf.train.Saver(max_to_keep=100)
    with tf.Session() as sess:
        if args['method'] == 'train':
            sess.run(tf.initialize_all_variables())
            # load best LSTM model
            saver_lstm.restore(sess, "tmp/"+args['exp_name_lstm']+".ckpt")
            Trainer = Training(sess, train_op_entail, class_loss_ff, entail_prediction_ff, input_feat, inputs1_entail, inputs2_entail, entail_labels, lengths1_entail, lengths2_entail, args['batch_size'], dropout_lstm_ph, args['dropout_lstm'], dropout_entail_ph, args['dropout_cpr'])

            best_dev = 0.0
            for e in range(args['num_epochs']):
                print("Outer epoch %d" % e)
                correct = Trainer.train(train_features, training_1_entail, training_2_entail, training_entail_labels, dev_features, development_1_entail, development_2_entail, development_entail_labels, train_lens1_entail, train_lens2_entail, dev_lens1_entail, dev_lens2_entail)
                if correct > best_dev:
                    save_path = saver_entail.save(sess, "tmp/"+args['exp_name_full']+"_best.ckpt")
                    print("Best model saved in file: %s" % save_path)
                    best_dev = correct
                print("--- %s seconds ---" % (time.time() - start_time))
                save_path = saver_entail.save(sess, "tmp/"+args['exp_name_full']+"_"+str(e)+".ckpt")
                print("Model saved in file: %s" % save_path)
        elif args['method'] == 'test':
            saver_entail.restore(sess, "tmp/"+args['exp_name_full']+".ckpt")
            Trainer = Training(sess, train_op_entail, class_loss_ff, entail_prediction_ff, input_feat, inputs1_entail, inputs2_entail, entail_labels, lengths1_entail, lengths2_entail, args['batch_size'], dropout_lstm_ph, args['dropout_lstm'], dropout_entail_ph, args['dropout_cpr'])
            _, accuracy, predictions = Trainer.eval(test_features, testing_1_entail, testing_2_entail, testing_entail_labels, test_lens1_entail, test_lens2_entail)
            data_name = args['test_entail_data'].split(".")[0]
            out_file = open(args['data_dir'] + data_name + "_predicted_entail.txt", "w")
            count = 0
            for idx, pred in enumerate(predictions):
                count += 1
                if count % 1000 == 0:
                    print(count)
                s1 = [str(a) for a in testing_1_entail[idx]]
                s2 = [str(a) for a in testing_2_entail[idx]]
                out_file.write("%s\t%s\t%s\t%s\n" % (testing_entail_labels[idx], pred, " ".join(s1), " ".join(s2)))
            out_file.close()
            print("Accuracy: %f" % (100.0*sum(predictions)/len(predictions)))
            print("--- %s seconds ---" % (time.time() - start_time))

