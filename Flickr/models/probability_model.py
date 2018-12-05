# train model to predict conditional probabilities and p(x) individual phrase probabilities

import random
import sys
from tensorflow.contrib.rnn.python.ops import lstm_ops
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.nan)
from util.Layer import Layers
from util.Train_probability import Training
from util.DataLoader import DataLoader
Sparse = DataLoader()
Layer = Layers()
from util import cube_exp_prob
from util import Probability
from util import Bilinear
from util import corr_prob
import time

sys.path.append(".")

def get_lstm_input(hidden_dim, embeddings, inputs1, inputs2, lengths1, lengths2, dropout, lstm):
    # lstm produce sentence representation, used in all models
    Wemb1 = tf.nn.embedding_lookup(embeddings, inputs1)
    Wemb2 = tf.nn.embedding_lookup(embeddings, inputs2)
    # Wemb1 = tf.Print(Wemb1, [Wemb1], 'term1 embedding')
    lstm_output, fstate1 = tf.nn.dynamic_rnn(lstm, Wemb1, sequence_length=lengths1, dtype=tf.float32)
    # fstate1 = tf.Print(fstate1, [fstate1[0], Wemb1], 'lstm_term1')
    tf.get_variable_scope().reuse_variables()
    lstm_output, fstate2 = tf.nn.dynamic_rnn(lstm, Wemb2, sequence_length=lengths2, dtype=tf.float32)
    return fstate1, fstate2

def run(**args):
    tf.reset_default_graph()
    tf.set_random_seed(20160408)
    random.seed(20160408)
    exp_name = 'train_data'+str(args['train_data'])+'batch_size'+str(args['batch_size'])+\
               '_dropout'+str(args['dropout'])+'_epochs'+str(args['num_epochs'])+\
               '_px'+str(args['lambda_px'])+'_cpr'+str(args['lambda_cpr'])+'_lr'+str(args['learning_rate'])+\
               '_mode'+args['mode']+'_cube'+args['cube']+'_layer1_init'+args['layer1_init']+\
               '_layer2_init'+args['layer2_init']+'lambda_v'+str(args['lambda_value'])+'delta_acti'+str(args['delta_acti'])+\
               'delta_initv'+str(args['layer2_init_value'])+'loss'+str(args['loss'])+'seed'+str(args['lstm_seed'])
    print('parameters', exp_name)
    start_time = time.time()

    # Read Training/Dev/Test data
    data_dir = './data/' + args['data_dir'] + '/'
    np_matrix, index = Sparse.read_glove_vectors('./data/' + args['data_dir'] + '/' + args['vector_file'])

    if args['method'] == 'train':
        train_1, train_2, train_xlabels, train_ylabels, train_xylabels, train_cpr_labels, train_lens1, train_lens2, maxlength, train_phrase1, train_phrase2, train_labels = Sparse.gzread_cpr(data_dir + args['train_data'], index)
        dev_1, dev_2, dev_xlabels, dev_ylabels, dev_xylabels, dev_cpr_labels, dev_lens1, dev_lens2, _, dev_phrase1, dev_phrase2, dev_labels            = Sparse.gzread_cpr(data_dir + args['dev_data'], index)
        test_1, test_2, test_xlabels, test_ylabels, test_xylabels, test_cpr_labels, test_lens1, test_lens2, _, test_phrase1, test_phrase2, test_labels           = Sparse.gzread_cpr(data_dir + args['test_data'], index)
    elif args['method'] == 'test': # predict probabilities on test file (probability format)
        test_1, test_2, test_xlabels, test_ylabels, test_xylabels, test_cpr_labels, test_lens1, test_lens2, maxlength, test_phrase1, test_phrase2, test_labels           = Sparse.gzread_cpr(data_dir + args['test_data'], index)

    graph = tf.get_default_graph()

    # Input -> LSTM -> Outstate
    dropout = tf.placeholder(tf.float32)
    inputs1 = tf.placeholder(tf.int32, [args['batch_size'], None]) # batch size * length
    inputs2 = tf.placeholder(tf.int32, [args['batch_size'], None]) # batch size * length
    x_labels = tf.placeholder(tf.float32, [args['batch_size']])
    y_labels = tf.placeholder(tf.float32, [args['batch_size']])
    xy_labels = tf.placeholder(tf.float32, [args['batch_size']], name='xy_labels')
    cpr_labels = tf.placeholder(tf.float32, [args['batch_size']])
    lengths1 = tf.placeholder(tf.int32, [args['batch_size']])
    lengths2 = tf.placeholder(tf.int32, [args['batch_size']])

    # RNN
    with tf.variable_scope('prob', initializer = tf.variance_scaling_initializer(seed = args['lstm_seed'])):
    # with tf.variable_scope('prob', initializer = tf.variance_scaling_initializer(seed = 1012)):
    # with tf.variable_scope('prob'):


        # LSTM
        embeddings = tf.Variable(np_matrix, dtype=tf.float32, trainable=False)
        bilinear_matrix = tf.Variable(tf.orthogonal_initializer(seed=20160408)(shape = (args['hidden_dim'], args['hidden_dim'])), trainable=True)

        # lstm = tf.contrib.rnn.LSTMCell(args['hidden_dim'], state_is_tuple=True)

        # poe model, including kl loss and correlation loss. Both contains kl between marginals
        if args['mode'] == 'poe':
            lstm = lstm_ops.LSTMBlockCell(args['hidden_dim'])
            lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=dropout)
	    fstate1, fstate2 = get_lstm_input(args['hidden_dim'], embeddings, inputs1, inputs2, lengths1, lengths2, dropout,lstm)
            joint_predicted, x_predicted, y_predicted, cpr_predicted, cpr_predicted_reverse = Probability.poe_model(args, fstate1, fstate2)

            if args['loss'] == 'kl':
                print('poe_kl')
                cpr_loss = Probability.kl_loss(args['batch_size'], cpr_predicted, cpr_labels)

            elif args['loss'] == 'corr':
                print('poe_correlation')
                cpr_loss = corr_prob.corr_loss(x_predicted, y_predicted, joint_predicted, x_labels, y_labels, xy_labels)
            else:
                print 'invalid loss'

        elif 'cube' in args['mode']:
            lstm = lstm_ops.LSTMBlockCell(args['hidden_dim'])
            lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=dropout)
            # init first feed forward network parameters
            W1 = Layer.W(args['hidden_dim'], args['output_dim'], 'Output')
            b1 = Layer.layer1_bias(args['output_dim'], args['layer1_init'],  -5.0, 'layer1_b')
            # get lstm output
            fstate1, fstate2 = get_lstm_input(args['hidden_dim'], embeddings, inputs1, inputs2, lengths1, lengths2, dropout, lstm)
            # init second feed forward network parameters
            W2 = Layer.W(args['hidden_dim'], args['output_dim'], 'Output1')
            b2 = Layer.layer2_bias(args['output_dim'], args['layer2_init'], args['layer2_init_value'], 'layer2_b')
            # get box embedding via lstm output
            t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed = cube_exp_prob.box_model_embed(args, fstate1, fstate2, W1, b1, W2, b2)
            join_min, join_max, meet_min, meet_max, not_have_meet = cube_exp_prob.box_model_params(t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed)
            joint_predicted, x_predicted, y_predicted, cpr_predicted, cpr_predicted_reverse = cube_exp_prob.box_prob(join_min, join_max, meet_min, meet_max, not_have_meet, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed)

            if args['loss'] == 'corr':
                print('cube_correlation')
                # calculate correlation loss, it's different when we need to use lower bound and xy_label greater than 0.0
                cpr_loss = cube_exp_prob.slicing_where(condition = not_have_meet & (xy_labels > 0),
                                                       full_input = ([join_min, join_max, meet_min, meet_max, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed, x_labels, y_labels, xy_labels, not_have_meet]),
                                                       true_branch = lambda x: corr_prob.lambda_upper_bound(*x),
                                                       false_branch = lambda x: corr_prob.lambda_corr_loss(*x))

            elif  args['loss'] == 'kl':
                # for training
                # calculate log conditional probability for positive examplse, and negative upper bound if two things are disjoing
                train_cpr_predicted = cube_exp_prob.slicing_where(condition = not_have_meet,
                                                                  full_input = ([join_min, join_max, meet_min, meet_max, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed]),
                                                                  true_branch = lambda x: cube_exp_prob.lambda_batch_log_upper_bound(*x),
                                                                  # true_branch= lambda  x: cube_exp_prob.lambda_batch_log_upper_bound_version2(*x),
                                                                  false_branch = lambda x: cube_exp_prob.lambda_batch_log_cube_measure(*x))
                # calculate log(1-p) if overlap, 0 if no overlap
                onem_cpr_predicted = cube_exp_prob.slicing_where(condition = not_have_meet,
                                                                 full_input = tf.tuple([join_min, join_max, meet_min, meet_max, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed]),
                                                                 true_branch = lambda x: cube_exp_prob.lambda_zero_log_upper_bound(*x),
                                                                 false_branch = lambda x: cube_exp_prob.lambda_batch_log_cond_cube_measure(*x))

                whole_cpr_predicted = tf.concat([tf.expand_dims(train_cpr_predicted, 1), tf.expand_dims(onem_cpr_predicted, 1)], 1)
                cpr_loss = tf.nn.softmax_cross_entropy_with_logits(logits= whole_cpr_predicted, labels= cube_exp_prob.create_distribution(cpr_labels, args['batch_size']))


        elif args['mode'] == 'bilinear':
            print('bilinear')
	    lstm = lstm_ops.LSTMBlockCell(args['hidden_dim'])
            lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=dropout)
            fstate1, fstate2 = get_lstm_input(args['hidden_dim'], embeddings, inputs1, inputs2, lengths1, lengths2, dropout, lstm)
            joint_predicted, x_predicted, y_predicted, cpr_predicted, cpr_predicted_reverse = Bilinear.bilinear_model(args, fstate1, fstate2, bilinear_matrix)
            cpr_loss = tf.nn.softmax_cross_entropy_with_logits(logits= Bilinear.create_log_distribution(cpr_predicted, args['batch_size']), labels= Bilinear.create_distribution(cpr_labels, args['batch_size']))
        else:
            print('mode is wrong')

        x_loss = tf.nn.softmax_cross_entropy_with_logits(logits= cube_exp_prob.create_log_distribution(x_predicted, args['batch_size']), labels= cube_exp_prob.create_distribution(x_labels, args['batch_size']))
        y_loss = tf.nn.softmax_cross_entropy_with_logits(logits= cube_exp_prob.create_log_distribution(y_predicted, args['batch_size']), labels= cube_exp_prob.create_distribution(y_labels, args['batch_size']))
        mean_loss = tf.reduce_mean(args['lambda_px'] * (x_loss + y_loss) + args['lambda_cpr'] * cpr_loss)

    ## Learning ##
    optimizer = tf.train.AdamOptimizer(args['learning_rate'])
    varlist = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='prob')
    gradient = optimizer.compute_gradients(mean_loss, var_list= varlist)
    train_op = optimizer.apply_gradients(gradient)

    # train_op = optimizer.minimize(mean_loss, var_list=varlist)

    tf.set_random_seed(20160408)
    saver = tf.train.Saver(max_to_keep=10)
    with tf.Session() as sess:
        tf.set_random_seed(20160408)
        if args['method'] == 'train':
            sess.run(tf.global_variables_initializer())
            Trainer = Training(sess, train_op, mean_loss, x_loss, cpr_loss, x_predicted, y_predicted, joint_predicted, cpr_predicted, cpr_predicted_reverse, inputs1, inputs2, x_labels, y_labels, xy_labels, cpr_labels, lengths1, lengths2, args['batch_size'], maxlength, dropout, args['dropout'], gradient)
            best_dev = float('inf')
            for e in range(args['num_epochs']):
                print("Outer epoch %d" % e)
                kl_div = Trainer.train(train_1, train_2, train_xlabels, train_ylabels, train_xylabels, train_cpr_labels, dev_1, dev_2, dev_xlabels, dev_ylabels, dev_xylabels, dev_cpr_labels, train_lens1, train_lens2, dev_lens1, dev_lens2)
                if kl_div < best_dev:
                    save_path = saver.save(sess, "./tmp/" + exp_name +"_best.ckpt")
                    print("Best model saved in file: %s" % save_path)
                    best_dev = kl_div
                print("--------------- %s seconds ---------------" % (time.time() - start_time))
                save_path = saver.save(sess, "./tmp/" + exp_name + "_"+str(e)+".ckpt")
                print("Model saved in file: %s" % save_path)
        elif args['method'] == 'test':
            saver.restore(sess, "./tmp/" + exp_name + "_best.ckpt")
            Trainer = Training(sess, train_op, mean_loss, x_loss, cpr_loss, x_predicted, y_predicted, joint_predicted, cpr_predicted, cpr_predicted_reverse, inputs1, inputs2, x_labels, y_labels, xy_labels, cpr_labels, lengths1, lengths2, args['batch_size'], maxlength, dropout, args['dropout'], gradient)
            test_loss, x_pred, x_corr, x_kl, y_pred, y_corr, y_kl, xy_pred, xy_corr, cpr_pred, cpr_xy_corr, cpr_kl, cpr_pred_reverse, corr_loss, tp, tn, fp, fn = Trainer.eval(test_1, test_2, test_xlabels, test_ylabels, test_xylabels, test_cpr_labels, test_lens1, test_lens2)
            #test_loss, x_pred, x_corr, y_pred, y_corr, xy_pred, xy_corr, cpr_pred, cpr_xy_corr, cpr_kl, cpr_pred_reverse, corr_loss, tp, tn, fp, fn = Trainer.eval(test_1, test_2, test_xlabels, test_ylabels, test_xylabels, test_cpr_labels, test_lens1, test_lens2)
	    out_file = open(data_dir +'after_respon_'+exp_name + "_" + args['test_data'].split(".")[0] + "_pred_prob.txt", "w")
            print('tp', tp)
            print('tn', tn)
            print('fp', fp)
            print('fn', fn)
            neg_count = 0
            ind_count = 0
            for idx, cpr_prob in enumerate(cpr_pred):
                cpr_prob = np.exp(cpr_prob)
                cpr_prob_rev = np.exp(cpr_pred_reverse[idx])
                x_prob = np.exp(x_pred[idx])
                y_prob = np.exp(y_pred[idx])
                xy_prob = np.exp(xy_pred[idx])
                pmi = np.log(xy_prob / (x_prob * y_prob)) / -np.log(xy_prob)
                s1 = [str(a) for a in test_1[idx]]
                s2 = [str(a) for a in test_2[idx]]
                p1 = test_phrase1[idx]
                p2 = test_phrase2[idx]
                out_file.write("%f\t%f\t%f\t%f\t%f\t%f\t%s\t%s\t%s\t%s" % (
                x_prob, y_prob, xy_prob, pmi, cpr_prob, cpr_prob_rev, " ".join(s1), p1, " ".join(s2), p2))
                out_file.write(str(x_prob) + "\t" +str(y_prob)+ "\t" + str(xy_prob)+ "\t" +str(pmi)+ "\t" + str(cpr_prob)+ "\t" +str(cpr_prob_rev)+ " ".join(s1)+ "\t" + p1+ "\t" + " ".join(s2)+ "\t" + p2)

                if len(test_labels) == len(cpr_pred):
                    out_file.write("\t%s" % test_labels[idx])
                out_file.write("\n")
            out_file.close()
	    print("X Prediction correlation: %f" % x_corr)
            print("X KL divergence: %f" % x_kl)
	    print("Y Prediction correlation: %f" % y_corr)
            print("Y KL divergence: %f" % y_kl)

            print("Prediction correlation: %f" % cpr_xy_corr)
            print("KL divergence: %f" % cpr_kl)
            print("Number of negative correlation", neg_count)
            print("Number of independence", ind_count)
            print("--- %s seconds ---" % (time.time() - start_time))
