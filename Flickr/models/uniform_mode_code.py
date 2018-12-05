            # uniform measure
            print('uniform_poe')
            # Prediction
            output_layer = Layer.W(args['hidden_dim'], args['output_dim'], 'Output')
            output_bias  = Layer.b(args['output_dim'], 'OutputBias')

            Wemb1 = tf.nn.embedding_lookup(embeddings, inputs1)
            Wemb2 = tf.nn.embedding_lookup(embeddings, inputs2)
            output, fstate1 = tf.nn.dynamic_rnn(lstm, Wemb1, sequence_length=lengths1, dtype=tf.float32)
            tf.get_variable_scope().reuse_variables()
            output, fstate2 = tf.nn.dynamic_rnn(lstm, Wemb2, sequence_length=lengths2, dtype=tf.float32)
            logits1 = tf.matmul(fstate1[0], output_layer) + output_bias
            logits2 = tf.matmul(fstate2[0], output_layer) + output_bias

            max_logits1 = tf.ones([tf.shape(logits1)[0], tf.shape(logits1)[1]])
            max_logits2 = tf.ones([tf.shape(logits1)[0], tf.shape(logits1)[1]])
            intersect_vec = uniform_prob.intersection_point_log(logits1, logits2, max_logits1, max_logits2)
            joint_predicted = uniform_prob.joint_probability_log(logits1, logits2, max_logits1, max_logits2)
            cpr_predicted = uniform_prob.cond_probability_log(logits1, logits2, max_logits1, max_logits2)
            cpr_predicted_reverse = uniform_prob.cond_probability_log(logits2, logits1, max_logits2, max_logits1)
            x_predicted = uniform_prob.probability(logits1, max_logits1)
            y_predicted = uniform_prob.probability(logits2, max_logits2)
            cpr_loss = tf.nn.softmax_cross_entropy_with_logits(logits= uniform_prob.create_log_distribution(cpr_predicted, args['batch_size']), labels= uniform_prob.create_distribution(cpr_labels, args['batch_size']))
        
            not_have_meet = x_predicted
            t1_min_embed = x_predicted
            t1_max_embed = x_predicted

            x_loss = tf.nn.softmax_cross_entropy_with_logits(logits= uniform_prob.create_log_distribution(x_predicted, args['batch_size']), labels= uniform_prob.create_distribution(x_labels, args['batch_size']))
            x_log_prob = uniform_prob.create_log_distribution(x_predicted, args['batch_size'])
            y_loss = tf.nn.softmax_cross_entropy_with_logits(logits= uniform_prob.create_log_distribution(y_predicted, args['batch_size']), labels= uniform_prob.create_distribution(y_labels, args['batch_size']))
        



        
        elif args['mode'] == 'uniform_cube':
            print('uniform_cube')

            Wemb1 = tf.nn.embedding_lookup(embeddings, inputs1)
            Wemb2 = tf.nn.embedding_lookup(embeddings, inputs2)
            output, fstate1 = tf.nn.dynamic_rnn(lstm, Wemb1, sequence_length=lengths1, dtype=tf.float32)
            tf.get_variable_scope().reuse_variables()
            output, fstate2 = tf.nn.dynamic_rnn(lstm, Wemb2, sequence_length=lengths2, dtype=tf.float32)

            if args['cube'] == 'sigmoid':
                print('use sigmoid to make cube')
                input_dim = args['hidden_dim']
                output_dim = args['output_dim']
                output_layer = tf.Variable(tf.random_uniform([input_dim, output_dim], minval=-0.05, maxval=0.05, seed=12132015), name='Output1', trainable=True)
                output_bias  = tf.Variable(tf.random_uniform([output_dim], minval=-0.05, maxval=0.05, seed=12132015), name='OutputBias1', trainable=True)

                logits1 = tf.matmul(fstate1[0], output_layer) + output_bias
                logits2 = tf.matmul(fstate2[0], output_layer) + output_bias

                output_layer1 = tf.Variable(tf.random_uniform([input_dim, output_dim], minval=-0.05, maxval=0.05, seed=12132015), name='Output2', trainable=True)
                output_bias1 = tf.Variable(tf.random_uniform([output_dim], minval=2.00, maxval=2.50, seed=12132015), name='OutputBias2', trainable=True)

                delta_logits1 = tf.matmul(fstate1[0], output_layer1) + output_bias1
                delta_logits2 = tf.matmul(fstate2[0], output_layer1) + output_bias1

                t1_min_embed = tf.sigmoid(logits1)
                t2_min_embed = tf.sigmoid(logits2)
                # t1_max_embed = t1_min_embed + (1-t1_min_embed) * tf.sigmoid(delta_logits1)
                # t2_max_embed = t2_min_embed + (1-t2_min_embed) * tf.sigmoid(delta_logits2)
                t1_max_embed = tf.ones_like(t1_min_embed)
                t2_max_embed = tf.ones_like(t2_min_embed)
            elif args['cube'] == 'softmax':
                input_dim = args['hidden_dim']
                output_dim = args['output_dim']
                batch_size = args['batch_size']
                # get the min value, softmax[0]
                output_layer = tf.Variable(tf.random_uniform([input_dim, output_dim], minval=-0.05, maxval=0.05, seed=12132015), name='Output1', trainable=True)
                output_bias  = tf.Variable(tf.random_uniform([output_dim], minval=-0.05, maxval=0.05, seed=12132015), name='OutputBias1', trainable=True)

                s01 = tf.matmul(fstate1[0], output_layer) + output_bias
                s02 = tf.matmul(fstate2[0], output_layer) + output_bias
                # get the delta value, softmax[1]
                output_layer1 = tf.Variable(tf.random_uniform([input_dim, output_dim], minval=-0.05, maxval=0.05, seed=12132015), name='Output2', trainable=True)
                output_bias1 = tf.Variable(tf.random_uniform([output_dim], minval=2.00, maxval=2.50, seed=12132015), name='OutputBias2', trainable=True)

                s11 = tf.matmul(fstate1[0], output_layer1) + output_bias1
                s12 = tf.matmul(fstate2[0], output_layer1) + output_bias1

                # fix third one to be 0, softmax[2]
                s21 = s22 =tf.zeros([batch_size, output_dim])

                t1_embed = tf.stack([s01, s11, s21], axis = 2)
                t2_embed = tf.stack([s02, s12, s22], axis = 2)
                
                # use softmax to get embedding
                t1_embed =tf.nn.softmax(t1_embed)
                t1_min_embed=t1_embed[:,:,0]
                t1_delta_embed = t1_embed[:,:,1]

                t2_embed =tf.nn.softmax(t2_embed)
                t2_min_embed=t2_embed[:,:,0]
                t2_delta_embed = t2_embed[:,:,1]


                t1_max_embed = t1_min_embed + t1_delta_embed
                t2_max_embed = t2_min_embed + t2_delta_embed


            join_min, join_max, meet_min, meet_max, not_have_meet = cube_prob.calc_join_and_meet(t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed)
            
            # cond =  tf.cast(tf.less_equal(t2, t1_min_embed), tf.float32) # batchsize * embed_size
            # # cond = tf.reduce_sum(cond, axis = 1)
            # test_tensor = tf.reduce_sum(cond) 
            # for evaluation
            joint_predicted = cube_prob.test_joint_probability_log(join_min, join_max, meet_min, meet_max, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed, not_have_meet)
            cpr_predicted = cube_prob.test_cond_probability_log(join_min, join_max, meet_min, meet_max, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed, not_have_meet)
            cpr_predicted_reverse = cube_prob.test_cond_probability_log(join_min, join_max, meet_min, meet_max, t2_min_embed, t2_max_embed, t1_min_embed, t1_max_embed, not_have_meet)

            # for training
            # calculate log conditional probability for positive examplse, and negative upper bound if two things are disjoing
            train_cpr_predicted = cube_prob.slicing_where(condition = not_have_meet,
                full_input = ([join_min, join_max, meet_min, meet_max, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed]),
                true_branch = lambda x: cube_prob.lambda_batch_log_upper_bound(*x),
                false_branch = lambda x: cube_prob.lambda_batch_log_cube_measure(*x))
            # calculate log(1-p) if overlap, 0 if no overlap
            onem_cpr_predicted = cube_prob.slicing_where(condition = not_have_meet,
                full_input = tf.tuple([join_min, join_max, meet_min, meet_max, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed]),
                true_branch = lambda x: cube_prob.lambda_zero_log_upper_bound(*x), 
                false_branch = lambda x: cube_prob.lambda_batch_log_cond_cube_measure(*x))

            print('cpr_predicted', train_cpr_predicted.get_shape)
            print('onem_cpr_predicted', onem_cpr_predicted.get_shape)

            whole_cpr_predicted = tf.concat([tf.expand_dims(train_cpr_predicted, 1), tf.expand_dims(onem_cpr_predicted, 1)], 1)

            x_predicted = cube_prob.probability(t1_min_embed, t1_max_embed)
            y_predicted = cube_prob.probability(t2_min_embed, t2_max_embed)
            cpr_loss = tf.nn.softmax_cross_entropy_with_logits(logits= whole_cpr_predicted, labels= Probability.create_distribution(cpr_labels, args['batch_size']))
            
            x_loss = tf.nn.softmax_cross_entropy_with_logits(logits= cube_prob.create_log_distribution(x_predicted, args['batch_size']), labels= cube_prob.create_distribution(x_labels, args['batch_size']))
            y_loss = tf.nn.softmax_cross_entropy_with_logits(logits= cube_prob.create_log_distribution(y_predicted, args['batch_size']), labels= cube_prob.create_distribution(y_labels, args['batch_size']))
            x_log_prob = cube_prob.create_log_distribution(x_predicted, args['batch_size'])
        
            test_tensor, test_tensor1 = cube_prob.test(join_min, join_max, meet_min, meet_max, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed)