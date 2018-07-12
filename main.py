import pandas as pd
import numpy as np
from functions import design_space_sample, mix_sum, normalize, cat_ratios, \
    mins_maxes, optimal_design
from timeit import default_timer as timer
import gc
# ####################READ IN CSV PARTITION INTO HEAD BODY INPUT OUTPUTS##################

def anal(df, num_requested):
    """
    Takes input dataframe - formatted according to template - of experimental data
    :param df:
    :param num_requested
    :return:
    """
    try:
        df = df.drop(columns=['Factor Name_', ])  # drop experiment run id.
        df_cols_in_order = df.columns.values.tolist()

        # print(df)

        df_head = df[0:3].reset_index().copy()
        df_body = df[3:].reset_index().copy()
        df_body = df_body.apply(pd.to_numeric, errors='ignore')
        df_body.dropna(inplace=True)    # = df_body.dropna() #nuke all rows containing NaNs
        # print(df_head)

        df_output = df_body.filter(regex=('OUT.*'))  # selects OUT**** to create DF of outputs
        df_output = df_output.apply(pd.to_numeric, errors='ignore')
        df_out_head = df_head.filter(regex=('OUT.*'))  # selects OUT**** to create DF of outputs
        df_out_head = df_out_head.fillna(value=1)  # sets default value for out weights to 1 if not input by user
        # print(df_out_head)
        #
        # df_input = df_body[df_body.columns.difference(df_output.columns)] #strips outputs to create input DF
        df_input = df_body.drop(columns=['index', ])
        df_input = df_input.drop(columns=list(df_output))  # drop output column names

        # print(df_input)

        df_head = df_head.drop(columns=['index', ])
        df_head = df_head.drop(columns=list(df_output))

        # print(df_head)

        types = df_head.loc[[0]].values.flatten()
        # experimental mins/maxs from inputted values if not supplied properly or supplied at all
        try:
            mins = df_head.loc[[1]].values.flatten().astype(np.float32)  # rows to #'s in np array
            maxes = df_head.loc[[2]].values.flatten().astype(np.float32)
        except Exception:
            mins, maxes = mins_maxes(df_input.values) ###DEUG ME PLEASE

        labels = df_head.columns.values
        out_weights = df_out_head.loc[[2]].values.flatten().astype(np.float32)

        # print(out_weights)

        cat_bool = (types == 'CATEGORICAL') | (types == 'CAT') #needs to be | instead of 'or'
        cat_encode_by_label = dict()  # nested column -> key -> replace
        cat_decode_by_label = dict()  #

        for i, label in enumerate(labels):  # recode all categorical factors to integer numeric
            if cat_bool[i]:
                existing = np.unique(df_input[label])
                replacing = np.arange(0, len(existing))
                mins[i] = 0
                maxes[i] = len(existing) - 1
                cat_encode_by_label[label] = dict(
                    zip(existing, replacing))  # makes full nested dict for later use if needed
                cat_decode_by_label[label] = dict(zip(replacing, existing))
                df_input[label] = df_input[label].map(cat_encode_by_label[label])

                # print(df_input)

                # cat_replaced_by_index[i] =  # generates matching 0 - x range as ndarray to labels dict
                # cat_replaced_by_index[i] = np.arange(0, len(cat_values_by_index[i])) # generates matching 0 - x range as ndarray to labels dict

        # df_input.replace()

        # ###mins/maxes might break here for cat values.
        # df_input.replace(cat_encode_by_label, inplace=True, method = 'ffill')

        # print(df_input)

        # asdf = np.unique(df_input[labels[1]]) # all unique entries in a column of data
        #
        # print(asdf)
        # print(type(asdf))
        num_factors = df_input.shape[1]
        num_outs = df_output.shape[1]
        import tensorflow as tf  ###down here to prevent memory isseus

        # ####Generating predictive expression#####
        # ####Model to fit prediction expression
        sess = tf.InteractiveSession()

        x = tf.placeholder(tf.float32, shape=[None, num_factors])  # width of input
        y_ = tf.placeholder(tf.float32, shape=[None, num_outs])  # width of output

        layer1 = tf.layers.dense(x, num_factors, tf.nn.softplus, bias_initializer=tf.random_uniform_initializer)
        layer2 = tf.layers.dense(layer1, num_factors, tf.nn.softplus, bias_initializer=tf.random_uniform_initializer)
        layer3 = tf.layers.dense(layer2, num_factors, tf.nn.softplus, bias_initializer=tf.random_uniform_initializer)
        layer4 = tf.layers.dense(layer3, num_factors, tf.nn.softplus, bias_initializer=tf.random_uniform_initializer)
        layer5 = tf.layers.dense(layer4, num_factors, tf.nn.softplus, bias_initializer=tf.random_uniform_initializer)

        y = tf.layers.dense(layer5, num_outs, tf.nn.softplus)

        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.losses.huber_loss(labels=y_, predictions=y)

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(learning_rate=.00005, epsilon=.00000001).minimize(cross_entropy)  # lr .00005 or =min([.00005, .0001 / df_input.shape[0]])
            # train_step = tf.train.GradientDescentOptimizer(.9).minimize(cross_entropy)

        ins_unnorm = df_input.values

        norm_vector_in = np.linalg.norm(ins_unnorm, axis=0, ord=2) / (ins_unnorm.shape[1])

        ins = ins_unnorm / norm_vector_in

        outs = df_output.values  # pd df to numpy
        norm_vector_out = np.linalg.norm(outs, axis=0, ord=2) / (outs.shape[1])
        outs = outs / norm_vector_out

        # print(ins)


        if True:  # do or do not run training

            sess.run(tf.global_variables_initializer())

            test_i = list()
            train_i = list()  ###empty the ins and outs index lists

            for index, _ in enumerate(ins):
                p = np.random.random()
                if p > 0.8:  # split train / test
                    test_i.append(index)
                else:
                    train_i.append(index)

            loop_start = timer()

            for i in range(max(100000, num_factors*2000, df_input.shape[0]*500)):  # 50000): # 100000 previously
                sess.run(train_step, feed_dict={x: ins[train_i],
                                                y_: outs[train_i]})

                if i % 25 == 0 or i == 2 or i == 4 or i == 10:  # batching
                    test_i = list()
                    train_i = list()  ###empty the ins and outs index lists

                    for index, _ in enumerate(ins):
                        p = np.random.random()
                        if p > 0.8:  # split train / test
                            test_i.append(index)
                        else:
                            train_i.append(index)

                    train_error = cross_entropy.eval(feed_dict={x: ins[train_i], y_: outs[train_i]})
                    test_error = cross_entropy.eval(feed_dict={x: ins[test_i], y_: outs[test_i]})

                if i % 10000 == 0:
                    loop_end = timer()
                    delta_t = loop_end - loop_start
                    print('step {0}, training error {1}, test error {2} in {3}-seconds'.format(i, train_error, test_error,
                                                                                               delta_t))

        # generate samples over design space
        samples = max(num_factors ** 2 * 1000, 10000000)

        space_sample = design_space_sample(mins, maxes, types, samples, mix_sum(ins_unnorm, types))
        space_sample_norm = normalize(space_sample, norm_vector_in)
        space_sample_norm = np.transpose(space_sample_norm)
        # Evaluate samples according to TF model

        per_loop = 100000
        loops = samples//per_loop
        space_sample_eval = np.zeros([samples, num_outs])

        for n, i in enumerate(range(loops)):
            space_sample_eval[n*per_loop:(n+1)*per_loop] = sess.run(y, {x: space_sample_norm[n*per_loop:(n+1)*per_loop]})

        # weight and sum the resulted models
        weighted_eval = np.sum(space_sample_eval * out_weights, axis=1)

        # select the top X percent or n samples that were evaluated
        werty, top_index = tf.nn.top_k(weighted_eval, 100000)  # samples//20000) #top 0.5% of samples
        best_ins = space_sample[:, top_index.eval()]

        # generate new mins, maxes based off of monte carlo evaluation above
        best_mins, best_maxes = mins_maxes(best_ins)

        # generate ratios for cat variables based off of monte carlo evaluation above

        cat_dict = cat_ratios(best_ins, types, mins, maxes)

        new_experiments = optimal_design \
            (best_mins, best_maxes, types, num_requested, mix_sum(ins_unnorm, types), norm_vector_in,
             cat_dict)  # just base mins/maxes right now for debugging!

        out_df = pd.DataFrame(data=np.transpose(new_experiments), columns=df_head.columns.values.tolist())

        # recode categorical factors:

        for i, label in enumerate(labels):  # recode all categorical factors to integer numeric
            if cat_bool[i]:
                # existing = np.unique(df_input[label])  #created above, why duplicate
                # replacing = np.arange(0, len(existing))
                mins[i] = 0
                maxes[i] = len(existing) - 1

                # cat_encode_by_label[label] = dict(zip(existing, replacing))  # makes full nested dict for later use if needed
                # cat_decode_by_label[label] = dict(zip(replacing, existing)) #created above
                df_input[label] = df_input[label].map(cat_decode_by_label[label])

        df = pd.DataFrame(columns=df_cols_in_order)  # only return newly generated experiments #comment me out if going for full array inc. data
        out_df = df.append(out_df, sort=False)  # df set at top to be full frame minus experiment #
        out_df = out_df[df_cols_in_order]
        out_df = out_df.reset_index(drop=True)
        sess.close()
        sess.__del__() #nuke session
        gc.collect()
        return out_df

    except Exception as e:
        raise
        sess.close()
        sess.__del__()  # nuke session
        gc.collect()
        return e



##test eval##

# df = pd.read_csv(r'C:\Users\georg\Downloads\1000_exp_4_data.csv')
# print(df)
# out_df = anal(df, 40)
# out_df.to_csv(r'C:\Users\georg\PycharmProjects\Bambi\out\bambi_test.csv', sep=',')




        #
        # # generate samples over design space
        # designs = max(10000000, num_factors ** 2 * 1000)
        # loops = min(50, designs//1000000)
        # per_loop = designs//loops
        # weighted_eval = np.zeros(designs) #preallocate
        #
        # for n, i in enumerate(range(loops)):
        #
        #     # samples = max(num_factors ** 2 * 1000, designs)
        #
        #     space_sample = design_space_sample(mins, maxes, types, per_loop, mix_sum(ins_unnorm, types))
        #     space_sample_norm = normalize(space_sample, norm_vector_in)
        #     space_sample_norm = np.transpose(space_sample_norm)
        #     # Evaluate samples according to TF model
        #     space_sample_eval = sess.run(y, {x: space_sample_norm[:]})
        #
        #     # weight and sum the resulted models
        #     print(n)
        #     weighted_eval[n*per_loop:((n+1)*per_loop)] = np.sum(space_sample_eval * out_weights, axis=1).flatten() #add weights to preallocated array
