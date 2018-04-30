import pandas as pd
import numpy as np
import scipy as sp
import tensorflow as tf
from timeit import default_timer as timer


# ####################READ IN CSV PARTITION INTO HEAD BODY INPUT OUTPUTS##################

df = pd.read_csv(r'C:\Users\georg\PycharmProjects\Bambi\bambi_test.csv')
df = df.drop(columns = ['Factor Name',]) #drop experiment run id.

df_head = df[0:4].reset_index().copy()
df_body = df[4:].reset_index().copy()
df_body = df_body.apply(pd.to_numeric, errors = 'ignore')


df_output = df_body.filter(regex=('OUT.*')) #selects OUT**** to create DF of outputs
df_output = df_output.apply(pd.to_numeric, errors = 'ignore')


# df_input = df_body[df_body.columns.difference(df_output.columns)] #strips outputs to create input DF
df_input = df_body.drop(columns=['index',])
df_input = df_input.drop(columns=list(df_output)) #drop output column names


num_factors = df_input.shape[1]
num_outs = df_output.shape[1]

# print(num_factors)
# print(num_outs)

#####Generating predictive expression#####
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape = [None,num_factors]) #width of input
y_ = tf.placeholder(tf.float32, shape = [None,num_outs]) #width of output


layer1 = tf.layers.dense(x, num_factors, tf.nn.relu)
layer2 = tf.layers.dense(layer1, num_factors, tf.nn.relu)
layer3 = tf.layers.dense(layer2, num_factors, tf.nn.relu)
layer4 = tf.layers.dense(layer3, num_factors, tf.nn.relu)
layer5 = tf.layers.dense(layer4, num_factors, tf.nn.relu)

y = tf.layers.dense(layer5, num_outs, tf.nn.relu)


with tf.name_scope('cross_entropy'):
    cross_entropy = tf.losses.huber_loss(labels = y_, predictions = y)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(epsilon = .001).minimize(cross_entropy)
    # train_step = tf.train.GradientDescentOptimizer(.9).minimize(cross_entropy)

ins = df_input.values
# ins = np.expand_dims(ins, axis = 0)
outs = df_output.values #pd df to numpy
# outs = np.expand_dims(outs, axis = 0)





if True :#True:    #do or do not run training

    sess.run(tf.global_variables_initializer())
    # train_i = df_body.shape[1] #length of body = number of runs

    test_i = list()
    train_i = list()  ###empty the ins and outs index lists

    for index, _ in enumerate(ins):
        p = np.random.random()
        if p > 0.6:  # split train / test
            test_i.append(index)
        else:
            train_i.append(index)

    loop_start = timer()

    for i in range(25000):
        # sess.run(train_step, feed_dict={x: np.expand_dims(ins[train_i], axis = 0),
        #                                 y_: np.expand_dims(outs[train_i], axis = 0)}) #first batch over whole set

        sess.run(train_step, feed_dict={x: ins[train_i],
                                        y_: outs[train_i]})

        if i % 25 == 0 or i == 2 or i == 4 or i == 10: #batching
            test_i = list()
            train_i = list() ###empty the ins and outs index lists

            for index, _ in enumerate(ins):
                p = np.random.random()
                if p > 0.6: #split train / test
                    test_i.append(index)
                else:
                    train_i.append(index)

                # print(test_i)
                # print(train_i)

            train_error = cross_entropy.eval(feed_dict={x: ins[train_i], y_: outs[train_i]})
            test_error = cross_entropy.eval(feed_dict={x: ins[test_i], y_: outs[test_i]})



        if i%500 == 0:
            loop_end = timer()
            delta_t = loop_end - loop_start
            print('step {0}, training error {1}, test error {2} in {3}-seconds'.format(i, train_error, test_error,
                                                                                           delta_t))
#
# #
#
# df_output = df_body.filter(regex=('OUT.*'))
# df_output = df_output.infer_objects()
# # df_output = pd.to_numeric(df_output)
#
# # df_output.convert_objects(convert_numeric='coerce')
# print(df_output.dtypes)
# print(df_output)
#
# # print(df_head)
# # print(df_body.dtypes)