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


layer1 = tf.layers.dense(x, num_factors, tf.nn.sigmoid, bias_initializer=tf.random_uniform_initializer)
layer2 = tf.layers.dense(layer1, num_factors, tf.nn.sigmoid, bias_initializer=tf.random_uniform_initializer)
layer3 = tf.layers.dense(layer2, num_factors, tf.nn.sigmoid, bias_initializer=tf.random_uniform_initializer)
layer4 = tf.layers.dense(layer3, num_factors, tf.nn.sigmoid, bias_initializer=tf.random_uniform_initializer)
layer5 = tf.layers.dense(layer4, num_factors, tf.nn.sigmoid, bias_initializer=tf.random_uniform_initializer)

y = tf.layers.dense(layer5, num_outs, tf.nn.sigmoid)


with tf.name_scope('cross_entropy'):
    cross_entropy = tf.losses.huber_loss(labels = y_, predictions = y)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(epsilon = .01).minimize(cross_entropy)


ins = df_input.values

norm_vector_in = np.linalg.norm(x)

print(norm_vector_in)

outs = df_output.values #pd df to numpy





if False: #for _ in range(100) :#True:    #do or do not run training

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

    for i in range(5000):
        # sess.run(train_step, feed_dict={x: np.expand_dims(ins[train_i], axis = 0),
        #                                 y_: np.expand_dims(outs[train_i], axis = 0)}) #first batch over whole set

        sess.run(train_step, feed_dict={x: ins[train_i],
                                        y_: outs[train_i]})

        if i % 25 == 0 or i in [1,2,3,4,5,6,8,10,12,15,20]: #batching
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
    prediction = sess.run(y,{x: ins[:]})
    # print(prediction)
    print(prediction[1])
    print(prediction[1,1])
    out_weights = [1,1,1]


# run prediction over whole range???
# optimize for predictions for weighted sum of each variable that is in the ~335+ percentile +/-