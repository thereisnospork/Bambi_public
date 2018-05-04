import pandas as pd
import numpy as np
import scipy as sp
import tensorflow as tf
from timeit import default_timer as timer


# ####################READ IN CSV PARTITION INTO HEAD BODY INPUT OUTPUTS##################

df = pd.read_csv(r'C:\Users\georg\PycharmProjects\Bambi\bambi_test2.csv')
df = df.drop(columns = ['Factor Name',]) #drop experiment run id.

# print(df)

df_head = df[0:3].reset_index().copy()
df_body = df[3:].reset_index().copy()
df_body = df_body.apply(pd.to_numeric, errors = 'ignore')

# print(df_head)

df_output = df_body.filter(regex=('OUT.*')) #selects OUT**** to create DF of outputs
df_output = df_output.apply(pd.to_numeric, errors = 'ignore')

# df_input = df_body[df_body.columns.difference(df_output.columns)] #strips outputs to create input DF
df_input = df_body.drop(columns=['index',])
df_input = df_input.drop(columns=list(df_output)) #drop output column names

# print(df_input)

df_head = df_head.drop(columns=['index',])
df_head = df_head.drop(columns=list(df_output))

# print(df_head)

types = df_head.loc[[0]].values.flatten()
mins = df_head.loc[[1]].values.flatten().astype(np.float32)  # rows to #'s in np array
maxes = df_head.loc[[2]].values.flatten().astype(np.float32)

def rotate(l, x):
    x = x%len(l)
    return l[-x:] + l[:-x]


def combinatorial_array(mins, maxes, types, resolution = 4, ):
    """
    takes dataframe of header, pre-pruned, returns a 2-d numpy array of inputs evenly spaced
    over the whole design space accounting for mixtures and categorical variables
    suitable for feeding into the tf prediction algorithm

    must be fed non-normalized values!!!!
    returns non-normalized values!!!!
    """

    pairs_arr = list()

    #create boolean arrays for types
    cat_bool = (types == 'CATEGORICAL')
    cont_bool = (types == 'CONTINUOUS')
    mix_bool = (types == 'MIX')

    num_mix = np.count_nonzero(mix_bool)
    mix_order = 0  # staggers mixture concentrations so sum adds up

    for i, _ in enumerate(mins):

        if cat_bool[i]:
            pair = np.arange(mins[i],maxes[i]+1,1)  # +1 required due to non-inclusive
            print(pair)
            pairs_arr.append(list(pair))

        if cont_bool[i]: #inc mixtures for now
            pair = list()
            print(mins[i])
            a = mins[i]
            b = maxes[i]
            pair.append(a)
            pair.append(b)
            pairs_arr.append(pair)

        if mix_bool[i]:
            pair = np.linspace(mins[i],maxes[i],num_mix) #DEBUG CHECK make sure works for n > 3 mixes!!!
            pair = pair / sum(pair) * 100 #normalization, dbl check to make sure works for maxes/mins not 0-100!!!!
            pair = rotate(list(pair), mix_order)
            mix_order += 1
            pairs_arr.append(pair)
            # print(pair)



        # arr[i] = list(mins[i]).append(maxes[i])
    print(pairs_arr)










    # out_length = (resolution**(np.sum(cont_bool+mix_bool)))*2**(np.sum(cat_bool))
    # out_length = (resolution**(25))*2**(np.sum(cat_bool))

    # print(out_length)

    # for _, i in enumerate(mins):
    #     for
    #need list of levels for each factor (1d array of lists)






    # for _, index in mins:
    #     if type[index] == 'MIX':
    #         'etc.'
    #     elif types[index] == 'CATEGORICAL':
    #         'ETC.'
    #     else types[index == ]:
    #         'etc'




combinatorial_array(mins, maxes, types)

num_factors = df_input.shape[1]
num_outs = df_output.shape[1]

# print(num_factors)
# print(num_outs)

#####Generating predictive expression#####

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape = [None,num_factors]) #width of input
y_ = tf.placeholder(tf.float32, shape = [None,num_outs]) #width of output


layer1 = tf.layers.dense(x, num_factors, tf.nn.softplus, bias_initializer=tf.random_uniform_initializer)
layer2 = tf.layers.dense(layer1, num_factors, tf.nn.softplus, bias_initializer=tf.random_uniform_initializer)
layer3 = tf.layers.dense(layer2, num_factors, tf.nn.softplus, bias_initializer=tf.random_uniform_initializer)
layer4 = tf.layers.dense(layer3, num_factors, tf.nn.softplus, bias_initializer=tf.random_uniform_initializer)
layer5 = tf.layers.dense(layer4, num_factors, tf.nn.softplus, bias_initializer=tf.random_uniform_initializer)

y = tf.layers.dense(layer5, num_outs, tf.nn.softplus)


with tf.name_scope('cross_entropy'):
    cross_entropy = tf.losses.huber_loss(labels = y_, predictions = y)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(epsilon = .00001).minimize(cross_entropy)
    # train_step = tf.train.GradientDescentOptimizer(.9).minimize(cross_entropy)

ins = df_input.values

norm_vector_in = np.linalg.norm(ins, axis = 0, ord = 1)
ins = ins / norm_vector_in

outs = df_output.values #pd df to numpy
norm_vector_out = np.linalg.norm(outs, axis = 0, ord = 1)
outs = outs / norm_vector_out




for _ in range(0) :#True:    #do or do not run training

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

    for i in range(50000):
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
            print('step {0}, training error {1}, test error {2} in {3}-seconds'.format(i, train_error, test_error, delta_t))
            prediction = sess.run(y,{x: ins[:]}) * norm_vector_out
    # print(prediction)
            print(100 * (prediction[1]- (outs[1] * norm_vector_out))/(outs[1] * norm_vector_out))
    # print(prediction[1,1])
    out_weights = [1,1,1]






# run prediction over whole range???
# optimize for predictions for weighted sum of each variable that is in the ~335+ percentile +/-