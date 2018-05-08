import pandas as pd
import numpy as np
import scipy as sp
import tensorflow as tf
import itertools
from timeit import default_timer as timer

# start_nontf = timer()
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


def perimeter_array(mins, maxes, types, resolution = 4, ):
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
            # print(pair)
            pairs_arr.append(list(pair))

        if cont_bool[i]: #inc mixtures for now
            pair = list()
            # print(mins[i])
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

    out_length = 1  # Formula for total number of combinations to preallocate numpy array
    for each in pairs_arr:
        out_length = out_length * len(each)

    out_arr = np.zeros([len(mins),out_length])

    for i, row in enumerate(itertools.product(*pairs_arr)):
        out_arr[:, i] = row
    return out_arr
    # for t in itertools.product(*pairs_arr):  ###fast flatten of rows, no needed
    #     print(t)

def design_space_sample(mins, maxes, samples):
    """
    For Monte Carlo method to create an average value over the design space using
    the tensorflow eval, generates array of un-normalized tests
    ignoring mixture/categorical value constraints
    """
    out_arr = np.zeros([len(mins),samples])
    for i in range(samples):
        out_arr[:, i] = [np.random.uniform(mins[n],maxes[n]) for n in range(len(mins))]
    return out_arr

def mix_sum(ins, types):
    """ returns float representing total amount of mix - calculated from data
    inputs:
    ins: input data ndarray (floats) representing non-normalized numbers
    types: label ndarray of str 'CATEGORICAL' 'CONTINUOUS' 'MIX' only valid entries
    """
    mix_bool = (types == 'MIX')
    ins_mix = ins * mix_bool #0's non mix inputs - check geometry on matrix  multiplication
    ins_mix = np.sum(ins_mix, axis=1) #sums across experiments
    return np.max(ins_mix)  #max value of summation


def design_space_sample_new(mins, maxes, types, samples, mix_sum):
    """replaces design_space_sample

    inputs: mins -ndarray of floats representing min value of space
    maxs -ndarray representing max values of space
    types -ndarray of str, 'CATEGORICAL' 'CONTINUOUS' 'MIX' only valid entries
    samples - int, number of valid experiments to be produced.
    mix_sum: total un-normalized target of mix variables

    outputs a len(min) x samples

    """

    #create boolean arrays for types
    cat_bool = (types == 'CATEGORICAL')
    cont_bool = (types == 'CONTINUOUS')
    mix_bool = (types == 'MIX')

    num_mix = np.count_nonzero(mix_bool)
    mix_order = 0  # staggers mixture concentrations so sum adds up

    out_arr = np.zeros([len(mins),samples], dtype=np.float32) #preallocate output
    cat_levels = list()

    for i, each in enumerate(mins): #generates list of range interators of valid categorical values for each cat factor
        if cat_bool[i]:
            cat_min = np.int(np.round(mins[i]))
            cat_max = np.int(np.round(maxes[i]))
            cat_level_range = range(cat_min, cat_max)
            cat_levels.append(cat_level_range)
        else:
            cat_levels.append(0) #placeholder 0

    for i, type in enumerate(types):
        if type == 'CONTINUOUS':
            out_arr[i, :] = np.random.uniform(mins[i],maxes[i],[1,samples])
        if type == 'CATEGPROCAL':
            out_arr[i, :] = np.random.choice(cat_levels[i],size = [1,samples])
        if type == 'MIX':
            out_arr[i, :] = np.random.uniform(mins[i], maxes[i], [1, samples]) #need to normalize to mix sum

    mix_sub_array = out_arr[mix_bool]
    mix_sub_sums = np.sum(mix_sub_array, axis = 0)
    mix_sub_norm = mix_sub_sums / mix_sum
    out_arr[mix_bool] = mix_sub_sums / mix_sub_norm  ####math almost certainly needs debugging

    return out_arr



    # for n in range(samples):
    #     mix_sum_local = mix_sum #reset for each set of experiments
    #     num_mix_local = num_mix
    #     for i, each in enumerate(mins):
    #         if cat_bool[i]:
    #             out_arr[i,n] = np.random.choice(cat_levels[i])
    #
    #         if cont_bool[i]:
    #             out_arr[i,n] = np.random.uniform(mins[i],maxes[i])
    #
    #         if mix_bool[i]:







def normalize(in_arr, norm_vector):
    """returns normalized vector in same shape as input based on the normalization vector
    vector length must match input row length"""
    in_arr = in_arr / norm_vector[:, np.newaxis]  # normalize
    in_arr = in_arr.transpose()
    return in_arr

def de_normalize(in_arr, norm_vector):
    """returns de-normalized array in same shape as input based on the normalization vector
    vector length must match input row length"""
    in_arr = in_arr * norm_vector  # normalize
    # in_arr = in_arr.transpose()
    return in_arr

num_factors = df_input.shape[1]
num_outs = df_output.shape[1]

# print(num_factors)
# print(num_outs)

#####Generating predictive expression#####
#####Model to fit prediction expression
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

ins_unnorm = df_input.values
norm_vector_in = np.linalg.norm(ins, axis = 0, ord = 2)/(ins.shape[1])

ins = ins_unnorm / norm_vector_in

outs = df_output.values #pd df to numpy
norm_vector_out = np.linalg.norm(outs, axis = 0, ord = 2)/(outs.shape[1])
outs = outs / norm_vector_out

# print(ins)


#####design space optimization model#########
min_bias = tf.get_variable('min_bias',[num_factors], dtype=tf.float32)
max_bias = tf.get_variable('max_bias',[num_factors], dtype=tf.float32)

min_constant = tf.placeholder(tf.float32, shape=[None,num_factors])
max_constant = tf.placeholder(tf.float32, shape=[None,num_factors])

parim_vals = perimeter_array(mins,maxes,types)
parim_vals = normalize(parim_vals, norm_vector_in)

# print(type(tf.add(mins,min_bias)))
# print(tf.add(mins,min_bias).eval())
# parim_vals2 = perimeter_array(np.add(mins,min_bias),
#                              np.add(maxes,max_bias),
#                              types)

# print(parim_vals)

if True:    #do or do not run training

    sess.run(tf.global_variables_initializer())

    test_i = list()
    train_i = list()  ###empty the ins and outs index lists

    for index, _ in enumerate(ins):
        p = np.random.random()
        if p > 0.6:  # split train / test
            test_i.append(index)
        else:
            train_i.append(index)

    loop_start = timer()

    for i in range(50000): #50000):
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

            train_error = cross_entropy.eval(feed_dict={x: ins[train_i], y_: outs[train_i]})
            test_error = cross_entropy.eval(feed_dict={x: ins[test_i], y_: outs[test_i]})

        if i%500 == 0:
            loop_end = timer()
            delta_t = loop_end - loop_start
            print('step {0}, training error {1}, test error {2} in {3}-seconds'.format(i, train_error, test_error, delta_t))
            # prediction = sess.run(y,{x: ins[:]}) * norm_vector_out
            # print(ins[:].shape)
            # print(parim_vals[:])
    out_weights = [1,2,3]

    perim_eval = sess.run(y,{x: parim_vals[:]}) # * norm_vector_out
    perim_eval = np.asarray(perim_eval)
    print(perim_eval)
    perim_eval = de_normalize(perim_eval,norm_vector_out)
    print(norm_vector_out)
    print(norm_vector_in)
    print(perim_eval)






    def testprint(perim_eval, out_weights):
        asdf = perim_eval*out_weights
        print(asdf)

    # testprint(perim_eval, out_weights)
    # print(perim_eval*out_weights)
    # print(perim_eval.shape)


    # print(prediction)
    #         print(100 * (prediction[1] - (outs[1] * norm_vector_out))/(outs[1] * norm_vector_out))
    # print(prediction[1,1])
    out_weights = [1,1,1]


MIXTURES SPACE GENERATION SUBTRACT FROM POOL!
proportional weighting for categorical in improved design space



# run prediction over whole range???
# optimize for predictions for weighted sum of each variable that is in the ~335+ percentile +/-