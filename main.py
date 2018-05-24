import pandas as pd
import numpy as np
import tensorflow as tf
from functions import design_space_sample, mix_sum, normalize, cat_ratios, \
    de_normalize, mins_maxes, optimal_design
from timeit import default_timer as timer

# ####################READ IN CSV PARTITION INTO HEAD BODY INPUT OUTPUTS##################

df = pd.read_csv(r'C:\Users\georg\PycharmProjects\Bambi\bambi_testing_2cat.csv')
df = df.drop(columns = ['Factor Name',]) #drop experiment run id.
df_cols_in_order =  df.columns.values.tolist()

# print(df)

df_head = df[0:3].reset_index().copy()
df_body = df[3:].reset_index().copy()
df_body = df_body.apply(pd.to_numeric, errors = 'ignore')

# print(df_head)

df_output = df_body.filter(regex=('OUT.*')) #selects OUT**** to create DF of outputs
df_output = df_output.apply(pd.to_numeric, errors = 'ignore')
df_out_head = df_head.filter(regex=('OUT.*')) #selects OUT**** to create DF of outputs
df_out_head = df_out_head.fillna(value = 1) # sets default value for out weights to 1 if not input by user
print(df_out_head)

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
labels = df_head.columns.values
out_weights = df_out_head.loc[[2]].values.flatten().astype(np.float32)

# print(out_weights)

cat_bool = (types == 'CATEGORICAL')
cat_encode_by_label = dict() # nested column -> key -> replace
cat_decode_by_label = dict() #

for i, label in enumerate(labels):  # recode all categorical factors to integer numeric
    if cat_bool[i]:
        existing = np.unique(df_input[label])
        replacing = np.arange(0, len(existing))
        mins[i] = 0
        maxes[i] = len(existing)-1
        cat_encode_by_label[label] = dict(zip(existing, replacing))  # makes full nested dict for later use if needed
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


# ####Generating predictive expression#####
# ####Model to fit prediction expression
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape = [None,num_factors])  # width of input
y_ = tf.placeholder(tf.float32, shape = [None,num_outs])  # width of output


layer1 = tf.layers.dense(x, num_factors, tf.nn.softplus, bias_initializer=tf.random_uniform_initializer)
layer2 = tf.layers.dense(layer1, num_factors, tf.nn.softplus, bias_initializer=tf.random_uniform_initializer)
layer3 = tf.layers.dense(layer2, num_factors, tf.nn.softplus, bias_initializer=tf.random_uniform_initializer)
layer4 = tf.layers.dense(layer3, num_factors, tf.nn.softplus, bias_initializer=tf.random_uniform_initializer)
layer5 = tf.layers.dense(layer4, num_factors, tf.nn.softplus, bias_initializer=tf.random_uniform_initializer)

y = tf.layers.dense(layer5, num_outs, tf.nn.softplus)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.losses.huber_loss(labels = y_, predictions = y)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=.00005, epsilon = .00000001).minimize(cross_entropy)
    # train_step = tf.train.GradientDescentOptimizer(.9).minimize(cross_entropy)

ins_unnorm = df_input.values



norm_vector_in = np.linalg.norm(ins_unnorm, axis = 0, ord = 2)/(ins_unnorm.shape[1])

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

# parim_vals = perimeter_array(mins,maxes,types)
# parim_vals = normalize(parim_vals, norm_vector_in)

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
        if p > 0.8:  # split train / test
            test_i.append(index)
        else:
            train_i.append(index)

    loop_start = timer()

    for i in range(100000): #50000):
        sess.run(train_step, feed_dict={x: ins[train_i],
                                        y_: outs[train_i]})

        if i % 25 == 0 or i == 2 or i == 4 or i == 10: #batching
            test_i = list()
            train_i = list() ###empty the ins and outs index lists

            for index, _ in enumerate(ins):
                p = np.random.random()
                if p > 0.8: #split train / test
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

    # perim_eval = sess.run(y,{x: parim_vals[:]}) # * norm_vector_out
    # perim_eval = np.asarray(perim_eval)
    # print(perim_eval)
    # perim_eval = de_normalize(perim_eval,norm_vector_out)
    # print(norm_vector_out)
    # print(norm_vector_in)
    # print(perim_eval)




# generate samples over design space
samples = max(num_factors ** 2 * 1000, 10000000)

space_sample = design_space_sample(mins, maxes, types, samples, mix_sum(ins_unnorm, types))
space_sample_norm = normalize(space_sample, norm_vector_in)
space_sample_norm = np.transpose(space_sample_norm)
# Evaluate samples according to TF model
space_sample_eval = sess.run(y, {x: space_sample_norm[:]})

#weight and sum the resulted models
weighted_eval = np.sum(space_sample_eval * out_weights, axis = 1)

#select the top X percent or n samples that were evaluated
werty, top_index = tf.nn.top_k(weighted_eval, 100000)#samples//20000) #top 0.5% of samples
best_ins = space_sample[:,top_index.eval()]
# print(top_index.eval())
# print(werty.eval())
# print(best_ins)

#generate new mins, maxes based off of monte carlo evaluation above
best_mins, best_maxes = mins_maxes(best_ins)

#generate ratios for cat variables based off of monte carlo evaluation above

#wrap this in function after it works

# def
# cat_bool = (types == 'CATEGORICAL')
# cat_levels = dict()
# list_of_cat_ratio_dict = list()
# for i, each in enumerate(types):
#     # print(i)
#     # print(cat_bool[i])
#     if cat_bool[i]:
#         unique, counts = np.unique(best_ins[i], return_index = True)
#         counts = counts/np.sum(counts) #normalize to a percentage
#         list_of_cat_ratio_dict.append(dict(zip(unique, counts))) #
# print(list_of_cat_ratio_dict)

cat_dict = cat_ratios(best_ins, types, mins, maxes)

new_experiments = optimal_design\
    (best_mins, best_maxes, types, 40,  mix_sum(ins_unnorm, types), norm_vector_in, cat_dict) # just base mins/maxes right now for debugging!

out_df = pd.DataFrame(data = np.transpose(new_experiments), columns = df_head.columns.values.tolist())

#recode categorical factors:

for i, label in enumerate(labels):  # recode all categorical factors to integer numeric
    if cat_bool[i]:
        # existing = np.unique(df_input[label])  #created above, why duplicate
        # replacing = np.arange(0, len(existing))
        mins[i] = 0
        maxes[i] = len(existing)-1

        # cat_encode_by_label[label] = dict(zip(existing, replacing))  # makes full nested dict for later use if needed
        # cat_decode_by_label[label] = dict(zip(replacing, existing)) #created above
        df_input[label] = df_input[label].map(cat_decode_by_label[label])

out_df = df.append(out_df)
out_df = out_df[df_cols_in_order]
out_df.to_csv(r'C:\Users\georg\PycharmProjects\Bambi\out\bambi_test.csv', sep = ',')
