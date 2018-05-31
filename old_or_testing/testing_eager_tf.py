import pandas as pd
import numpy as np
import tensorflow as tf
from timeit import default_timer as timer
from functions import rotate, perimeter_array, design_space_sample, normalize, de_normalize
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

df = pd.read_csv(r'C:\Users\georg\PycharmProjects\Bambi\bambi_test2.csv')
df = df.drop(columns=['Factor Name', ])  # drop experiment run id.

# print(df)

df_head = df[0:3].reset_index().copy()
df_body = df[3:].reset_index().copy()
df_body = df_body.apply(pd.to_numeric, errors='ignore')

# print(df_head)

df_output = df_body.filter(regex=('OUT.*'))  # selects OUT**** to create DF of outputs
df_output = df_output.apply(pd.to_numeric, errors='ignore')

# df_input = df_body[df_body.columns.difference(df_output.columns)] #strips outputs to create input DF
df_input = df_body.drop(columns=['index', ])
df_input = df_input.drop(columns=list(df_output))  # drop output column names

# print(df_input)

df_head = df_head.drop(columns=['index', ])
df_head = df_head.drop(columns=list(df_output))

# print(df_head)

types = df_head.loc[[0]].values.flatten()
mins = df_head.loc[[1]].values.flatten().astype(np.float32)  # rows to #'s in np array
maxes = df_head.loc[[2]].values.flatten().astype(np.float32)

ins = df_input.values
norm_vector_in = np.linalg.norm(ins, axis=0, ord=2) / (ins.shape[1])

ins = ins / norm_vector_in
ins = ins.astype(np.float32)

outs = df_output.values  # pd df to numpy
norm_vector_out = np.linalg.norm(outs, axis=0, ord=2) / (outs.shape[1])
outs = outs / norm_vector_out
outs = outs.astype(np.float32)

num_factors = df_input.shape[1]
num_outs = df_output.shape[1]

###model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(num_factors, activation=tf.nn.softplus, input_shape=(num_factors,)),
    tf.keras.layers.Dense(num_factors, activation=tf.nn.softplus),
    tf.keras.layers.Dense(num_factors, activation=tf.nn.softplus),
    tf.keras.layers.Dense(num_factors, activation=tf.nn.softplus),
    tf.keras.layers.Dense(num_outs, activation=tf.nn.softplus),
])


def loss(model, x, y):
    y_ = model(x)
    return tf.losses.huber_loss(labels= y_, predictions = y)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)


optimizer = tf.train.AdadeltaOptimizer(epsilon=.000001)


def train_input_fn(ins, outs, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((ins, outs))
    dataset = dataset.shuffle(1000).repeat().batch(10)
    return dataset


if True:  # run training
    train_loss_results = []
    train_accuracy_results = []

    num_epochs = 100000


    loop_start = timer()

    for epoch in range(num_epochs):
        data = train_input_fn(ins, outs, int(0.6*len(ins[:,1]))) #batch as fraction of total experiments, reset every epoch
        # iterator = data.make_one_shot_iterator()
        # next_element = iterator.get_next()

        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()
        #optimize
        #
        # x, y = iter(data).next()
        # print("Initial loss: {:.3f}".format(loss(model, x, y)))

        for i,(x, y) in enumerate(data):  # rebatch every epoch
            # print(int(0.6*len(ins[:,1])))
            # print(epoch)
            grads = grad(model,x, y)
            optimizer.apply_gradients(zip(grads, model.variables),
                                      global_step = tf.train.get_or_create_global_step())

            #track progress
            epoch_loss_avg(loss(model, x, y))
            if i > 2 * len(ins[:,1]):  #have to manually break loop because dataset iterator is fucked
                break

        #end epoch
        train_loss_results.append(epoch_loss_avg.result())
        loop_end = timer()
        delta_t = loop_end-loop_start
        if epoch % 25 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Time: {:.3f}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                delta_t))









    # for i in range(500):  # 50000):
    #     tfe.run(train_step, feed_dict={x: ins[train_i],
    #                                    y_: outs[train_i]})

#
#
#
# model = tf.keras.Sequential([
#   tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),  # input shape required
#   tf.keras.layers.Dense(10, activation="relu"),
#   tf.keras.layers.Dense(3)
# ])
#
# def loss(model, x, y):
#   y_ = model(x)
#   return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
#
#
# def grad(model, inputs, targets):
#   with tf.GradientTape() as tape:
#     loss_value = loss(model, inputs, targets)
#   return tape.gradient(loss_value, model.variables)
#
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#
# train_loss_results = []
# train_accuracy_results = []
#
# num_epochs = 201
#
# for epoch in range(num_epochs):
#     epoch_loss_avg = tfe.metrics.Mean()
#     epoch_accuracy = tfe.metrics.Accuracy()
#
#     # Training loop - using batches of 32
#     for x, y in train_dataset:
#         # Optimize the model
#         grads = grad(model, x, y)
#         optimizer.apply_gradients(zip(grads, model.variables),
#                                   global_step=tf.train.get_or_create_global_step())
#
#         # Track progress
#         epoch_loss_avg(loss(model, x, y))  # add current batch loss
#         # compare predicted label to actual label
#         epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)
#
#     # end epoch
#     train_loss_results.append(epoch_loss_avg.result())
#     train_accuracy_results.append(epoch_accuracy.result())
#
#     if epoch % 50 == 0:
#         print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
#                                                                     epoch_loss_avg.result(),
#                                                                     epoch_accuracy.result()))
