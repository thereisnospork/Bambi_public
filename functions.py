# import pandas as pd
import numpy as np
# import scipy as sp
# import tensorflow as tf
import itertools
# from timeit import default_timer as timer

# tf.enable_eager_execution()
# start_nontf = timer()
# ####################READ IN CSV PARTITION INTO HEAD BODY INPUT OUTPUTS##################

# df = pd.read_csv(r'C:\Users\georg\PycharmProjects\Bambi\bambi_test2.csv')
# df = df.drop(columns = ['Factor Name',]) #drop experiment run id.
#
# # print(df)
#
# df_head = df[0:3].reset_index().copy()
# df_body = df[3:].reset_index().copy()
# df_body = df_body.apply(pd.to_numeric, errors = 'ignore')
#
# # print(df_head)
#
# df_output = df_body.filter(regex=('OUT.*')) #selects OUT**** to create DF of outputs
# df_output = df_output.apply(pd.to_numeric, errors = 'ignore')
#
# # df_input = df_body[df_body.columns.difference(df_output.columns)] #strips outputs to create input DF
# df_input = df_body.drop(columns=['index',])
# df_input = df_input.drop(columns=list(df_output)) #drop output column names
#
# # print(df_input)
#
# df_head = df_head.drop(columns=['index',])
# df_head = df_head.drop(columns=list(df_output))
#
# # print(df_head)
#
# types = df_head.loc[[0]].values.flatten()
# mins = df_head.loc[[1]].values.flatten().astype(np.float32)  # rows to #'s in np array
# maxes = df_head.loc[[2]].values.flatten().astype(np.float32)
#
# ins = df_input.values
# norm_vector_in = np.linalg.norm(ins, axis = 0, ord = 2)/(ins.shape[1])
#
# ins = ins / norm_vector_in
#
# outs = df_output.values #pd df to numpy
# norm_vector_out = np.linalg.norm(outs, axis = 0, ord = 2)/(outs.shape[1])
# outs = outs / norm_vector_out


def rotate(l, x):
    x = x%len(l)
    return l[-x:] + l[:-x]


def perimeter_array(mins, maxes, types):
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

def design_space_sample(mins, maxes, types, samples, mix_sum):
    """replaces old design_space_sample

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
            cat_level_range = range(cat_min, cat_max +1) #+1 for non-inclusivity of range
            cat_levels.append(cat_level_range)
        else:
            cat_levels.append(0) #placeholder 0

    for i, type in enumerate(types):
        if type == 'CONTINUOUS':
            out_arr[i, :] = np.random.uniform(mins[i],maxes[i],[1,samples])
        if type == 'CATEGORICAL':
            out_arr[i, :] = np.random.choice(cat_levels[i],size = [1,samples])
        if type == 'MIX':
            out_arr[i, :] = np.random.uniform(mins[i], maxes[i], [1, samples]) #need to normalize to mix sum

    mix_sub_array = out_arr[mix_bool]
    mix_sub_sums = np.sum(mix_sub_array, axis = 0)
    mix_sub_norm = mix_sum / mix_sub_sums
    out_arr[mix_bool] = mix_sub_array * mix_sub_norm  # ###works, but doesn't respect ranges exactly.
                                                    # will respect ranges when generating new experiments
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


def normalize(in_arr, norm_vector):
    """returns normalized vector in same shape as input based on the normalization vector
    vector length must match input row length"""
    in_arr = in_arr / norm_vector[:, np.newaxis]  # normalize
    # in_arr = in_arr.transpose() ###not necessary????
    return in_arr


def de_normalize(in_arr, norm_vector):
    """returns de-normalized array in same shape as input based on the normalization vector
    vector length must match input row length"""
    in_arr = in_arr * norm_vector  # normalize
    # in_arr = in_arr.transpose()
    return in_arr

