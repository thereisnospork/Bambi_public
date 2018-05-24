# import pandas as pd
import numpy as np
# import scipy as sp
import tensorflow as tf
import itertools
from scipy.spatial.distance import pdist, squareform
# from timeit import default_timer as timer

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
            cat_min = np.int(np.round(mins[i]))  #maybe rewrite with cat_level funciton/dict method?
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

def design_space_sample_exact(mins, maxes, types, samples, mix_sum, cat_ratio_dict = False):
    """
    comparable to design_space_sample except! that it fully respects mixture bounds.

    :param mins: ndarray 1d of mins for factors
    :param maxes: ndarray 1d of maxes for factors
    :param types: ndarray of str for type of factor.  continuous categorical mix are the allowed choices
    :param samples: how many designs to generate
    :param mix_sum: sum of mixture variables, found by mix_sum method
    :return: len(mins) x samples ndarray of floats representing samples chosen random uniformely from design space
    """

    raw_design = design_space_sample(mins, maxes, types, samples, mix_sum, )
    mix_bool = (types == 'MIX')
    cat_bool = (types == 'CATEGORICAL')
    mix_sub_array = raw_design[mix_bool]
    cat_sub_array = raw_design[cat_bool]
    mix_mins = mins[mix_bool]
    mix_maxes = maxes[mix_bool]

    flag = True #continues through array over and over till no values found to be excessive
    while flag:
        flag = False
        for i, column in enumerate(mix_sub_array):
            for n, item in enumerate(column):
                if item < mix_mins[i] or item > mix_maxes[i]:
                    mix_sub_array[i,n] = np.random.uniform(mix_mins[i], mix_maxes[i])    #mix_mins[i]+item/(mix_mins[i]+1) #divide by zero potential error. maybe need to rethink formula here
                    flag = True
        if cat_ratio_dict is not False:

            for i, _ in enumerate(types):
                if cat_bool[i]:
                    cat_levels = cat_ratio_dict[i].keys()
                    cat_levels = list(map(np.int, cat_levels))
                    cat_p = cat_ratio_dict[i].values()
                    cat_p = list(cat_p)
                    cat_p = cat_p / np.sum(cat_p)
                    raw_design[i,:] = np.random.choice(cat_levels, p=cat_p)



    #renormalize

        mix_sub_sums = np.sum(mix_sub_array, axis=0)
        mix_sub_norm = mix_sum / mix_sub_sums

    raw_design[mix_bool] = mix_sub_array * mix_sub_norm

    return raw_design


def optimal_design(mins, maxes, types, k, mix_sum, norm_ins, cat_dict):
    """
    returns a space-averaged design, normalizing respective dimensions,
    between mins and maxes respecting mixtures/categories
    returns unnormalized
    :param mins:
    :param maxes:
    :param types:
    :param k:
    :param mix_sum:
    :param norm_ins:
    :return:
    """
    #initial designs to trim:
    top_half_indicies = np.arange(0,k,1)
    unnormed_samples = design_space_sample_exact(mins, maxes, types, k*2, mix_sum)

    not_cat_bool = (types != 'CATEGORICAL') #dropping categorical factors from distance calcs.
    # ratios of cat will be determined from design_space_sample_exact method - needs refining to account for cat variables


    # norm_samples = normalize(init_samples, norm_ins)
    #loop it!
    for n in range(200):
        # print(n)
        # kept_samples = unnormed_samples[:,top_half_indicies]
        new_samples = design_space_sample_exact(mins, maxes, types, k, mix_sum, cat_ratio_dict= cat_dict)
        unnormed_samples[:,~top_half_indicies] = new_samples
            # = np.hstack((kept_samples, new_samples))

        normed = normalize(unnormed_samples, norm_ins)
        distances = pdist(np.transpose(normed[not_cat_bool]))  # m observation by n samples, so transpose input
        distances = squareform(distances)
        dist_totals = np.sum(distances, axis=0)  # matrix is symmetrical, so axis is irrelevant
        _, top_half_indicies = tf.nn.top_k(dist_totals, k)
        top_half_indicies = top_half_indicies.eval() #tensor to numpy array conversion #has to be here due to looping

    normed = unnormed_samples[:,top_half_indicies]

    return normed #return k samples the furthest away from all others.


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


def mins_maxes(ins):
    """returns two ndarrays representing mins, maxes for the subsetted
    design space from a selected list of experiments
    intended to take non-normalized values

    ins: ndarray ins with experimental conditions, shape [num_factors, num_samples]

    out: mins, maxes : two ndarray of floats.  categorical mins/maxes will be nominally integers (1.0, 2.0, etc.)
    """
    mins = np.zeros(ins.shape[0])
    maxes = np.zeros(ins.shape[0]) #preallocate outputs

    for i, _ in enumerate(ins):
        mins[i] = np.min(ins[i])
        maxes[i] = np.max(ins[i])

    return mins, maxes


def normalize(in_arr, norm_vector):
    """returns normalized vector in same shape as input based on the normalization vector
    vector length must match input row length"""
    in_arr = in_arr / norm_vector[:, np.newaxis]  # normalize
    # in_arr = in_arr.transpose() ###not necessary????
    return in_arr


def de_normalize(in_arr, norm_vector):
    """returns de-normalized array in same shape as input based on the normalization vector
    vector length must match input row length"""
    in_arr = in_arr * norm_vector[:, np.newaxis]  # normalize
    # in_arr = in_arr.transpose()
    return in_arr

def cat_ratios(ins, types, mins, maxes):
    """
    calculates relative percentages as decimal of each level of each categorical factor
    in a set of experiments(ins).  Intended for weighting of categorical factors in
    optimal design generation.
    :param ins: ndarray of experiments, inten
    :param types:
    :return: dict of dicts, first key is integer referring to which column the categorical value is
                first value is dict of values: probabilities for that column
    """
    cat_bool = (types == 'CATEGORICAL')
    dict_of_cat_ratio_dict = dict()
    for i, each in enumerate(types):
        # print(i)
        # print(cat_bool[i])
        if cat_bool[i]:
            levels_i = range(np.int(mins[i]), np.int(maxes[i])+1)
            temp = dict()
            for level in levels_i:
                temp[level] = 0 #incase next line defaults to False and doesn't eval
                temp[level] = (ins[i] == level).sum()
            dict_of_cat_ratio_dict[i] = temp


    # print(dict_of_cat_ratio_dict)
    return dict_of_cat_ratio_dict
