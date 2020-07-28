import numpy as np
import esig.tosig as ts
import math
import random
import itertools
from . import auxiliaryfunct as ax

"""
This file contains unit tests of the esig library as well as auxiliary functions needed to make the tests run.

The tests are run each time esig is installed, to make sure the installation works correctly.

"""


def compare_reverse_test(l,a,dims,degs):
    """ Compare signature of a random path with its inverse

    If a path has steps [a,b,c], its inverse has steps [c,b,a]

    Args:
        l (int): length of the random path to be produced
        a (list): list of modifiers applied at each step of the random path
        dims (int): number of dimensions each path step has
        degs (int): number of signature degrees to be used in the test

    Returns:
        str: "ok" if the test has been successful

    Raises:
        ValueError: If the signature of the inverse path is not actually computed for the inverse of the original path

    Example:
        >>> compare_reverse_test(100,[-1,0,1],2,4)
        "ok"
    """

    input = np.array(ax.random_path(l,a,dims))

    # make a reverse path
    rev = list(reversed(input))
    rev = [list(x) for x in rev]
    rev = np.array(rev)
    out_sig = ts.stream2sig(input,degs)
    out_rev = ts.stream2sig(rev,degs)

    # calculate corresponding 'k' addresses for the signature elements of the original and the inverse paths
    pos = ts.sigkeys(dims,degs).split()
    how_long = []
    pos_out = []
    pos_rev = []
    for row in pos:
        row = row.strip('(')
        row = row.strip(')')
        row = row.split(',')
        pos_out.append(''.join(row))
        pos_rev.append(''.join(list(reversed(row))))
        if len(row) % 2 == 0: # even key lengths
            how_long.append(-1)
        else: # odd key lengths
            how_long.append(1)

    # match addresses for the signature elements of the original and the inverse paths
    out_and_rev = []
    for ind, row in enumerate(pos_out):
        temp = []
        temp.append(out_sig[ind])
        temp.append(out_rev[pos_rev.index(row)])
        temp.append(how_long[ind])
        out_and_rev.append(temp)

    # check if the differences are correct (corresponding addresses of odd length have opposite signs, corresponding addresses with even length have the sae signs)
    out = []
    for row in out_and_rev:
        test_calculation = row[-1]*row[-3]+row[-2]
        out.append(test_calculation)

    if out[0] != 2:
        raise ValueError
    for row in out[1:]:
        if row >= -0.00001 and row <= 0.00001:
            pass
        else:
            raise ValueError
    return 1


def shuffle_test (l,a,dims,degs):

    """ Evaluates whether a product sum of shuffles is equal to the multiplication evaluated pairs of signature elements

    stream2sig and stream2logsig functions included in the esig library return a list of values according to (a) the dimensions present in the data and (b) signature degrees.
    sigkeys and logsigkeys functions included in the esig library return keys which correspond to each of the signature element produced with either stream2sig and stream2logsig functions.

    Properties of path signature are such that for any 2 keys, if one multiplies the values which they correspond to, the result of the multiplication is equal to a sum of
    values for all keys which are shuffles of the two keys. This function evaluates whether this property of path signature holds true for all possible pairs of
    signature elements whose keys, together, are shorter or as long as the longest keys of values included in the signature.

    Args:
        l (int): length of a random path to be produced
        a (list): list of modifiers applied at each step of the random path
        dims (int): number of dimensions each path step has
        degs (int): number of signature degrees to be used in the test

    Returns:
        str: "ok" if the test has been successful

    Raises:
        ValueError: If the sum of values corresponding to key shuffles is not equal to the multiplication of the pair of values with corresponding keys

    Example:
        >>> shuffle_test(100,[-1,0,1],3,3)
        "ok"

    """

    test_path = np.array(ax.random_path(l,a,dims)) # generates a random path given number of path steps 'l', possible moves at each step 'a',
                                                # and the number dimensions at each path step 'dims'
    signature = list(ts.stream2sig(test_path,degs)) # generates signature for the provided random path and chosen number of signature degrees 'degs'

    #5 LINES BELOW COULD BE A SEPARATE TEST
    if signature[0] != 1: # check if the first signature element is 1
        print("The first signature element has a value different from 0 in shuffle_test")
        raise ValueError
    else:
        pass

    keys = ts.sigkeys(dims,degs).split() # generates keys of the signature for the provided random path
    keys2 = []
    lengths = []
    for row in keys:
        row = row.strip('(')
        row = row.strip(')')
        if len(row) > 0:
            row = row.split(',')
            lengths.append(len(row))
            keys2.append(row)
        else:
            pass

    del signature[0]

    keys3 = list(zip(keys2,lengths,signature))




    max_mutant_length = len(keys2[-1])

    # obtain all possible mutations which are no longer than the max_mutant_length
    mutants = []
    mutant_parent_indices = []
    for row_ind, row in enumerate(keys3):
        vals = []
        inds = []
        for col_ind, col in enumerate(keys3):
            if row_ind != col_ind:
                tmp = list(row[0]) + list(col[0])
                if len(tmp) <= max_mutant_length:
                    val = [list(row[0]), list(col[0])]
                    vals.append(val)
                    ind = [row_ind,col_ind]
                    inds.append(ind)
        mutants.extend(vals)
        mutant_parent_indices.extend(inds)


    # generate shuffles, get results
    shuffle_prep = list(zip(mutants,mutant_parent_indices))
    for row in shuffle_prep:
        my_shuffles = ax.shuffles(row[0][0],row[0][1])

        # sum shuffles, knowing their values in keys3 table
        shuffle_product = 0
        for shuffle in my_shuffles:
            for x in keys3:
                if str(x[0]) != str(shuffle):
                    pass
                else:
                    shuffle_product = shuffle_product + x[2]

        # multiply components, knowing their indics in keys3 table
        parent1 = row[1][0]
        parent1 = keys3[parent1][2]
        parent2 = row[1][1]
        parent2 = keys3[parent2][2]
        mutant_parent_multiplication = parent1*parent2

        # shuffle product - multiplication product into 'solutions' variable
        solution = shuffle_product-mutant_parent_multiplication
        if solution >= -0.00001 and solution <= 0.00001:
            pass
        else:
            raise ValueError
    return 1



# reparametrizing test
def reparametrizing_test(l,a,dims,degs,reparam_factor):
    """ Checks if the path signature for a reparametrized path is identical to the original signature.

    The test works for any path which is linear everywhere except of a list of inflexion points.
    New steps are added which fall on the lines connecting the steps of the inputted path. The overall shape of the path is the same as a result


    Args:
        l (int): length of a random path to be produced
        a (list): list of modifiers applied at each step of the random path
        dims (int): number of dimensions each path step has
        degs (int): number of signature degrees to be used in the test
        reparam_factor (int): factor by which the number of steps is increased along the existing path

    Returns:
        str: "ok" if the test has been successful

    Raises:
        ValueError: If reparametrized path put into stream2sig function resulted in a different path

    Examples:
        >>> print (reparametrizing_test(10,[-1,0,1],2,3,3))
        "ok"

    """
    path_orig = ax.random_path(l,a,dims) # generates a random path given number of path steps 'l', possible moves at each step 'a',
                                                    # and the number dimensions at each path step 'dims'
    mid_step = 1.0/reparam_factor
    # obtaining additional mid-steps
    path_reparam = []
    for ind, step in enumerate(path_orig):
        path_reparam.append(step)
        next_step = ind + 1
        change = []
        if next_step < len(path_orig):
            change = [x1 - x2 for (x1, x2) in zip(path_orig[next_step], step)]
            for x in range(1,reparam_factor):
                mid_step_vals = np.array(change)*mid_step*x + step
                path_reparam.append(list(mid_step_vals))

    orig_signature = list(ts.stream2sig(np.array(path_orig), degs))
    reparam_signature = list(ts.stream2sig(np.array(path_reparam), degs))
    tmp = zip(orig_signature,reparam_signature)

    solutions = [x1 - x2 for (x1, x2) in tmp]
    for solution in solutions:
        if solution >= -0.00001 and solution <= 0.00001:
            pass
        else:
            raise ValueError
    return 1
