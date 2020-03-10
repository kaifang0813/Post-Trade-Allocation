import numpy as np
import pandas
import datetime
import copy
from operator import add
import math
import multiprocessing as mp
import time

def sanity_check_on_net_pos_acct(net_pos_acct, net_pos_acct_new, scale, qt):
    """
    Function to make sure that position allocated to each account has same sign as the original
    trade, for example, we would not allow [2,2,2,-1,0] or [-2,-2,-2,1,0] but would allow
    [-2,-2,-3,0] or [2,2,3,0]

    Parameters:
    net_pos_acct: list
        the net position of each account
    net_pos_acct_new: list
        the new position after add the allocation of the new trade (inital guess of next allocation result)
    scale: int
        if buy, 1, else -1
    qt: int
        the quantity of trade/order

    Output:
    net_pos_acct_new: list
        the allocation of position that pass the sanity check
    """
    diff_position = [x - y for x, y in zip(net_pos_acct_new, net_pos_acct)]
    diff_position_p = list(map(lambda x: 1 if x < 0 else 0, diff_position))
    diff_position_n = list(map(lambda x: 1 if x > 0 else 0, diff_position))

    if scale > 0 and sum(diff_position_n) > 0:
        net_pos_acct_new = [y if x < y else x for x, y in zip(net_pos_acct_new, net_pos_acct)]
        temp = [x - y for x, y in zip(net_pos_acct_new, net_pos_acct)]
        while sum(temp) != scale * qt:
            index_j = temp.index(max(temp))
            net_pos_acct_new[index_j] = net_pos_acct_new[index_j] - 1
            temp = [x - y for x, y in zip(net_pos_acct_new, net_pos_acct)]

    elif scale < 0 and sum(diff_position_p) > 0:
        net_pos_acct_new = [y if x > y else x for x, y in
                            zip(net_pos_acct_new, net_pos_acct)]
        temp = [x - y for x, y in zip(net_pos_acct_new, net_pos_acct)]

        while sum(temp) != scale * qt:
            index_j = temp.index(min(temp))
            net_pos_acct_new[index_j] = net_pos_acct_new[index_j] + 1
            temp = [x - y for x, y in zip(net_pos_acct_new, net_pos_acct)]

    return net_pos_acct_new


def criterion(pnl_acct_sof_far, cum_pnl, af, option):
    """ objective function of optimization
        Parameters:
        cum_pnl: float
            cumulative pnl at period j/ current period
        pnl_acct_sof_far: list
            the pnl for each acctount so far
        af: list
            allocation factor for each account
        options: str
            abreviation for objective function

        Output:
        mae: float
            the objective value
    """

    mae = 0
    n_acct = len(af)
    if option == '0':
        mae = sum([(pnl_acct_sof_far[i] / af[i] - pnl_acct_sof_far[j] / af[j]) ** 2
                   for i in range(0, n_acct) for j in range(0, n_acct)])
    elif option == '1':
        mae = sum([(pnl_acct_sof_far[i] / af[i] - cum_pnl) ** 2 for i in range(0, n_acct)])
    elif option == '2':
        mae = sum([(pnl_acct_sof_far[i] - af[i] * cum_pnl) ** 2 for i in range(0, n_acct)])
    elif option == '3':
        mae = sum([abs(pnl_acct_sof_far[i] - af[i] * cum_pnl) for i in range(0, n_acct)])
    else:
        raise ValueError('Do not support Option {}'.format(option))

    return mae


def grid_search_optimization(cum_pnl_j, net_pos_j_1, net_pos_acct, af, qty,
                             pnl_acct_so_far, price_j_1, price_j, multiplier,
                             buy_or_sell, criteria_option, sanity_on_net_position):
    """
    Parameters:
        cum_pnl_j: float
            cumulative pnl at period j/ current period
        net_pos_j_1: int
            the net position at period j-1 / previous period
        net_pos_acct: list
            the net position for each account
        af: list
            allocation factor for each account
        qty: int
            the quantity of the trade, absolute value
        pnlAcctSoFar: list
            the pnl of each account accumalated so far
        price_j_1: float
            the trade price at period j-1/ previous period
        price_j: float
            the trade price at period j/ current period
        multiplier: int
            lot size of the asset, quantity multiplier
        buy_or_sell: int
            1 for buy, -1 for sell
        criteria_option: str
            which optimization method to use
        sanity_on_net_position:


    """
    sign_net_pos = 1 if net_pos_j_1 >= 0 else -1
    para = [x * abs(net_pos_j_1) for x in af]
    initial_guess = [round(x) for x in para]
    while (sum(initial_guess) != abs(net_pos_j_1)):
        adjust = [abs(x-y) for x, y in zip(para, initial_guess)]
        index = adjust.index(max(adjust))
        if sum(initial_guess) > abs(net_pos_j_1):
            # use value before round to find the max
            initial_guess[index] = initial_guess[index] - 1
        else:
            initial_guess[index] = initial_guess[index] + 1

    net_pos_acct_new = [sign_net_pos * i for i in initial_guess]
    net_pos_acct_new = sanity_check_on_net_pos_acct(net_pos_acct, net_pos_acct_new, buy_or_sell, qty)

    parameters_set = {}
    parameters_set['Optimal Allocation'] = [abs(x - y) for x, y in zip(net_pos_acct_new, net_pos_acct)]
    parameters_set['net_pos_acct_new'] = net_pos_acct_new
    parameters_set['net_pos_acct'] = net_pos_acct
    parameters_set['sign_net_pos'] = sign_net_pos
    parameters_set['allocation factor'] = af
    parameters_set['cum_pnl_j'] = cum_pnl_j
    parameters_set['pnl_acct_so_far'] = pnl_acct_so_far
    parameters_set['buy_or_sell'] = buy_or_sell
    parameters_set['sanity_on_net_position'] = sanity_on_net_position
    parameters_set['price_j_1'] = price_j_1
    parameters_set['price_j'] = price_j
    parameters_set['multiplier'] = multiplier
    parameters_set['qty'] = qty
    pnl_acct_so_far_new = pnl_acct_so_far + [(price_j - price_j_1) * multiplier * x for x in net_pos_acct_new]
    parameters_set['criteria_option'] = criteria_option

    parameters_set['pnl_acct_so_far_new'] = pnl_acct_so_far_new
    parameters_set['calculate'] = 0


    # parameters_sets = []
    # vec_sets = [ [1, -1], [2, -2], [3, -3]]
    #
    # for vec_set in vec_sets:
    #     para_set = copy.deepcopy(parameters_set)
    #     para_set['pertub_vec'] = vec_set
    #     parameters_sets.append(para_set)
    #
    # pool = mp.Pool(processes=(mp.cpu_count() - 1))
    # results = pool.map(optimizing_pnl_by_perturbing_net_pos, parameters_sets)
    # pool.close()
    # pool.join()
    #
    # return results[1]
    # #     this part should be paralled: and we need to compare the mae of the result to see which one is the best one

    parameters_set['pertub_vec'] = [1, -1]
    parameters_set = optimizing_pnl_by_perturbing_net_pos(parameters_set);

    # parameters_set['pertub_vec'] = [1, -1, 1, -1]
    # parameters_set = optimizing_pnl_by_perturbing_net_pos(parameters_set);

    # parameters_set['pertub_vec'] = [-1, -1, 2]
    # parameters_set = optimizing_pnl_by_perturbing_net_pos(parameters_set);
    #
    # parameters_set['pertub_vec'] = [1, 1, -2]
    # parameters_set = optimizing_pnl_by_perturbing_net_pos(parameters_set);
    #
    # parameters_set['pertub_vec'] = [2, -2]
    # parameters_set = optimizing_pnl_by_perturbing_net_pos(parameters_set);
    #
    # parameters_set['pertub_vec'] = [3, -3]
    # parameters_set = optimizing_pnl_by_perturbing_net_pos(parameters_set);
    #
    # parameters_set['pertub_vec'] = [1, 2, -3]
    # parameters_set = optimizing_pnl_by_perturbing_net_pos(parameters_set);
    #
    # parameters_set['pertub_vec'] = [3, -1, -2]
    # parameters_set = optimizing_pnl_by_perturbing_net_pos(parameters_set);

    return parameters_set


def sanity_check_on_net_pos_acct_v(net_pos_acct, net_pos_acct_new, scale, qt):
    """
    This is the vectorized version of sanity check on net position, for more readable code, please check on
    sanity_check_on_net_pos_acct, the following code is optimized to use numpy matrix calculation to down size
    O(n) problem to O(1) problem

    Function to make sure that position allocated to each account has same sign as the original
    trade, for example, we would not allow [2,2,2,-1,0] or [-2,-2,-2,1,0] but would allow
    [-2,-2,-3,0] or [2,2,3,0]

    Parameters:
    net_pos_acct: 1-dimension numpy array
        the net position of each account
    net_pos_acct_new: 2-dimension numpy array
        the new position after add the allocation of the new trade (inital guess of next allocation result)
    scale: int
        if buy, 1, else -1
    qt: int
        the quantity of trade/order

    Output:
    net_pos_acct_new: list
        the allocation of position that pass the sanity check
    """
    x, y = net_pos_acct_new.shape
    net_pos_acct = np.array([net_pos_acct] * x)
    temp_array = np.array([list(range(0, y))] * x)

    if scale > 0:
        # positive position
        positive_pos = np.maximum(net_pos_acct, net_pos_acct_new)
        use_max = (np.where(net_pos_acct_new < net_pos_acct, 1, 0))
        max_filter = np.where(sum(np.transpose(use_max)), 1, 0)

        max_orig = np.transpose(np.transpose(net_pos_acct) * max_filter)
        max_new = np.transpose(np.transpose(positive_pos) * max_filter)
        max_diff_pos = max_new - max_orig
        positive_loop = np.transpose(np.transpose(positive_pos) * max_filter)
        loop_filter = np.where((sum(np.transpose(max_diff_pos)) == max_filter * qt * scale) != True, 1, 0)
        # this loop could take a long time if the vector is not well designed
        count = 0
        while sum(loop_filter) > 0:
            positive_update = np.transpose(np.where(np.transpose(temp_array) == np.argmax(max_diff_pos, axis=1), 1, 0))
            positive_update = np.transpose(np.transpose(positive_update) * loop_filter)
            positive_loop = np.transpose(np.transpose(positive_loop - positive_update) * max_filter)
            max_diff_pos = positive_loop - max_orig
            loop_filter = np.where((sum(np.transpose(max_diff_pos)) == max_filter * qt * scale) != True, 1, 0)
            count = count + 1
            if count > 10:
                print('loop too much, there must be an error')

        other_filter = np.where(max_filter == 1, 0, 1)
        other_pos = np.transpose(np.transpose(net_pos_acct_new) * other_filter)
        result = positive_loop + other_pos

    if scale < 0:
        # negative position
        negative_pos = np.minimum(net_pos_acct, net_pos_acct_new)
        use_min = (np.where(net_pos_acct_new > net_pos_acct, 1, 0))
        min_filter = np.where(sum(np.transpose(use_min)), 1, 0)
        min_orig = np.transpose(np.transpose(net_pos_acct) * min_filter)
        min_new = np.transpose(np.transpose(negative_pos) * min_filter)
        min_diff_pos = min_new - min_orig
        negative_loop = np.transpose(np.transpose(negative_pos) * min_filter)
        loop_filter = np.where((sum(np.transpose(min_diff_pos)) == min_filter * qt * scale) != True, 1, 0)
        count = 0
        while sum(loop_filter) > 0:

            negative_update = np.transpose(np.where(np.transpose(temp_array) == np.argmin(min_diff_pos, axis=1), 1, 0))
            negative_update = np.transpose(np.transpose(negative_update) * loop_filter)
            negative_loop = np.transpose(np.transpose(negative_loop + negative_update) * min_filter)
            min_diff_pos = negative_loop - min_orig
            loop_filter = np.where((sum(np.transpose(min_diff_pos)) == min_filter * qt * scale) != True, 1, 0)
            count = count + 1
            if count > 10:
                print('loop too much, there must be an error')

        other_filter = np.where(min_filter == 1, 0, 1)
        other_pos = np.transpose(np.transpose(net_pos_acct_new) * other_filter)
        result = negative_loop + other_pos
    return result


def criterion_v(pnl_acct_sof_far, cum_pnl, af, option):
    """ objective function of optimization
        Parameters:
        cum_pnl: float
            cumulative pnl at period j/ current period
        pnl_acct_sof_far: 2 - Dimension numpy array
            the pnl for each acctount so far
        af: 1 dimension numpy array
            allocation factor for each account
        options: str
            abreviation for objective function

        Output:
        mae: 1 dimension numpy array
            the objective value
    """
    x, y = pnl_acct_sof_far.shape
    n_acct = y
    cum_pnl = np.full(pnl_acct_sof_far.shape, cum_pnl)

    if option == '0':
        temp = np.transpose(pnl_acct_sof_far / af)
        mae = sum([np.power(temp[i] - temp[j], 2)
                   for i in np.arange(n_acct) for j in np.arange(n_acct)])
    elif option == '1':
        mae = np.sum(np.power(pnl_acct_sof_far / af - cum_pnl, 2), axis=1)
    elif option == '2':
        mae = np.sum(np.power(pnl_acct_sof_far - cum_pnl * af, 2), axis=1)
    elif option == '3':
        mae = np.sum(np.absolute(pnl_acct_sof_far - cum_pnl * af), axis=1)

    else:
        raise ValueError('Do not support Option {}'.format(option))

    return mae


def generate_grid(net_pos_acct_new, pertub_vec):
    """Generate a 2-dimension numpy array that represent possibly allocation which would be used in grid search to find best allocation
        Parameters:
            net_pos_acct_new: 1-dimensional numpy arrary
                The starting point to generate the grid, contain information of how we net position of each account after we allocate the new trade
            pertub_vector: 1-dimensional numpy arrary
                vector to tell use how to generate the grid
        Return:
            two dimensional numpy array, each row represent a case in the grid search
    """
    tic = time.clock()
    grid = [net_pos_acct_new]
    n_account = len(net_pos_acct_new)
    n_account_range = np.arange(n_account)
    if len(pertub_vec) == 2:
        for i in n_account_range:
            for j in n_account_range:
                if i != j:
                    action = np.zeros(n_account).astype(int)
                    action[i] = pertub_vec[0]
                    action[j] = pertub_vec[1]
                    grid.append(net_pos_acct_new + action)

    elif len(pertub_vec) == 3:
        for i in n_account_range:
            for j in n_account_range:
                for z in n_account_range:
                    if len(set([i, j, z])) == 3:
                        action = np.zeros(n_account).astype(int)
                        action[i] = pertub_vec[0]
                        action[j] = pertub_vec[1]
                        action[z] = pertub_vec[2]
                        grid.append(net_pos_acct_new + action)
    elif len(pertub_vec) == 4:
        for i in n_account_range:
            for j in n_account_range:
                for z in n_account_range:
                    for w in n_account_range:
                        if len(set([i, j, z, w])) == 4:
                            action = np.zeros(n_account).astype(int)
                            action[i] = pertub_vec[0]
                            action[j] = pertub_vec[1]
                            action[z] = pertub_vec[2]
                            action[w] = pertub_vec[3]
                            grid.append(net_pos_acct_new + action)
    else:
        raise ValueError('only support pertub_vec of length 2,3,4 for now')

    if len(set(sum(np.transpose(grid)))) > 1:
        raise ValueError("the grid generated has different net position side in different row, something must be wrong")
    tok = time.clock()
    print(tok-tic)
    return grid


def entropy_v(data):
    """
    Vectorized verison of entropy calculation
    Get a 2 dimension numpy array, return a 1 dimension numpy array that represent the entropy of each row

    Parameters:
        data: two dimensional numpy array representing net account position in each case of the grid search

    Return:
        entropy: one dimensional numpy array, represent entropy for each case of grid search
    """
    x, y = data.shape
    n_acct = y
    temp = np.transpose(data)
    entropy = sum([np.power(temp[i] - temp[j], 2)
                   for i in np.arange(n_acct) for j in np.arange(n_acct)])
    return entropy


def optimizing_pnl_by_perturbing_net_pos(parameters_set):
    """Do Perurbing Net Position Optimization given parameters set which contain detail information of each trade and current pnl """

    net_pos_acct_new = np.array(parameters_set['net_pos_acct_new'])
    net_pos_acct = np.array(parameters_set['net_pos_acct'])
    scale = parameters_set['buy_or_sell']
    af = np.array(parameters_set['allocation factor'])
    sgn_net = parameters_set['sign_net_pos']
    cum_pnl = parameters_set['cum_pnl_j']
    pnl_acct_so_far = np.array(parameters_set['pnl_acct_so_far'])
    price_j_1 = parameters_set['price_j_1']
    price_j = parameters_set['price_j']
    multiplier = parameters_set['multiplier']
    qty = parameters_set['qty']
    criteria = parameters_set['criteria_option']
    pertub_vec = parameters_set['pertub_vec']
    grid = generate_grid(net_pos_acct_new, pertub_vec)
    # vectorization computing:
    net_pos_acct_new = np.array(grid).astype(int)
    check_alloc = sanity_check_on_net_pos_acct_v(net_pos_acct, net_pos_acct_new, scale, qty)
    check_alloc =  np.unique(check_alloc, axis =0)
    # np.unique(check_alloc, axis =0) not working in some numpy version
    x, y = check_alloc.shape
    net_pos_acct = np.array([net_pos_acct] * x)
    pnl_acct_so_far_new = np.array([pnl_acct_so_far] * x) + check_alloc * (price_j - price_j_1) * multiplier
    mae = criterion_v(pnl_acct_so_far_new, cum_pnl, af, criteria);
    min_index = np.where(mae == mae.min())[0]

    # if multiple allocation has the same kind of result, have a one more step allocation to choose the lowest entropy allocation
    if len(min_index) > 1:
        fur_optimize = check_alloc[min_index, :]
        pnl = pnl_acct_so_far_new[min_index, :]
        pos_entropy = entropy_v(fur_optimize / af)
        min_e = np.where(pos_entropy == pos_entropy.min())[0]
        net_pos_acct_new = fur_optimize[min_e, :][0].tolist()
        parameters_set['net_pos_acct_new'] = net_pos_acct_new
        parameters_set['pnl_acct_so_far_new'] = pnl[min_e, :][0].tolist()
        parameters_set['Optimal Allocation'] = (net_pos_acct_new - net_pos_acct[0]).tolist()
    else:
        net_pos_acct_new = check_alloc[min_index, :][0]
        parameters_set['net_pos_acct_new'] = net_pos_acct_new.tolist()
        parameters_set['pnl_acct_so_far_new'] = pnl_acct_so_far_new[min_index, :][0].tolist()
        parameters_set['Optimal Allocation'] = (net_pos_acct_new - net_pos_acct[0]).tolist()

        parameters_set['calculate'] = parameters_set['calculate'] + 1

    return parameters_set