from numba import jit
import numpy as np
from allocation_optimization import sanity_check_on_net_pos_acct
from operator import add

@jit(nopython=True, cache = True)
def sanity_check_on_net_pos_acct(net_pos_acct, net_pos_acct_new, scale, qt):
    """
    Function to make sure that position allocated to each account has same sign as the original
    trade, for example, we would not allow [2,2,2,-1,0] or [-2,-2,-2,1,0] but would allow
    [-2,-2,-3,0] or [2,2,3,0]

    Parameters:
    net_pos_acct: numpy array
        the net position of each account
    net_pos_acct_new: numpy array
        the new position after add the allocation of the new trade (inital guess of next allocation result)
    scale: int
        if buy, 1, else -1
    qt: int
        the quantity of trade/order

    Output:
    net_pos_acct_new: list
        the allocation of position that pass the sanity check
    """
    diff_position = net_pos_acct_new - net_pos_acct
    if scale > 0 and np.sum(diff_position[np.where(diff_position < 0)]) > 0:
        net_pos_acct_new[np.where(diff_position < 0)] = net_pos_acct[np.where(diff_position < 0)]
        while np.sum(net_pos_acct_new - net_pos_acct) != scale * qt:
            temp = net_pos_acct_new - net_pos_acct
            index_j = np.argmax(temp)
            net_pos_acct_new[index_j] = net_pos_acct_new[index_j] - 1

    elif scale < 0 and np.sum(diff_position[np.where(diff_position > 0)]) > 0:
        # print(net_pos_acct_new)
        # print(net_pos_acct)
        # print(np.sum(net_pos_acct_new - net_pos_acct))
        net_pos_acct_new[np.where(diff_position > 0)] = net_pos_acct[np.where(diff_position > 0)]
        while np.sum(net_pos_acct_new - net_pos_acct) != scale * qt:
            # print(np.sum(net_pos_acct_new - net_pos_acct))
            # print(net_pos_acct_new)
            # print(net_pos_acct)
            temp = net_pos_acct_new - net_pos_acct
            index_j = np.argmin(temp)
            net_pos_acct_new[index_j] = net_pos_acct_new[index_j] + 1
    return net_pos_acct_new


@jit(nopython=True, cache = True)
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
    n_acct = af.size
    if option == '0':
        mae = 0
        n_account = pnl_acct_sof_far.size
        for i in np.arange(n_acct):
            for j in np.arange(n_acct):
                if i != j:
                    mae = mae + (pnl_acct_sof_far[i] / af[i] - pnl_acct_sof_far[j] / af[j]) ** 2
    elif option == '1':
        mae = np.sum(np.power(pnl_acct_sof_far / af - cum_pnl, 2))
    elif option == '2':
        mae = np.sum(np.absolute((pnl_acct_sof_far - af * cum_pnl) / (af * cum_pnl)))
    elif option == '3':
        mae = np.sum(np.power(pnl_acct_sof_far - af * cum_pnl, 2))
    elif option == '4':
        mae = np.sum(np.absolute(pnl_acct_sof_far - af * cum_pnl))

    return mae


@jit(nopython=True, cache = True)
def entropy(net_pos_acct):
    """
    Vectorized verison of entropy calculation
    Get a 2 dimension numpy array, return a 1 dimension numpy array that represent the entropy of each row

    Parameters:
        data: two dimensional numpy array representing net account position in each case of the grid search

    Return:
        entropy: one dimensional numpy array, represent entropy for each case of grid search
    """

    n_acct = net_pos_acct.size
    entropy = 0
    for i in np.arange(n_acct):
        for j in np.arange(i + 1, n_acct):
            entropy = entropy + (net_pos_acct[i] - net_pos_acct[j]) ** 2
    return entropy


@jit(nopython=True, cache = True)
def generate_grid(net_pos_acct_new, net_pos_acct, pnl_acct_so_far, sgn_net, scale, af,
                  qt, pertub_vec, multiplier, price_j, price_j_1, cum_pnl, optimal_allocation, final_pos_acct_new, final_pnl_acct_so_far_new, minMae):
    """Generate a 2-dimension numpy array that represent possibly allocation which would be used in grid search to find best allocation
        Parameters:
            net_pos_acct_new: 1-dimensional numpy arrary
                The starting point to generate the grid, contain information of how we net position of each account after we allocate the new trade
            pertub_vector: 1-dimensional numpy arrary
                vector to tell use how to generate the grid
        Return:
            two dimensional numpy array, each row represent a case in the grid search
    """
    n_account = len(net_pos_acct_new)
    n_account_range = np.arange(n_account)
    mae = 0
    flag = 0
    qtAcct = np.zeros(n_account)
    par = np.absolute(net_pos_acct_new)
    if len(pertub_vec) == 2:
        for i in n_account_range:
            for j in n_account_range:
                if i != j:
                    action = np.zeros(n_account)
                    action[i] = pertub_vec[0]
                    action[j] = pertub_vec[1]
                    par1 = par + action
                    if np.sum(par1[np.where(par1 < 0)]) == 0:
                        net_pos_acct_new = par1 * sgn_net
                        # print((i,j))
                        # print((net_pos_acct, net_pos_acct_new))
                        net_pos_acct_new = sanity_check_on_net_pos_acct(net_pos_acct, net_pos_acct_new, scale, qt)
                        # print('done')
                        pnl_acct_so_far2 = pnl_acct_so_far + net_pos_acct_new * (price_j - price_j_1) * multiplier
                        mae = criterion(pnl_acct_so_far2, cum_pnl, af, '1')

                        if mae == minMae:
                            e1 = entropy(net_pos_acct_new / af)
                            e2 = entropy(final_pos_acct_new / af)
                            if e1 < e2:
                                flag = 1

                        if mae < minMae or flag:
                            minMae = mae
                            qtAcct = net_pos_acct_new - net_pos_acct
                            optimal_allocation = net_pos_acct_new - net_pos_acct
                            final_pos_acct_new = net_pos_acct_new
                            final_pnl_acct_so_far_new = pnl_acct_so_far2
                            flag = 0

    elif len(pertub_vec) == 3:
        for i in n_account_range:
            for j in n_account_range:
                for z in n_account_range:
                    if i != j and i != z and j != z:
                        action = np.zeros(n_account)
                        action[i] = pertub_vec[0]
                        action[j] = pertub_vec[1]
                        action[z] = pertub_vec[2]
                        par1 = par + action
                        if np.sum(par1[np.where(par1 < 0)]) == 0:
                            net_pos_acct_new = par1 * sgn_net
                            net_pos_acct_new = sanity_check_on_net_pos_acct(net_pos_acct, net_pos_acct_new, scale, qt)
                            pnl_acct_so_far2 = pnl_acct_so_far + net_pos_acct_new * (price_j - price_j_1) * multiplier
                            mae = criterion(pnl_acct_so_far2, cum_pnl, af, '1')
                            if mae == minMae:
                                e1 = entropy(net_pos_acct_new / af)
                                e2 = entropy(final_pos_acct_new / af)
                                if e1 < e2:
                                    flag = 1

                            if mae < minMae or flag:
                                minMae = mae
                                qtAcct = net_pos_acct_new - net_pos_acct
                                optimal_allocation = net_pos_acct_new - net_pos_acct
                                final_pos_acct_new = net_pos_acct_new
                                final_pnl_acct_so_far_new = pnl_acct_so_far2
                                flag = 0
    elif len(pertub_vec) == 4:
        for i in n_account_range:
            for j in n_account_range:
                for z in n_account_range:
                    for w in n_account_range:
                        if i != j and i != z and j != z and i != w and j != w and z != w:
                            action = np.zeros(n_account)
                            action[i] = pertub_vec[0]
                            action[j] = pertub_vec[1]
                            action[z] = pertub_vec[2]
                            action[w] = pertub_vec[3]
                            par1 = par + action
                            if np.sum(par1[np.where(par1 < 0)]) == 0:
                                net_pos_acct_new = par1 * sgn_net
                                net_pos_acct_new = sanity_check_on_net_pos_acct(net_pos_acct, net_pos_acct_new, scale, qt)
                                pnl_acct_so_far2 = pnl_acct_so_far + net_pos_acct_new * (
                                            price_j - price_j_1) * multiplier
                                mae = criterion(pnl_acct_so_far2, cum_pnl, af, '1')
                                if mae == minMae:
                                    e1 = entropy(net_pos_acct_new / af)
                                    e2 = entropy(final_pos_acct_new / af)
                                    if e1 < e2:
                                        flag = 1

                                if mae < minMae or flag:
                                    minMae = mae
                                    qtAcct = net_pos_acct_new - net_pos_acct
                                    optimal_allocation = np.absolute(net_pos_acct_new - net_pos_acct)
                                    final_pos_acct_new = net_pos_acct_new
                                    final_pnl_acct_so_far_new = pnl_acct_so_far2
                                    flag = 0
    return optimal_allocation, final_pos_acct_new, final_pnl_acct_so_far_new, minMae


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
    af = np.array(af)
    para = abs(net_pos_j_1)* af
    initial_guess = np.rint(para)
    net_pos_acct = np.array(net_pos_acct)
    pnl_acct_so_far = np.array(pnl_acct_so_far)
    while (np.sum(initial_guess) != abs(net_pos_j_1)):
        #I check something compare to the original matlab code to make sure that the sign of the position would be the same as the sign of net position
        if np.sum(initial_guess) > abs(net_pos_j_1):
            index = np.argmax(initial_guess)
            # use value before round to find the max
            initial_guess[index] = initial_guess[index] - 1
        else:
            index = np.argmin(initial_guess)
            initial_guess[index] = initial_guess[index] + 1
    # print(np.sum(initial_guess))
    net_pos_acct_new = sign_net_pos  * initial_guess
    net_pos_acct_new = sanity_check_on_net_pos_acct(net_pos_acct, net_pos_acct_new, buy_or_sell, qty)
    # print(np.sum(initial_guess))
    parameters_set = {}
    parameters_set['Optimal Allocation'] = np.absolute(net_pos_acct_new - net_pos_acct)
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
    pnl_acct_so_far_new = pnl_acct_so_far + (price_j - price_j_1) * multiplier * net_pos_acct_new
    parameters_set['criteria_option'] = criteria_option
    parameters_set['pnl_acct_so_far_new'] = pnl_acct_so_far_new
    parameters_set['calculate'] = 0
    parameters_set['mse'] = criterion(pnl_acct_so_far_new, cum_pnl_j, af, '1')

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
    parameters_set['perturb_vec'] = [1, -1 ]
    parameters_set = optimizing_pnl_by_perturbing_net_pos(parameters_set);

    return parameters_set


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
    pertub_vec = np.array(parameters_set['perturb_vec'])
    optimal_allocation = parameters_set['Optimal Allocation']
    final_pos_acct_new = net_pos_acct_new
    final_pnl_acct_so_far_new = parameters_set['pnl_acct_so_far_new']
    minMae = parameters_set['mse']
    #numba do not support dictionary, so we need to put argument one by one
    optimal_allocation, final_pos_acct_new, final_pnl_acct_so_far_new, minMae = generate_grid(net_pos_acct_new, net_pos_acct, pnl_acct_so_far, sgn_net, scale,  af,
                                                                                                qty, pertub_vec, multiplier, price_j, price_j_1, cum_pnl, optimal_allocation,
                                                                                                final_pos_acct_new, final_pnl_acct_so_far_new, minMae)
    parameters_set['net_pos_acct_new'] = final_pos_acct_new
    parameters_set['pnl_acct_so_far_new'] = final_pnl_acct_so_far_new
    parameters_set['Optimal Allocation'] = optimal_allocation
    parameters_set['mse'] = minMae
    return parameters_set

@jit(nopython=True, cache = True)
def get_par_util(af, qty):
    """Utility function to calculate parameters in EOD Optimization
    af: numpy array
        allocation factor, representing the allocation weight to each account
    qty: int
        quantity to be allocation
    """
    par0 = af * qty
    par = np.rint(par0)
    while (sum(par) != qty):
        if sum(par) > qty:
            idx = np.argmin(par0)
            if par[idx] > 0:
                par[idx] = par[idx] - 1
        else:
            idx = np.argmax(par0)
            par[idx] = par[idx] + 1
    return par

@jit(nopython=True, cache = True)
def criterion_net_pos(net_pos_acct, af):
    """Objective function for optimization based on net position, use in EOD optimization
        Used only at the end of data, but could be vectorized to speed up the code too
    Parameters:
    net_pos_acct: numpy array
        a list representing the net position of each account
    Return:
        mae: float
    """
    n_acct = af.size
    mae = 0
    for i in np.arange(n_acct):
        for j in np.arange(n_acct):
            if i != j:
                mae = mae + (net_pos_acct[i] / af[i] - net_pos_acct[j] / af[j]) ** 2
    return mae

def last_trade_allocation(qty, scale, af, net_pos_acct, scenario):
    """EOD allocation check to make sure that the position satisfy the requirement of different scenario
    Parameters:

    """
    n_acct = af.size
    par = np.rint(af * qty)
    allocation = np.rint(af * qty)
    result = np.zeros(n_acct)
    if scenario == '1':
        result =  scale * np.absolute(net_pos_acct)
        result = net_pos_acct + result
        # return result
    elif scenario == '2':
        par0 = af * qty
        par = np.rint(par0)
        while (np.sum(par) != qty):
            if np.sum(par) > qty:
                idx = np.argmin(par0)
                if par[idx] > 0:
                    par[idx] = par[idx] - 1
            else:
                idx = np.argmax(par0)
                par[idx] = par[idx] + 1
        result = net_pos_acct + scale * np.absolute(par)
        # return result
    elif scenario == '3':
        net_pos = np.sum(net_pos_acct) + scale * qty
        sgn_net = 1 if net_pos >= 0 else -1
        par0 = af * np.absolute(net_pos)
        par = np.rint(par0)
        while (np.sum(par) != np.absolute(net_pos)):
            if np.sum(par) > np.absolute(net_pos):
                idx = np.argmin(par0)
                if par[idx] > 0:
                    par[idx] = par[idx] - 1
            else:
                idx = np.argmax(par0)
                par[idx] = par[idx] + 1

        net_pos_acct_new = sgn_net * par
        qt_acct = net_pos_acct_new - net_pos_acct
        result = sanity_check_on_net_pos_acct(net_pos_acct, net_pos_acct_new, scale, qty)
        # qt_acct = net_pos_acct_new - net_pos_acct
        # allocation = np.absolute(qt_acct)
        # return net_pos_acct_new
    elif scenario == '4':
        par0 = af * qty
        par = np.rint(par0)
        while (np.sum(par) != qty):
            if np.sum(par) > qty:
                idx = np.argmin(par0)
                if par[idx] > 0:
                    par[idx] = par[idx] - 1
            else:
                idx = np.argmax(par0)
                par[idx] = par[idx] + 1

        net_pos_acct_new = net_pos_acct + scale * par
        mae2 = criterion_net_pos(net_pos_acct_new, af)
        min_mae = mae2;
        allocation = par;
        # just do this once, so we do not optimize using numpy here, but could do that
        for i in np.arange(n_acct):
            for j in np.arange(n_acct):
                if i != j:
                    action = np.zeros(n_acct)
                    action[i] = 1;
                    action[j] = -1;
                    par1 =  par + action
                    if np.sum(par1[np.where(par1<0)]) > 0:
                        par1 = par
                    net_pos_acct_new = net_pos_acct + scale * par1
                    mae2 = criterion_net_pos(net_pos_acct_new, af)

                    if mae2 < min_mae:
                        min_mae = mae2
                        allocation = par1
        result = net_pos_acct + scale * allocation
    return result