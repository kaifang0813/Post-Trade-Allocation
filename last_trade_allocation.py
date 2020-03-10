import numpy as np
from allocation_optimization import sanity_check_on_net_pos_acct
from operator import add

def get_par_util(af, qty):
    """Utility function to calculate parameters in EOD Optimization
    af: numpy array
        allocation factor, representing the allocation weight to each account
    qty: int
        quantity to be allocation
    """
    par0 = af * qty
    par = par0.astype(int)

    while (sum(par) != qty):
        if sum(par) > qty:
            idx = np.argmin(par0)
            if par[idx] > 0:
                par[idx] = par[idx] - 1
        else:
            idx = np.argmax(par0)
            par[idx] = par[idx] + 1
    return par


def criterion_net_pos(net_pos_acct, af):
    """Objective function for optimization based on net position, use in EOD optimization
        Used only at the end of data, but could be vectorized to speed up the code too
    Parameters:
    net_pos_acct: list
        a list representing the net position of each account
    Return:
        mae: float
    """
    n_acct = len(net_pos_acct)
    mae = sum([(net_pos_acct[i] / af[i] - net_pos_acct[j] / af[j]) ** 2 for i in range(n_acct) for j in range(n_acct)])
    return mae


def last_trade_allocation(qty, scale, af, net_pos_acct, scenario):
    """EOD allocation check to make sure that the position satisfy the requirement of different scenario
    Parameters:

    """
    af = np.array(af)
    net_pos_acct = np.array(net_pos_acct)
    n_acct = len(af)
    if scenario == '1':
        allocation = np.absolute(net_pos_acct)
        net_pos_acct = net_pos_acct + scale * allocation

        if sum(list(map(abs, net_pos_acct))) != 0:
            raise ValueError('Something is wrong')
        return net_pos_acct
    elif scenario == '2':
        par = get_par_util(af, qty)
        allocation = np.absolute(par)
        net_pos_acct_new = net_pos_acct + scale * allocation
        return net_pos_acct_new
    elif scenario == '3':
        net_pos = sum(net_pos_acct) + scale * qty
        sgn_net = 1 if net_pos >= 0 else -1
        par = get_par_util(af, abs(net_pos))
        net_pos_acct_new = sgn_net * par
        qt_acct = net_pos_acct_new - net_pos_acct
        net_pos_acct_new = sanity_check_on_net_pos_acct(net_pos_acct, net_pos_acct_new, scale, qty)
        net_pos_acct_new = np.array(net_pos_acct_new)
        qt_acct = net_pos_acct_new - net_pos_acct
        allocation = np.absolute(qt_acct)
        return net_pos_acct_new
    elif scenario == '4':
        par = get_par_util(af, qty)
        net_pos_acct_new = net_pos_acct + scale * par
        mae2 = criterion_net_pos(net_pos_acct_new, af)
        min_mae = mae2;
        allocation = par;
        # just do this once, so we do not optimize using numpy here, but could do that
        for i in range(n_acct):
            for j in range(n_acct):
                if i != j:
                    action = [0] * n_acct
                    action[i] = +1;
                    action[j] = -1;
                    par1 = list(map(add, par, action))
                    if sum([i if i < 0 else 0 for i in par1]) > 0:
                        continue
                    net_pos_acct_new = net_pos_acct + scale * par1
                    mae2 = criterion_net_pos(net_pos_acct_new, af)

                    if mae2 < min_mae:
                        min_mae = mae2
                        allocation = par1
        return net_pos_acct_new