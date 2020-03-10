import pandas
import numpy as np
import datetime
from last_trade_allocation import last_trade_allocation
import copy
from operator import add
import math
import multiprocessing as mp
import time


class AssetAllocation:
    """Asset Allocation Class used to do asset allocation

        The trade allocation contains several process:
        1. Read data from a trade file or dataframe with certain format, and do data cleaning and pre-processing to get certain format that will used in calculation, the
            function could be customized in order_data_preprocessing
            Also, we could choose to expand the trade into single unit trade using the 'expand' functionality
        2. Allocate trade: the trade allocation is a per-trade allocation plus a End of Day sanity check, using Grid search with Perturbing Optimization
            Detial methodology could be explained as the following:
            1) seperate the trade order into a daily basis


    """

    def __init__(self, managed_account, lot_size, which_action, criteria_option,
                 roll_date=[], sanity_on_net_position='', which_order='', previous_pnl_acct=None,
                 previous_net_pos_acct=None,
                 eod_price=None, use_file=1, order_data=None):
        """
        Parameters:
        managedAccount: dict
            dictionary with same format as MANAGED_ACCOUNTS which provide you detail information of management account
                        symbol: str
                            symbol that we would allocate asset to

                        # FIX ME : not sure whether SYMBOL2 FOLLOWING is necessary or not, if we have the whole allocation trade book, could do trade
                        # allocation on each symbol, unless we want to treat symbol 1 and symbol 2 the same asset in allocation

                        symbol2: str
                            if symbol is something like futures, when future of symbol1 expire, would roll into symbol2
        roll_date: list of datetime.date
            rolling date of future contract, must equal or before the last tradig date of the contract of symbol1
        which_action: str   ----- should only be expand
            'expand' or 'contract', preprocessing on trade data to downside the trade quantity or upside the trade quantity
        criteria_option: str
            objective function choice in the optimization

        lot_size: int
            lot size of one unit of the contract
        sanity_on_net_position: int
            ?
        whichOrder: int
            ?

        previous_net_pos_acct: list
            list with same length as allocation factor, if we already have some position at the start of the day, we could use previous_net_pos_acct
            to provide information about pnl of different SMAs, and use it as the start point of the optimization
        previous_pnl_acct:list
            list with same length as allocation factor, if we already have some position at the start of the day, we could use previous_pnl_acct
            to provide information about pnl of different SMAs, and use it as the start point of the optimization
        eod_price: float
            if pnl_acct_so_far is not None, we would need the eod_price of the previous date to calculate the over_night pnl change at the first period
        use_file: int
            0 or 1, if 1, use the file to load data, if 0, use the order data
        order_data: pandas.DataFrame
            provide data that you could get from the order file as a dataframe
        """
        self._managed_account = managed_account
        self._lot_size = lot_size
        self._roll_date = roll_date
        self._which_action = which_action
        self._criteria_option = criteria_option
        self._sanity_on_net_position = sanity_on_net_position
        self._which_order = which_order
        self.order_data = None  # used to save order data
        self.allocation_factor = [value[3] for value in managed_account.values()]
        self.allocation_account = pandas.DataFrame(columns=list(managed_account.keys()))
        self.allocation_account_pnl = pandas.DataFrame(columns=list(managed_account.keys()))
        self._multiplier = None
        self._previous_pnl_acct = previous_pnl_acct
        self._previous_net_pos_acct = previous_net_pos_acct
        self._eod_price = eod_price
        self.use_file = use_file
        self.order_data = order_data


        # generate the cached grid that could be used in grid search later
        self.pertub_grid_1 = self.generate_grid(self.allocation_factor, [1,-1])
        self.pertub_grid_2 = self.generate_grid(self.allocation_factor, [2, -2])
        self.pertub_grid_3 = self.generate_grid(self.allocation_factor, [3, -3])
        self.pertub_grid_4 = self.generate_grid(self.allocation_factor, [1, 2, -3])
        self.pertub_grid_5 = self.generate_grid(self.allocation_factor, [1, 2, -3])
        # self.pertub_grid_6 = self.generate_grid(self.allocation_factor,[1, -1, 1, -1])

        # parameters_set['pertub_vec'] = [1, -1, 1, -1]
        # parameters_set = optimizing_pnl_by_perturbing_net_pos(parameters_set);


    def order_data_preprocessing(self, df):
        """Normalize the format of the portfolio to be used later
            Parameters:
                df: dataframe
                contain trade information
            Output:
                df: dataframe
                contain normalized data"""
        df.rename(columns={
            'CALENDAR DATE': 'Date',
            'CALENDAR TIME': 'Time',
            'SYMBO': 'Symbol',
            'LAST PRICE': 'Price',
            'LAST SHARES': 'Quantity',
            'SIDE': 'Side',
            'trade_id': 'ID',
        }, inplace=True)

        df.dropna(inplace = True)
        # transform integer/ date string  to date
        if self.use_file:
            try:
                df['Date'] = df['Date'].apply(lambda x: datetime.datetime(1900, 1, 1)+ datetime.timedelta(days=x-2) if not isinstance(x, str) else datetime.datetime.strptime(x, '%m/%d/%Y'))
            except:
                try:
                    df['Date'] = df['Date'].apply(lambda x: datetime.datetime(1900, 1, 1)+ datetime.timedelta(days=x-2) if not isinstance(x, str) else datetime.datetime.strptime(x, '%m/%d/%y'))
                except:
                    raise ValueError('Could not process the Date Input')
        df['Side'] = df['Side'].apply(lambda x: 'Buy' if x == 0 else 'Sell')

        return df

    def contract_expand_orders(self, data, action='expand'):
        data.reset_index(inplace=True, drop=True)
        if action == 'contract':
            # ALGO 3 groupby, merge the expand trade to original state:
            cols = data.columns.tolist()
            cols.remove('Quantity')
            result = data.groupby(cols).sum().reset_index()
            return result
        elif action == 'expand':
            # the expand algorithm is fast as it is based on built in function of pandas
            result = pandas.DataFrame(columns=data.columns)
            # expand trade that has more than one lot to multiple trade that has more than one lots
            for i in data['Quantity'].unique():
                copy_row = [data[data['Quantity'] == i]] * i
                temp = pandas.concat(copy_row, ignore_index=True)
                result = result.append(temp)
            result['Quantity'] = 1
            result.sort_values(by=['ID'], inplace=True)
            result.reset_index(inplace=True, drop=True)
            return result
        else:
            return data

    # define fill order function
    def fill_orders_file(self, file_name):
        """fill orders using csv file, could customized to other method
           Parameters:
               file_name: str
                   csv file which contain the trade information, the test file we use have the following field
                    ['CALENDAR DATE', 'CALENDAR TIME', 'trade_id', 'SYMBO', 'LAST PRICE', 'LAST SHARES', 'SIDE']
        """
       
        if self.use_file == 1:
            data = pandas.read_csv(file_name)
        else:
            data = self.order_data

        data = self.order_data_preprocessing(data)
        data = self.contract_expand_orders(data, self._which_action)
        lot_multiplier = self._lot_size
        # get pnl of at portfolio level
        data['Price_diff'] = data['Price'].diff()
        data['Quant'] = data.apply(lambda row: row['Quantity']
        if row['Side'] == 'Buy' else
        -row['Quantity'], axis=1)
        data['NetPosition'] = data['Quant'].cumsum()
        data['NetPosition_j_1'] = data['NetPosition'].shift(1)
        data['PNL'] = data['Price_diff'] * data['NetPosition_j_1'] * lot_multiplier
        data['cum_pnl'] = data['PNL'].cumsum()
        data['cum_pnl_j_1'] = data['cum_pnl'].shift(1)
        data['buy_or_sell'] = data['Side'].apply(lambda x: 1 if x == 'Buy' else -1)
        data.fillna(value=0, inplace=True)
        self._multiplier = lot_multiplier

        self.order_data = data

    def allocate_trade(self):
        data = self.order_data
        len_data = len(data)
        af = self.allocation_factor
        len_af = len(af)
        criteria_option = self._criteria_option
        sanity_on_net_position = self._sanity_on_net_position
        net_pos_acct_all = np.zeros((len_data, len_af)).astype(int)
        pnl_acct_all = np.zeros((len_data, len_af))
        pnl_acct_so_far = list(np.zeros(len_af))
        criteria_option = self._criteria_option
        net_pos_acct_so_far = np.zeros(len_af).astype(int)
        multiplier = self._multiplier
        # loop parameters: j is the index
        last_price = 0
        if self._eod_price is not None:
            last_price = self._eod_price

        if self._previous_net_pos_acct is not None:
            net_pos_acct_so_far = np.array(self._previous_net_pos_acct)
            net_pos = sum(net_pos_acct_so_far)
            data['NetPosition'] = data['NetPosition'] + net_pos
            data['NetPosition_j_1'] = data['NetPosition_j_1'] + net_pos
            price_diff_idx = data.columns.tolist().index('Price_diff')
            price_idx = data.columns.tolist().index('Price')
            data.iloc[0, price_diff_idx] = data.iloc[0, price_idx] - self._eod_price
            data['PNL'] = data['Price_diff'] * data['NetPosition_j_1'] * multiplier

        if self._previous_pnl_acct is not None:
            pnl_acct_so_far = self._previous_pnl_acct
            prev_pnl = sum(pnl_acct_so_far)
            data['cum_pnl'] = data['PNL'].cumsum()
            data['cum_pnl_j_1'] = data['cum_pnl'].shift(1)
            data['cum_pnl_j_1'] = data['cum_pnl'].shift(1)
            data['cum_pnl'] = data['cum_pnl'] + prev_pnl
            data['cum_pnl_j_1'] = data['cum_pnl_j_1'] + prev_pnl

        #         last_period_pnl = 0
        qty = 0
        buy_or_sell = 0
        for date in data.Date.unique():
            if pandas.to_datetime(date).strftime('%Y-%m-%d') == "2009-06-08":
                print(date)
            temp_data = data[data.Date == date]
            count = 0
            for j, row in temp_data.iterrows():
                print(j)
                tick = time.clock()
                # for j, we are actually doing allocation for j-1
                if count > 0:
                    cum_pnl_j = row['cum_pnl']
                    net_pos_j_1 = int(row['NetPosition_j_1'])
                    net_pos_acct = list(net_pos_acct_so_far)
                    price_j = row['Price']
                    parameters_set = self.grid_search_optimization(cum_pnl_j, net_pos_j_1, net_pos_acct, af, qty,
                                                              pnl_acct_so_far, last_price, price_j, multiplier,
                                                              buy_or_sell, criteria_option, sanity_on_net_position)

                    net_pos_acct_so_far = net_pos_acct_so_far + np.array(parameters_set['Optimal Allocation']).astype(
                        int)
                    net_pos_acct_all[j - 1] = net_pos_acct_so_far
                    pnl_acct_so_far = parameters_set['pnl_acct_so_far_new']
                    pnl_acct_all[j] = pnl_acct_so_far
                    # print(net_pos_acct_so_far)

                    # continue the loop by updating the price and last_period_pnl_acct
                if count == 0 and last_price is not None:
                    # calculate the first pnl for the position based on eod price and the price of first trade in the day:

                    pnl_acct_all[j] = [pnl + multiplier * (row['Price'] - last_price) * pos for pnl, pos in
                                       zip(pnl_acct_so_far, net_pos_acct_so_far)]
                    pnl_acct_so_far = pnl_acct_all[j]

                last_price = row['Price']
                #                 last_period_pnl = row['PNL']
                qty = row['Quantity']
                buy_or_sell = row['buy_or_sell']

                # end of day check
                scenario = None
                if count == len(temp_data) - 1:
                    if row['NetPosition'] == 0:
                        scenario = '1'
                    elif row['NetPosition_j_1'] == 0:
                        scenario = '2'
                    else:
                        scenario = '3'
                    net_pos_acct_new = last_trade_allocation(qty, buy_or_sell, af, net_pos_acct_all[j - 1], scenario)
                    net_pos_acct_all[j] = net_pos_acct_new
                    net_pos_acct_so_far = net_pos_acct_all[j]


                count = count + 1
                tock = time.clock()
                print(tock-tick)

        acct_columns_name = ['Acct_Position_{}'.format(int(i)) for i in self._managed_account.keys()]
        pnl_columns_name = ['Acct_PNL_{}'.format(int(i)) for i in self._managed_account.keys()]

        net_postion_data_table = pandas.DataFrame.from_records(net_pos_acct_all)
        net_postion_data_table.columns = acct_columns_name
        pnl_acct_data_table = pandas.DataFrame.from_records(pnl_acct_all)
        pnl_acct_data_table.columns = pnl_columns_name
        final_result = data.join(net_postion_data_table).join(pnl_acct_data_table)
        final_result['Date'] = final_result['Date'].apply(lambda x: x.strftime('%m/%d/%y'))
        self.allocation_detail =  final_result
        
        
    def get_allocation_result(self, display_format= "allocation", pivot = True, aum=None):
        
        len_af = len(self.allocation_factor)
        acct_list = ['Acct_Position_{}'.format(i) for i in self._managed_account.keys()]
        pnl_list = ['Acct_PNL_{}'.format(i) for i in self._managed_account.keys()]
        allocation_list = ['Acct_Allocation_{}'.format(i) for i in self._managed_account.keys()]
        default_cols = ['Date', 'Time', 'ID', 'Symbol', 'Price', 'Quantity', 'Side']
        default_display = self.allocation_detail[ default_cols + acct_list]
        
        pnl_table = self.allocation_detail[ default_cols + pnl_list]
        unpivot_pnl_table = pandas.melt(pnl_table, id_vars=default_cols, var_name='Acct', value_name='PNL')
        unpivot_pnl_table['Acct'] = unpivot_pnl_table['Acct'].apply(lambda x : x.split('_')[-1])
        
        net_position_table = default_display
        unpivot_net_positionl_table = pandas.melt(net_position_table, id_vars=default_cols, var_name='Acct', value_name='PNL')
        unpivot_net_positionl_table['Acct'] = unpivot_net_positionl_table['Acct'].apply(lambda x : x.split('_')[-1])

        if  display_format.lower() == 'allocation':
            allocation_table =  default_display[acct_list].diff()
            allocation_table.columns = allocation_list
            allocation_table.loc[0, allocation_list] = default_display[acct_list].iloc[0].values
        
            allocation_table = pandas.merge(self.allocation_detail[default_cols],  allocation_table, left_index=True, right_index=True)
            
            if pivot:
                return  allocation_table
            else:
                unpivot_allocation = pandas.melt(allocation_table, id_vars=default_cols, var_name='Acct', value_name='Allocation')
                unpivot_allocation['Acct'] = unpivot_allocation['Acct'].apply(lambda x : x.split('_')[-1])
                return  unpivot_allocation
        elif display_format.lower() == 'pnl':
            pnl_table = self.allocation_detail[ default_cols + pnl_list]
            if pivot:
                return pnl_table
            else:
                unpivot_pnl_table = pandas.melt(pnl_table, id_vars=default_cols, var_name='Acct', value_name='PNL')
                unpivot_pnl_table['Acct'] = unpivot_pnl_table['Acct'].apply(lambda x : x.split('_')[-1])
                return  unpivot_pnl_table
        elif display_format.lower() == 'position':
            net_position_table = default_display
            if pivot:
                return net_position_table
            else:
                unpivot_net_positionl_table = pandas.melt(net_position_table, id_vars=default_cols, var_name='Acct', value_name='Position')
                unpivot_net_positionl_table['Acct'] = unpivot_net_positionl_table['Acct'].apply(lambda x : x.split('_')[-1])
                return unpivot_net_positionl_table
        elif display_format.lower() =='eod_pnl':
            pnl_table = self.allocation_detail[ ['Date', 'Time','Price'] + pnl_list].groupby(['Date']).last().reset_index()
            if pivot:
                return pnl_table
            else:
                unpivot_pnl_table = pandas.melt(pnl_table, id_vars=['Date', 'Time','Price'], var_name='Acct', value_name='PNL')
                unpivot_pnl_table['Acct'] = unpivot_pnl_table['Acct'].apply(lambda x : x.split('_')[-1])
                return  unpivot_pnl_table
        elif display_format.lower() =='eod_position':
            pnl_table = self.allocation_detail[ ['Date', 'Time','Price'] + acct_list].groupby(['Date']).last().reset_index()
            if pivot:
                return pnl_table
            else:
                unpivot_pnl_table = pandas.melt(pnl_table, id_vars=['Date', 'Time','Price'], var_name='Acct', value_name='PNL')
                unpivot_pnl_table['Acct'] = unpivot_pnl_table['Acct'].apply(lambda x : x.split('_')[-1])
                return  unpivot_pnl_table
        elif display_format.lower() =='return':
            pnl_table = self.allocation_detail[ ['Date', 'Time','Price'] + pnl_list].groupby(['Date']).last().reset_index()
            if aum is None:
                raise ValueError('AUM should not be none when calculating return')
            returnn_list = ['Return_{}(%)'.format(i) for i in self._managed_account.keys()]
            for name, af in zip(pnl_list, self.allocation_factor):
                pnl_table[name] = pnl_table[name].apply(lambda x: x/(aum*af)*100)

            pnl_table.rename(index=str, columns={pnl:rt for pnl ,rt in zip(pnl_list, returnn_list)}, inplace = True)
            if pivot:
                return pnl_table
            else:
                unpivot_pnl_table = pandas.melt(pnl_table, id_vars=['Date', 'Time','Price'], var_name='Acct', value_name='Return(%)')
                unpivot_pnl_table['Acct'] = unpivot_pnl_table['Acct'].apply(lambda x : x.split('_')[-1])
                return  unpivot_pnl_table
        else:
            return default_display


    # The following function is for optimization, in order to improve the performance and reduce the proecssing time in grid search using vectorization, we
    # would put the perverb data inside the obj



    def sanity_check_on_net_pos_acct(self, net_pos_acct, net_pos_acct_new, scale, qt):
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

    def criterion(self, pnl_acct_sof_far, cum_pnl, af, option):
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

    def grid_search_optimization(self, cum_pnl_j, net_pos_j_1, net_pos_acct, af, qty,
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
            adjust = [abs(x - y) for x, y in zip(para, initial_guess)]
            index = adjust.index(max(adjust))
            if sum(initial_guess) > abs(net_pos_j_1):
                # use value before round to find the max
                initial_guess[index] = initial_guess[index] - 1
            else:
                initial_guess[index] = initial_guess[index] + 1

        net_pos_acct_new = [sign_net_pos * i for i in initial_guess]
        net_pos_acct_new = self.sanity_check_on_net_pos_acct(net_pos_acct, net_pos_acct_new, buy_or_sell, qty)

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
        parameters_set = self.optimizing_pnl_by_perturbing_net_pos(parameters_set, self.pertub_grid_1);
        # parameters_set = self.optimizing_pnl_by_perturbing_net_pos(parameters_set, self.pertub_grid_2);
        # parameters_set = self.optimizing_pnl_by_perturbing_net_pos(parameters_set, self.pertub_grid_3);
        # parameters_set = self.optimizing_pnl_by_perturbing_net_pos(parameters_set, self.pertub_grid_4);
        # parameters_set = self.optimizing_pnl_by_perturbing_net_pos(parameters_set, self.pertub_grid_5);

        # parameters_set = self.optimizing_pnl_by_perturbing_net_pos(parameters_set, self.pertub_grid_6);
        return parameters_set

    def sanity_check_on_net_pos_acct_v(self, net_pos_acct, net_pos_acct_new, scale, qt):
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
                positive_update = np.transpose(
                    np.where(np.transpose(temp_array) == np.argmax(max_diff_pos, axis=1), 1, 0))
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

                negative_update = np.transpose(
                    np.where(np.transpose(temp_array) == np.argmin(min_diff_pos, axis=1), 1, 0))
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

    def criterion_v(self, pnl_acct_sof_far, cum_pnl, af, option):
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

    def generate_grid(self, net_pos_acct_new, pertub_vec):
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
        grid = np.zeros(shape=(pow(n_account, len(pertub_vec)), n_account))

        if len(pertub_vec) == 2:
            for i in n_account_range:
                for j in n_account_range:
                    if len(set([i,j])) ==2:
                        action = np.zeros(n_account).astype(int)
                        action[i] = pertub_vec[0]
                        action[j] = pertub_vec[1]
                        grid[i * pow(n_account, 1) + j, :] = action
        elif len(pertub_vec) == 3:
            for i in n_account_range:
                for j in n_account_range:
                    for z in n_account_range:
                        if len(set([i, j, z])) == 3:
                            action = np.zeros(n_account).astype(int)
                            action[i] = pertub_vec[0]
                            action[j] = pertub_vec[1]
                            action[z] = pertub_vec[2]
                            grid[i * pow(n_account, 2) + j * pow(n_account, 1) + z, :] = action
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
                                grid[i * pow(n_account, 3) + j * pow(n_account, 2) + z * pow(n_account, 1) + w, :] = action

        else:
            raise ValueError('only support pertub_vec of length 2,3,4 for now')

        if len(set(sum(np.transpose(grid)))) > 1:
            raise ValueError(
                "the grid generated has different net position side in different row, something must be wrong")

        return grid

    def entropy_v(self, data):
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

    def optimizing_pnl_by_perturbing_net_pos(self, parameters_set, grid_search_change):
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
        n_account = len(net_pos_acct_new)
        n_row, n_col = grid_search_change.shape
        net_pos_acct_new = np.unique(np.broadcast_to(net_pos_acct_new, (n_row, n_col)) + grid_search_change, axis=0)
        # vectorization computing:
        check_alloc = self.sanity_check_on_net_pos_acct_v(net_pos_acct, net_pos_acct_new, scale, qty)

        check_alloc = np.unique(check_alloc, axis=0)
        # np.unique(check_alloc, axis =0) not working in some numpy version
        x, y = check_alloc.shape
        net_pos_acct = np.array([net_pos_acct] * x)
        pnl_acct_so_far_new = np.array([pnl_acct_so_far] * x) + check_alloc * (price_j - price_j_1) * multiplier
        mae = self.criterion_v(pnl_acct_so_far_new, cum_pnl, af, criteria);
        min_index = np.where(mae == mae.min())[0]
        # if multiple allocation has the same kind of result, have a one more step allocation to choose the lowest entropy allocation
        if len(min_index) > 1:
            fur_optimize = check_alloc[min_index, :]
            pnl = pnl_acct_so_far_new[min_index, :]
            pos_entropy = self.entropy_v(fur_optimize / af)
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
            
            
