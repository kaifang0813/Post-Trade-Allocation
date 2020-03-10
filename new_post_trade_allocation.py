import pandas
import numpy as np
import datetime
from last_trade_allocation import last_trade_allocation
import copy
from operator import add
import math
import multiprocessing as mp
import time
from jit_grid_search import generate_grid, sanity_check_on_net_pos_acct, criterion, entropy, grid_search_optimization

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

        df.dropna(inplace=True)
        # transform integer/ date string  to date
        if self.use_file:
            try:
                df['Date'] = df['Date'].apply(
                    lambda x: datetime.datetime(1900, 1, 1) + datetime.timedelta(days=x - 2) if not isinstance(x,
                                                                                                               str) else datetime.datetime.strptime(
                        x, '%m/%d/%Y'))
            except:
                try:
                    df['Date'] = df['Date'].apply(
                        lambda x: datetime.datetime(1900, 1, 1) + datetime.timedelta(days=x - 2) if not isinstance(x,
                                                                                                                   str) else datetime.datetime.strptime(
                            x, '%m/%d/%y'))
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
            net_pos = np.sum(net_pos_acct_so_far)
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
            temp_data = data[data.Date == date]
            count = 0

            tik = time.time()
            for j, row in temp_data.iterrows():
                # for j, we are actually doing allocation for j-1
                if count > 0:
                    cum_pnl_j = row['cum_pnl']
                    net_pos_j_1 = int(row['NetPosition_j_1'])
                    net_pos_acct = net_pos_acct_so_far
                    price_j = row['Price']
                    parameters_set = grid_search_optimization(cum_pnl_j, net_pos_j_1, net_pos_acct, af, qty,
                                                                   pnl_acct_so_far, last_price, price_j, multiplier,
                                                                   buy_or_sell, criteria_option, sanity_on_net_position)
                    net_pos_acct_so_far = parameters_set['net_pos_acct_new']

                    net_pos_acct_all[j - 1] = net_pos_acct_so_far
                    pnl_acct_so_far = parameters_set['pnl_acct_so_far_new']
                    pnl_acct_all[j] = pnl_acct_so_far

                    # continue the loop by updating the price and last_period_pnl_acct
                if count == 0 and last_price is not None:
                    # calculate the first pnl for the position based on eod price and the price of first trade in the day:

                    pnl_acct_all[j] = pnl_acct_so_far +  multiplier * (row['Price'] - last_price) * net_pos_acct_so_far
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
            tok = time.time()
            print("{}:{}".format(pandas.to_datetime(str(date)).strftime("%Y-%m-%d"), tok - tik))
            tik = time.time()

        acct_columns_name = ['Acct_Position_{}'.format(int(i)) for i in self._managed_account.keys()]
        pnl_columns_name = ['Acct_PNL_{}'.format(int(i)) for i in self._managed_account.keys()]

        net_postion_data_table = pandas.DataFrame.from_records(net_pos_acct_all)
        net_postion_data_table.columns = acct_columns_name
        pnl_acct_data_table = pandas.DataFrame.from_records(pnl_acct_all)
        pnl_acct_data_table.columns = pnl_columns_name
        final_result = data.join(net_postion_data_table).join(pnl_acct_data_table)
        final_result['Date'] = final_result['Date'].apply(lambda x: x.strftime('%m/%d/%y'))
        self.allocation_detail = final_result

    def get_allocation_result(self, display_format="allocation", pivot=True, aum=None):

        len_af = len(self.allocation_factor)
        acct_list = ['Acct_Position_{}'.format(i) for i in self._managed_account.keys()]
        pnl_list = ['Acct_PNL_{}'.format(i) for i in self._managed_account.keys()]
        allocation_list = ['Acct_Allocation_{}'.format(i) for i in self._managed_account.keys()]
        default_cols = ['Date', 'Time', 'ID', 'Symbol', 'Price', 'Quantity', 'Side']
        default_display = self.allocation_detail[default_cols + acct_list]

        pnl_table = self.allocation_detail[default_cols + pnl_list]
        unpivot_pnl_table = pandas.melt(pnl_table, id_vars=default_cols, var_name='Acct', value_name='PNL')
        unpivot_pnl_table['Acct'] = unpivot_pnl_table['Acct'].apply(lambda x: x.split('_')[-1])

        net_position_table = default_display
        unpivot_net_positionl_table = pandas.melt(net_position_table, id_vars=default_cols, var_name='Acct',
                                                  value_name='PNL')
        unpivot_net_positionl_table['Acct'] = unpivot_net_positionl_table['Acct'].apply(lambda x: x.split('_')[-1])

        if display_format.lower() == 'allocation':
            allocation_table = default_display[acct_list].diff()
            allocation_table.columns = allocation_list
            allocation_table.loc[0, allocation_list] = default_display[acct_list].iloc[0].values

            allocation_table = pandas.merge(self.allocation_detail[default_cols], allocation_table, left_index=True,
                                            right_index=True)

            if pivot:
                return allocation_table
            else:
                unpivot_allocation = pandas.melt(allocation_table, id_vars=default_cols, var_name='Acct',
                                                 value_name='Allocation')
                unpivot_allocation['Acct'] = unpivot_allocation['Acct'].apply(lambda x: x.split('_')[-1])
                return unpivot_allocation
        elif display_format.lower() == 'pnl':
            pnl_table = self.allocation_detail[default_cols + pnl_list]
            if pivot:
                return pnl_table
            else:
                unpivot_pnl_table = pandas.melt(pnl_table, id_vars=default_cols, var_name='Acct', value_name='PNL')
                unpivot_pnl_table['Acct'] = unpivot_pnl_table['Acct'].apply(lambda x: x.split('_')[-1])
                return unpivot_pnl_table
        elif display_format.lower() == 'position':
            net_position_table = default_display
            if pivot:
                return net_position_table
            else:
                unpivot_net_positionl_table = pandas.melt(net_position_table, id_vars=default_cols, var_name='Acct',
                                                          value_name='Position')
                unpivot_net_positionl_table['Acct'] = unpivot_net_positionl_table['Acct'].apply(
                    lambda x: x.split('_')[-1])
                return unpivot_net_positionl_table
        elif display_format.lower() == 'eod_pnl':
            pnl_table = self.allocation_detail[['Date', 'Time', 'Price'] + pnl_list].groupby(
                ['Date']).last().reset_index()
            if pivot:
                return pnl_table
            else:
                unpivot_pnl_table = pandas.melt(pnl_table, id_vars=['Date', 'Time', 'Price'], var_name='Acct',
                                                value_name='PNL')
                unpivot_pnl_table['Acct'] = unpivot_pnl_table['Acct'].apply(lambda x: x.split('_')[-1])
                return unpivot_pnl_table
        elif display_format.lower() == 'eod_position':
            pnl_table = self.allocation_detail[['Date', 'Time', 'Price'] + acct_list].groupby(
                ['Date']).last().reset_index()
            if pivot:
                return pnl_table
            else:
                unpivot_pnl_table = pandas.melt(pnl_table, id_vars=['Date', 'Time', 'Price'], var_name='Acct',
                                                value_name='PNL')
                unpivot_pnl_table['Acct'] = unpivot_pnl_table['Acct'].apply(lambda x: x.split('_')[-1])
                return unpivot_pnl_table
        elif display_format.lower() == 'return':
            pnl_table = self.allocation_detail[['Date', 'Time', 'Price'] + pnl_list].groupby(
                ['Date']).last().reset_index()
            if aum is None:
                raise ValueError('AUM should not be none when calculating return')
            returnn_list = ['Return_{}(%)'.format(i) for i in self._managed_account.keys()]
            for name, af in zip(pnl_list, self.allocation_factor):
                pnl_table[name] = pnl_table[name].apply(lambda x: x / (aum * af) * 100)

            pnl_table.rename(index=str, columns={pnl: rt for pnl, rt in zip(pnl_list, returnn_list)}, inplace=True)
            if pivot:
                return pnl_table
            else:
                unpivot_pnl_table = pandas.melt(pnl_table, id_vars=['Date', 'Time', 'Price'], var_name='Acct',
                                                value_name='Return(%)')
                unpivot_pnl_table['Acct'] = unpivot_pnl_table['Acct'].apply(lambda x: x.split('_')[-1])
                return unpivot_pnl_table
        else:
            return default_display

    # The following function is for optimization, in order to improve the performance and reduce the proecssing time in grid search using vectorization, we
    # would put the perverb data inside the obj



