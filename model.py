import numpy as np
from new_post_trade_allocation import AssetAllocation
import pandas
import datetime

class Model():

    def __init__(self):
        self.managed_account_file = None
        self.managed_account = None
        self.expand_or_contract = 'expand'
        self.criteria_option = '1'
        self.lot_size = None
        self.order_data = None
        self.symbol = None
        self.pivot_or_not=False
        self.result = None
        self.previous_net_pos_acct = None
        self.previous_pnl_acct=None
        self.eod_price = None
        self.aum = None


    def calculate(self):


        self.post_allocation = AssetAllocation(managed_account=self.managed_account,
                                          which_action=self.expand_or_contract,
                                          criteria_option=self.criteria_option,
                                          lot_size=self.lot_size,
                                          order_data=self.order_data,
                                          previous_pnl_acct=self.previous_pnl_acct,
                                          previous_net_pos_acct=self.previous_net_pos_acct,
                                          eod_price=self.eod_price,
                                          use_file=0,
                                          )

        self.post_allocation.fill_orders_file(None)
        self.post_allocation.allocate_trade()
        self.result = self.post_allocation.get_allocation_result('allocation', self.pivot_or_not, self.aum)

    def display(self,type, pivot) :
        return self.post_allocation.get_allocation_result(type, pivot, self.aum)






