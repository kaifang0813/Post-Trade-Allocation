"""
Post Trade Allocation Excel Addin

"""
from xlwings import xw
from post_trade_allocation import AssetAllocation

@xw_func
@xw.ret(index=False, header=True, expand='table')
def post_trade_allocation_file_runner(file_name, expand_or_contract, criteria_option, lot_size, company, account_numer, broker, allocation_percentage  ):
    """run post trade allocation with excel file name
    Parameters:
        file_name: str
            the file that contain the order information
        expand_or_contract: str, 'expand', 'contract' or anything else
            to expand the order to single unit size, or contract it back, if not 'expand' and 'contract', would do nothing
        criteria_option: str
            '1' - '4' different kind of optimization objective function
        lot_size: int
            lot size of each unit for the contract
        company: str[]
            a list of company name for the SMAs
        account_numer: str[]
            a list of acct number for the SMAs
        broker: str[]
            a list of broker name for each SMAs
        allocation_percentage: float[]
            a list of number shows the allocation percentage for each SMAs for the trade book

    """

    managed_account = {i:(company, acct_num, bro, af) for i, company, acct_num, bro, af in zip(range(1,len(allocation_percentage)+1), company, account_numer, broker, allocation_percentage)}

    post_allocation = AssetAllocation(managed_account=managed_account,
                                       which_action=expand_or_contract,
                                       criteria_option=criteria_option,
                                       lot_size=lot_size)

    post_allocation.fill_orders_file(file_name)
    post_allocation.allocate_trade()
    return post_allocation.allocation_detail

@xw_func
@xw.arg('order_data', pd.DataFrame, index=False, header=False)
@xw.ret(index=False, header=True, expand='table')
def post_trade_allocation_data_frame_runner(order_data, expand_or_contract, criteria_option, lot_size, company, account_numer, broker, allocation_percentage  ):
    """run post trade allocation with excel file name
    Parameters:
        order_data: pandas dataframe
            the dataframe that contain the order information
        expand_or_contract: str, 'expand', 'contract' or anything else
            to expand the order to single unit size, or contract it back, if not 'expand' and 'contract', would do nothing
        criteria_option: str
            '1' - '4' different kind of optimization objective function
        lot_size: int
            lot size of each unit for the contract
        company: str[]
            a list of company name for the SMAs
        account_numer: str[]
            a list of acct number for the SMAs
        broker: str[]
            a list of broker name for each SMAs
        allocation_percentage: float[]
            a list of number shows the allocation percentage for each SMAs for the trade book

    """

    managed_account = {i:(company, acct_num, bro, af) for i, company, acct_num, bro, af in zip(range(1,len(allocation_percentage)+1), company, account_numer, broker, allocation_percentage)}

    post_allocation = AssetAllocation(managed_account=managed_account,
                                       which_action=expand_or_contract,
                                       criteria_option=criteria_option,
                                       lot_size=lot_size,
                                       order_data = order_data,
                                       use_file=0)

    post_allocation.fill_orders_file(None)
    post_allocation.allocate_trade()
    return post_allocation.allocation_detail
