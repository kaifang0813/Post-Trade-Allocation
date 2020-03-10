"""
Post Trade Allocation Excel Addin

"""
from pyxll import xl_func
from post_trade_allocation import AssetAllocation
import pandas
import datetime

@xl_func("dataframe<index=False>: dataframe<index=False>", auto_resize=True)
def read_df(df):
    return df

@xl_func("str, str, str, int, str[], str[], str[], float[], float[], int[], float, str, str, float: dataframe<index=False>", auto_resize=True)
def post_trade_allocation_file_runner(file_name, expand_or_contract, 
    criteria_option, lot_size, company, account_numer, 
    broker, allocation_percentage ,
    previous_pnl_acct,
    previous_net_pos_acct,
    eod_price,
    display_type, pivot_or_not, aum ):
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
        display_type: str
            ways to display the result, could be pnl, allocation, position
        pivot_or_not: str YES OR NO
            whether to do pivot table view or not for the result

    """

    managed_account = {i:(company, acct_num, bro, af) for i, company, acct_num, bro, af in zip(range(1,len(allocation_percentage)+1), company, account_numer, broker, allocation_percentage, display_type, pivot_or_not )}

    post_allocation = AssetAllocation(managed_account=managed_account,
                                      which_action=expand_or_contract,
                                      criteria_option=criteria_option,
                                      lot_size=lot_size,
                                      previous_pnl_acct=previous_pnl_acct,
                                      previous_net_pos_acct=previous_net_pos_acct,
                                      eod_price=eod_price,
                                      use_file=1,
                                      )


    post_allocation.fill_orders_file(file_name)
    post_allocation.allocate_trade()
    
    if pivot_or_not.lower() in ['y', 'yes','true']:
        pivot_or_not =True
    elif pivot_or_not.lower() in ['n','no','false']:
        pivot_or_not = False
    else:
        raise ValueError('pivot_or_not should be either YES or NO')
        
    return post_allocation.get_allocation_result(display_type, pivot_or_not, aum)

@xl_func("dataframe<index=False>, str, str, int, str[], str[], str[], float[], float[], int[], float, str, str, float: dataframe<index=False>", auto_resize=True)
def post_trade_allocation_data_frame_runner(order_data, expand_or_contract, 
    criteria_option, lot_size, company, account_numer, 
    broker, allocation_percentage ,
    previous_pnl_acct,
    previous_net_pos_acct,
    eod_price,
    display_type, pivot_or_not, aum):

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
        display_type: str
            ways to display the result, could be pnl, allocation, position
        pivot_or_not: str YES OR NO
            whether to do pivot table view or not for the result
    

    """

    managed_account = {i:(company, acct_num, bro, af) for i, company, acct_num, bro, af in zip(range(1,len(allocation_percentage)+1), company, account_numer, broker, allocation_percentage, display_type, pivot_or_not )}

    post_allocation = AssetAllocation(managed_account=managed_account,
                                      which_action=expand_or_contract,
                                      criteria_option=criteria_option,
                                      lot_size=lot_size,
                                      previous_pnl_acct=previous_pnl_acct,
                                      previous_net_pos_acct=previous_net_pos_acct,
                                      eod_price=eod_price,
                                      use_file=0,
                                      order_data = order_data
                                      )


    post_allocation.fill_orders_file(file_name)
    post_allocation.allocate_trade()
    
    if pivot_or_not.lower() in ['y', 'yes','true']:
        pivot_or_not =True
    elif pivot_or_not.lower() in ['n','no','false']:
        pivot_or_not = False
    else:
        raise ValueError('pivot_or_not should be either YES or NO')
        
    return post_allocation.get_allocation_result(display_type, pivot_or_not, aum)

@xl_func("str, str, str : dataframe<index=False>", auto_resize=True)
def get_order_data(file_path, start_date, end_date):

    data  = pandas.read_csv(file_path, encoding='UTF-8')
    
    sd =  start_date
    ed =  end_date
    try:
        data['CALENDAR DATE'] = data['CALENDAR DATE'].apply(lambda x : datetime.datetime.strptime(x, '%m/%d/%Y').date())
    except:
        try:
            data['CALENDAR DATE'] = data['CALENDAR DATE'].apply(lambda x : datetime.datetime.strptime(x, '%m/%d/%y').date())
        except:
            try:
                data['CALENDAR DATE'] = data['CALENDAR DATE'].apply(lambda x :(datetime.datetime(1900, 1, 1)+ datetime.timedelta(days=x-2)).date())
            except:
                raise ValueError('Could not transform Date to datetime in raw data')
    
    try:
        sd = datetime.datetime.strptime(sd, '%m/%d/%Y').date()
        ed = datetime.datetime.strptime(ed, '%m/%d/%Y').date()
    except:
        try:
           sd = datetime.datetime.strptime(sd, '%m/%d/%y').date()
           ed = datetime.datetime.strptime(ed, '%m/%d/%y').date()
        except:
            try:
                sd = (datetime.datetime(1900, 1, 1)+ datetime.timedelta(days=int(sd)-2)).date()
                ed = (datetime.datetime(1900, 1, 1)+ datetime.timedelta(days=int(ed)-2)).date()
            except:
                raise ValueError('Could not transform start date and end date to datetime')
        
    data = data[(data['CALENDAR DATE'] >= sd )&(data['CALENDAR DATE'] <= ed)]
    data['CALENDAR DATE'] = data['CALENDAR DATE'].apply(lambda x: x.strftime("%m/%d/%Y"))
    return data

