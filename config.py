import os

file_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(file_path, os.path.pardir))
# project_path = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(project_path, 'order_data/orders_{}.csv')
ALLOCATION_CONFIG_PATH = os.path.join(project_path, 'config/allocation_config_{}.csv')
SYMBOL_CONFIG_PATH = os.path.join(project_path, 'config/symbol_config.csv')
AUM_CONFIG_PATH = os.path.join(project_path, 'config/aum_config.csv')

EOD_PNL = os.path.join(project_path, 'allocation_result/eod_pnl_{}.csv')
EOD_POSITION = os.path.join(project_path, 'allocation_result/eod_position_{}.csv')
ALLOCATION_RESULT_DETAIL = os.path.join(project_path, 'allocation_result/allocation_{}.csv')
ALLOCATION_RESULT_DAILY = os.path.join(project_path, 'allocation_result/daily/allocation_{}_{}.csv')
PNL_RESULT_DETAIL = os.path.join(project_path, 'allocation_result/pnl_{}.csv')
POSITION_RESULT_DETAIL = os.path.join(project_path, 'allocation_result/position_{}.csv')

EOD_PNL_DAILY  = os.path.join(project_path, 'allocation_result/daily/eod_pnl_{}_{}.csv')
EOD_POSITION_DAILY  = os.path.join(project_path, 'allocation_result/daily/eod_position_{}_{}.csv')
ALLOCATION_RESULT_DETAIL_DAILY  = os.path.join(project_path, 'allocation_result/daily/allocation_{}_{}.csv')
PNL_RESULT_DETAIL_DAILY  = os.path.join(project_path, 'allocation_result/daily/pnl_{}_{}.csv')
POSITION_RESULT_DETAIL_DAILY  = os.path.join(project_path, 'allocation_result/daily/position_{}_{}.csv')
