# Post-Trade-Allocation
Post Trade Allocation Implementation based on US20130013482A1

This code is the local version of Post Trade Allocation Implementation using PYTHON for demo purpose, 
the cloud version would be based on C++ and run in distributed mode.

The project contains several Part:
1) Post Trade Dynamic allocation to minimize the pnl gap between mulitple management account traded with the same fund manager.
2) Simple UI to demo the allocation.
3) XLL Addin to easily utilize the code in excel

1) Post Trade Dynamic allocation algorithm:
This method is based on Pattern US20130013482A1.
Component:
post_trade_allocation.py: Allocation Algorithm Class to handle data cleaning and algorithm calculation
new_post_trade_allocation.py: Optimization of the original code based on numbra to demonstrate the speedup with distributed mode
allocation_optimization.py: helper function to do allocation
jit_grid_search.py:  optimization of original code using numbra to speed up the code
last_trade_allocation.py helper function of the optimization
config.py config file of the related path

2) UI Component:
Following MVC design pattern:
- view.py
- side_pannel.py
- model.py

3) XLL Addin
- allocation_xll.py  using xll library
- allocation_xlwings.py using xlwings library
