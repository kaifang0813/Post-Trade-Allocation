import tkinter as Tk # python 3

from model import Model
from view import View
from config import *
import datetime
from tkinter import messagebox
import pandas
import os
import platform
from tkinter import filedialog

class Controller():
    def __init__(self):
        self.root = Tk.Tk()
        self.model = Model()
        self.view = View(self.root, self.model)
        self.view.sidepanel.loadDataBut.bind("<Button>", self.load_order_data)
        self.view.sidepanel.AFPreviewBut.bind("<Button>", self.load_af_data)
        self.view.sidepanel.symbolPreviewBut.bind("<Button>", self.load_symbol_data)
        self.view.sidepanel.editAFBut.bind("<Button>", self.edit_allocation_factor)
        self.view.sidepanel.AUMPreviewBut.bind("<Button>", self.load_aum_data)
        self.view.sidepanel.editAUMBut.bind("<Button>", self.edit_aum)

        self.view.sidepanel.editSymbolBut.bind("<Button>", self.edit_symbol_factor)
        self.view.sidepanel.symbol.trace('w', self.change_symbol_dropdown)
        self.view.sidepanel.criteria_option.trace('w', self.change_criteria_option_dropdown)
        self.view.sidepanel.order_process.trace('w', self.change_order_process_dropdown)

        self.view.sidepanel.CalculateButton.bind("<Button>", self.calculate)
        self.view.sidepanel.display_option.trace('w', self.change_display_dropdown)
        self.view.sidepanel.ResultPreviewBut.bind("<Button>",self.update_result)
        self.view.sidepanel.saveResultBut.bind("<Button>",self.save_data)
        self.view.sidepanel.recentResultButton.bind("<Button>",self.get_recent_result)
        self.view.sidepanel.addDataBut.bind("<Button>",self.add_order_data)

        self.get_symbol_data(update=False)
        self.view.sidepanel.symbol.set(self.model.all_symbol['Symbol'].unique().tolist()[0])

        menu = self.view.sidepanel.symbol_popupMenu["menu"]
        for string in set(self.model.all_symbol['Symbol'].unique().tolist()):
            menu.add_command(label=string,
                             command=lambda value=string:
                             self.om_variable.set(value))

    def change_order_process_dropdown(self, *args):
        print(self.view.sidepanel.order_process.get())

    def change_display_dropdown(self, *args):
        print(self.view.sidepanel.display_option.get())

    def change_criteria_option_dropdown(self, *args):
        print(self.view.sidepanel.criteria_option.get())

    def change_symbol_dropdown(self, *args):
        print(self.view.sidepanel.symbol.get())

    def run(self):
        self.root.title("Post Trade Allocation")
        self.root.deiconify()
        self.root.mainloop()

    def get_sd_ed(self):
        start_date = self.view.sidepanel.startDate.get()
        end_date = self.view.sidepanel.endDate.get()

        try:
            sd = datetime.datetime.strptime(start_date, '%m/%d/%Y').date()
            ed = datetime.datetime.strptime(end_date, '%m/%d/%Y').date()
        except:
            try:
                sd = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
                ed = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
            except:
                messagebox.showerror("Error",
                                     'Could Not recognize the start date and end date, should be formated as %Y-%m-%d or %m/%d/%Y')
                # raise ValueError('Could Not recognize the start date and end date, should be formated as %Y-%m-%d or %m/%d/%Y')
                return None
        return (start_date, end_date, sd, ed)

    def get_order_data(self, update=True):
        self.view.datapanel.update(DATA_PATH.format(self.view.sidepanel.symbol.get()))

        date_list = self.get_sd_ed()
        if date_list is not None:
            start_date, end_date, sd, ed = date_list
        else:
            return
        temp_data = self.view.datapanel.table.model.df
        if temp_data.empty:
            messagebox.showerror("Error",
                                 'Could find order data')
            return
        try:
            temp_data['CALENDAR DATE'] = temp_data['CALENDAR DATE'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y').date())
        except:
            try:
                temp_data['CALENDAR DATE'] = temp_data['CALENDAR DATE'].apply(
                    lambda x: datetime.datetime.strptime(x, '%m/%d/%y'))
            except:
                messagebox.showerror("Error",
                                     'Could Not process CALENDAR DATE, should be formated as %m/%d/%y or %m/%d/%Y')
                return
        temp_data = temp_data[(temp_data['CALENDAR DATE']<=ed) & (temp_data['CALENDAR DATE']>=sd)]

        if temp_data.empty:
            messagebox.showerror("Error", 'Do not have any data in the date range {} - {}'.format(start_date, end_date))
            if update:
                self.view.datapanel.table.model.df = pandas.DataFrame()
                self.view.datapanel.table.redraw()
            return

        self.model.order_data = temp_data
        if update:
            self.view.datapanel.table.model.df = temp_data
            self.view.datapanel.table.redraw()

    def get_allocation_data(self, update=True):
        date_list = self.get_sd_ed()
        if date_list is not None:
            start_date, end_date, sd, ed = date_list
        else:
            return

        af_file = pandas.read_csv(ALLOCATION_CONFIG_PATH.format(self.view.sidepanel.symbol.get()))
        af_file.dropna(inplace=True)
        if af_file.empty:
            messagebox.showerror("Error",
                                 'No allocation factor provided')
        try:
            af_file['As_of_Date'] = af_file['As_of_Date'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%y').date())
            af_file['Entry_Date'] = af_file['Entry_Date'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%y').date())
        except:
            try:
                af_file['As_of_Date'] = af_file['As_of_Date'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y').date())
                af_file['Entry_Date'] = af_file['Entry_Date'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y').date())
            except:
                messagebox.showerror("Error",
                                     'Could Not process As_of_Date and Entry_Date, should be formated as %m/%d/%y or %m/%d/%Y')
                return

        af_file = af_file[af_file['As_of_Date'] <= sd]
        if af_file.empty:
            messagebox.showerror("Error",
                                 'No allocation factor with as_of_date before the allocation date')
            return

        max_date = max(af_file['As_of_Date'].unique())
        af_file = af_file[af_file['As_of_Date'] == max_date]
        max_date = max(af_file['Entry_Date'].unique().tolist())
        af_file = af_file[af_file['Entry_Date'] == max_date]
        self.model.managed_account_file = af_file
        maf = self.model.managed_account_file
        self.model.managed_account = {int(k):(c, n, b, p)
                                      for k, c, n, b, p
                                      in zip( maf['Account'],
                                              maf['Company'],
                                              maf['Account Number'],
                                              maf['Broker'],
                                              maf['Percentage'])}
        if update:
            self.view.datapanel.table.model.df = af_file
            self.view.datapanel.table.redraw()

    def get_aum(self, update=True):
        date_list = self.get_sd_ed()
        if date_list is not None:
            start_date, end_date, sd, ed = date_list
        else:
            return

        aum_file = pandas.read_csv(AUM_CONFIG_PATH)
        aum_file.dropna(inplace=True)
        if aum_file.empty:
            messagebox.showerror("Error",
                                 'No allocation factor provided')


        try:
            aum_file['Date'] = aum_file['Date'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%y').date())
        except:
            try:
                aum_file['Date'] = aum_file['Date'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y').date())
            except:
                messagebox.showerror("Error",
                                     'Could Not process DATE, should be formated as %m/%d/%y or %m/%d/%Y')
                return
        aum_file = aum_file[aum_file['Date'] <= sd]
        if aum_file.empty:
            messagebox.showerror("Error",
                                 'No allocation factor with as_of_date before the allocation date')
            return

        self.model.aum = aum_file.iat[0,1]

        if update:
            self.view.datapanel.table.model.df = aum_file
            self.view.datapanel.table.redraw()

    def get_symbol_data(self, update=True):
        try:
            symbol_file = pandas.read_csv(SYMBOL_CONFIG_PATH)
            self.model.all_symbol = symbol_file
            if update:
                self.view.datapanel.table.model.df = symbol_file
                self.view.datapanel.table.redraw()
        except:
            messagebox.showerror("Error",
                     'Fail to load symbol data, please check whether the symbol file has correct format')

    def load_aum_data(self,event):
        try:
            self.get_aum()
        except:
            messagebox.showerror("Error",
                     'Fail to load aum data, please check whether the aum file has correct format')

    def load_af_data(self, event):
        try:
            self.get_allocation_data()
        except:
            messagebox.showerror("Error",
                     'Fail to load allocation factor data, please check whether the allocation factor file has correct format')

    def load_symbol_data(self, event):
        self.get_symbol_data()

    def load_order_data(self, event):
        try:
            self.get_order_data()
        except:
            messagebox.showerror("Error",
                     'Fail to load order data, please check whether the order file has correct format')

    def calculate(self, event):
        # try:
            # preparing for referrence data
        if self.model.managed_account_file is None:
            self.get_allocation_data(update=False)

        self.get_order_data(update=False)
        self.model.expand_or_contract = self.view.sidepanel.order_process.get()
        self.model.criteria_option = self.view.sidepanel.criteria_option.get()
        self.model.symbol = self.view.sidepanel.symbol.get()
        self.get_symbol_data(update=False)
        self.get_aum(update=False)
        symbol_data = self.model.all_symbol
        symbol_data = symbol_data[symbol_data['Symbol'] == self.model.symbol]
        if len(symbol_data) >0:
            self.model.lot_size = symbol_data.iat[0,2] # the third row is the unit quantity
        else:
            messagebox.showerror("Error",
                                 'could not find the lot size of symbol {} '.format(self.model.symbol))
            return
        if self.view.sidepanel.pnl_var.get() == 1:
            self.get_last_result(update=False)
        if self.view.sidepanel.pnl_var.get() == 0:
            self.model.previous_pnl_acct = None
        if self.view.sidepanel.position_var.get() ==0:
            self.model.previous_net_pos_acct = None
            self.model.eod_price = None

        self.model.calculate()
        self.view.datapanel.table.model.df = self.model.result
        self.view.datapanel.table.redraw()
        # except:
        #     messagebox.showerror("Error",
        #              'Fail to calculate allocation result, please check the command prompt/terminal for error message')

    def update_result(self, event):
        try:
            if self.model.result is None:
                self.model.calculate()

            self.model.result = self.model.display(self.view.sidepanel.display_option.get(), bool(self.view.sidepanel.pivot_var.get()))
            self.view.datapanel.table.model.df =self.model.result
            self.view.datapanel.table.redraw()
        except:
            messagebox.showerror("Error",
                     'Fail to update result, please check the command prompt/terminal for error message')


    def edit_excel_file(self, file_path):
        try:
            if platform.system() == 'Darwin':
                os.system('open {} -a "Microsoft Excel"'.format(file_path))
            elif platform.system() == 'Windows':
                os.system('start excel "{}"'.format(file_path))
            else:
                raise ValueError('Only support windows and mac excel now, please contact client service to add more platform')
        except:
            messagebox.showerror("Error",
                     'Fail to open file, please check the command prompt/terminal for error message')

    def edit_allocation_factor(self, event):
        self.edit_excel_file(ALLOCATION_CONFIG_PATH.format(self.view.sidepanel.symbol.get()))

    def edit_symbol_factor(self, event):
        self.edit_excel_file(SYMBOL_CONFIG_PATH)

    def edit_aum(self, event):
        self.edit_excel_file(AUM_CONFIG_PATH)

    def update_result_file(self, fp, display, pivot=True, daily = False):
        new_data = self.model.display(display, pivot)

        if daily:
            date_list = self.get_sd_ed()
            if date_list is not None:
                start_date, end_date, sd, ed = date_list
            else:
                return
            fp = fp.format(format(self.view.sidepanel.symbol.get()), ed.strftime('%m-%d-%Y'))
        else:
            fp = fp.format(format(self.view.sidepanel.symbol.get()))

        mode = 'a' if os.path.exists(fp) else 'w'
        with open(fp, mode) as f:
            print('check file {} existence, if not create'.format(fp))

        if daily:
            data = new_data
        else:
            try:
                old_data = pandas.read_csv(fp)
            except:
                old_data = pandas.DataFrame(columns = ['Date'])
            try:
                old_data['Date'] = old_data['Date'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y').date())
                new_data['Date'] = new_data['Date'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y').date())
                flag =1
            except:
                try:
                    old_data['Date'] = old_data['Date'].apply(
                        lambda x: datetime.datetime.strptime(x, '%m/%d/%y'))
                    new_data['Date'] = new_data['Date'].apply(
                        lambda x: datetime.datetime.strptime(x, '%m/%d/%y'))
                    flag = 2
                except:
                    messagebox.showerror("Error",
                                         'Could Not process DATE, both original order data and new order data should be formated as %m/%d/%y or %m/%d/%Y')
                    return
            try:
                max_date = old_data['Date'].iloc[-1]
                new_data = new_data[new_data['Date'] > max_date]
            except:
                 messagebox.showerror("Warning",
                                         'Do not have any old data')
                    
            data = old_data.append(new_data)
            if flag ==1:
                data['Date']  = data['Date'].apply(lambda x: x.strftime('%m/%d/%Y'))
            else:
                data['Date'] = data['Date'].apply(lambda x: x.strftime('%m/%d/%y'))
        export_csv = data.to_csv (fp, index = None, header=True)
        return export_csv

    def save_data(self,event):
        self.update_result_file(EOD_PNL, 'eod_pnl', True)
        self.update_result_file(EOD_POSITION, 'eod_position', True)
        self.update_result_file(ALLOCATION_RESULT_DETAIL, 'allocation', False)
        self.update_result_file(PNL_RESULT_DETAIL, 'pnl', False)
        self.update_result_file(POSITION_RESULT_DETAIL,'position', False)
        self.update_result_file(EOD_RETURN, 'return', True)
        # daily
        self.update_result_file(EOD_PNL_DAILY, 'eod_pnl', True, True)
        self.update_result_file(EOD_RETURN_DAILY, 'return', True, True)

        self.update_result_file(EOD_POSITION_DAILY, 'eod_position', True, True)
        self.update_result_file(ALLOCATION_RESULT_DETAIL_DAILY, 'allocation', False, True)
        self.update_result_file(PNL_RESULT_DETAIL_DAILY, 'pnl', False, True)
        self.update_result_file(POSITION_RESULT_DETAIL_DAILY,'position', False, True)

    def add_order_data(self, event):
        try:
            myfiletypes = [('CSV File', '*.csv'), ('All files', '*')]
            self.open_file = filedialog.Open(self.root, filetypes=myfiletypes)
            text = self.open_file.show()
            new_data = pandas.read_csv(text)
            fp = DATA_PATH.format(format(self.view.sidepanel.symbol.get()))
            mode = 'a' if os.path.exists(fp) else 'w'
            with open(fp, mode) as f:
                print('check file {} existence, if not create'.format(fp))
            old_data = pandas.read_csv(fp)
            flag = 0
            try:
                old_data['CALENDAR DATE'] = old_data['CALENDAR DATE'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y').date())
                new_data['CALENDAR DATE'] = new_data['CALENDAR DATE'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y').date())
                flag =1
            except:
                try:
                    old_data['CALENDAR DATE'] = old_data['CALENDAR DATE'].apply(
                        lambda x: datetime.datetime.strptime(x, '%m/%d/%y'))
                    new_data['CALENDAR DATE'] = new_data['CALENDAR DATE'].apply(
                        lambda x: datetime.datetime.strptime(x, '%m/%d/%y'))
                    flag = 2
                except:
                    messagebox.showerror("Error",
                                         'Could Not process CALENDAR DATE, both original order data and new order data should be formated as %m/%d/%y or %m/%d/%Y')
                    return

            max_date = old_data['CALENDAR DATE'].iloc[-1]
            new_data = new_data[new_data['CALENDAR DATE'] > max_date]
            data = old_data.append(new_data)
            if flag ==1:
                data['CALENDAR DATE']  = data['CALENDAR DATE'].apply(lambda x: x.strftime('%m/%d/%Y'))
            else:
                data['CALENDAR DATE'] = data['CALENDAR DATE'].apply(lambda x: x.strftime('%m/%d/%y'))
            export_csv = data.to_csv(fp, index=None, header=True)
            self.get_order_data()
            return export_csv
        except:
            messagebox.showerror("Error",
                     'Fail to add new order data, please check the command prompt/terminal for error message')


    def get_last_result(self, update=True):
        try:
            pnl=pandas.read_csv(EOD_PNL.format(self.view.sidepanel.symbol.get()))
            position = pandas.read_csv(EOD_POSITION.format(self.view.sidepanel.symbol.get()))
            result = pandas.merge(pnl, position, how='inner', on=['Date', 'Time', 'Price'])
            try:
              result['Date'] = result['Date'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%y').date())
            except:
                try:
                    result['Date'] = result['Date'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y').date())
                except:
                    messagebox.showerror("Error",
                                         'could not process Date format of eod pnl and eod position data, should be %m/%d/%y or %m/%d/%Y ')
                    return

            date_list = self.get_sd_ed()
            if date_list is not None:
                start_date, end_date, sd, ed = date_list
            else:
                return

            result = result[result['Date'] < sd]
            if result.empty:
                messagebox.showerror("Error",
                                     'Do not have any previous pnl and position result availale ')
                return
            last_row = result.iloc[-1] # the data should be ascending order based on time
            self.model.eod_price = last_row['Price']
            self.model.previous_pnl_acct = []
            self.model.previous_net_pos_acct = []
            if self.model.managed_account is None:
                self.get_allocation_data(update=False)

            n_acct = len(self.model.managed_account)
            for i in self.model.managed_account.keys():
                try:
                    self.model.previous_pnl_acct.append(last_row['Acct_PNL_{}'.format(str(i))] )
                except:
                    self.model.previous_pnl_acct.append(0)

                try:
                    self.model.previous_net_pos_acct.append(last_row['Acct_Position_{}'.format(str(i))] )
                except:
                    self.model.previous_net_pos_acct.append(0)

            print(self.model.previous_pnl_acct)
            print(self.model.previous_net_pos_acct)
            if update:
                self.view.datapanel.table.model.df = result.tail(1)
                self.view.datapanel.table.redraw()
        except:
            messagebox.showerror("Error",
                     'Fail to get last allocaton result, please check the command prompt/terminal for error message')

    def get_recent_result(self,event):
        self.get_last_result()




