from pandastable import Table, TableModel
import pandas
import tkinter as Tk

class DataPanel():
    def __init__(self, root):
        self.frame3 = Tk.Frame(root)
        self.frame3.__init__()
        self.frame3.pack( fill=Tk.BOTH, expand=1)
        df = pandas.DataFrame()
        self.table = Table(self.frame3, dataframe=df,
                                showtoolbar=True, showstatusbar=True)
        self.table.show()

    def update(self, file_path):

        self.table.importCSV(file_path)
        self.table.redraw()
