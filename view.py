from side_panel import SidePanel
import tkinter as Tk
from data_explore import DataPanel

class View:
    def __init__(self, root, model):
        self.datapanel = DataPanel(root)
        self.sidepanel = SidePanel(root)




