
import datetime
from tkinter import *
import os
from tkinter import filedialog

PADX_SIZE =5
TEXT_SIZE  = 10

def makeentry(parent, caption, width=None, side = TOP, **options):
    Label(parent, text=caption, padx = PADX_SIZE,  font = ("Times",TEXT_SIZE ), anchor="w").pack(side=side,fill=BOTH)
    entry = Entry(parent, **options)
    if width:
        entry.config(width=width)
    entry.pack(side=side)
    return entry



class SidePanel():
    def __init__(self, root):
        self.frame1 = Frame( root ,width=2, bd=2, relief='sunken')
        self.frame1.pack(side=TOP, fill=BOTH, expand=1)

        self.frame6 = Frame( root , width=2, bd=1, relief='sunken')
        self.frame6.pack(side=RIGHT, fill=BOTH, expand=1)

        self.frame2 = Frame( root , width=2, bd=1, relief='sunken')
        self.frame2.pack(side=LEFT, fill=BOTH, expand=1)


        self.frame3 = Frame( root ,width=2, bd=2, relief='sunken')
        self.frame3.pack(side=TOP, fill=BOTH, expand=1)

        self.frame4 = Frame( root ,width=2, bd=2, relief='sunken')
        self.frame4.pack(side=BOTTOM, fill=BOTH, expand=1)

        self.frame5 = Frame( root ,width=2, bd=2, relief='sunken')
        self.frame5.pack(side=BOTTOM, fill=BOTH, expand=1)

        self.frame7 = Frame( self.frame2 ,width=2, bd=2, relief='sunken')
        self.frame7.pack(side=BOTTOM, fill=BOTH, expand=1)

        self.frame8 = Frame( self.frame6 ,width=2, bd=2)
        self.frame8.pack(side=BOTTOM, fill=BOTH, expand=1)

        self.frame9 = Frame( self.frame8 ,width=2, bd=2, relief='sunken')
        self.frame9.pack(side=LEFT, fill=BOTH, expand=1)


        # ==================== Entry In Frame2  ==========================#
        # Date Input
        start_date = StringVar()
        start_date.set(datetime.datetime.today().strftime("%m/%d/%Y"))
        self.startDate = makeentry(self.frame1, "Start Date:", None, LEFT, textvariable=start_date, )

        end_date = StringVar()
        end_date.set(datetime.datetime.today().strftime("%m/%d/%Y"))
        self.endDate = makeentry(self.frame1, "End Date:", None,  LEFT, textvariable=end_date)

        # Create a Tkinter variable
        self.symbol = StringVar()

        # Dictionary with options
        # self.symbol_choices = {''}
        self.symbol.set('')  # set the default option

        self.symbol_popupMenu = OptionMenu(self.frame1, self.symbol, ())
        Label(self.frame1, text="Choose a Symbol", padx=PADX_SIZE, font = ("Times",TEXT_SIZE ), anchor="w").pack(side=LEFT,fill=BOTH)
        self.symbol_popupMenu.pack(side=LEFT,fill=BOTH)

        #Optimization Parameters
        self.criteria_option = StringVar()
        self.criteria_option.set('1')

        # Dictionary with options
        self.criteria_choices = {'0', '1', '2', '3', '4'}

        self.criteria_optionMenu = OptionMenu(self.frame2, self.criteria_option, *self.criteria_choices)
        Label(self.frame2, text="Criteria_Option", padx=PADX_SIZE,  font = ("Times",TEXT_SIZE ), anchor="w").pack(side=TOP,fill=BOTH)
        self.criteria_optionMenu.pack(side=TOP,fill=BOTH)

        #Optimization Parameters
        self.order_process = StringVar()
        self.order_process.set('None')

        # Dictionary with options
        self.order_process_choices= {'None', 'expand'}

        self.PreprocessMenu = OptionMenu(self.frame2, self.order_process, *self.order_process_choices)
        Label(self.frame2, text="PreProcessing_Option", padx=PADX_SIZE,  font = ("Times",TEXT_SIZE ), anchor="w").pack(side=TOP,fill=BOTH)
        self.PreprocessMenu.pack(side=TOP,fill=BOTH)



        #Symol Table
        self.data_input = Label(self.frame7, text="Reference Data:", padx=PADX_SIZE, font = ("Times",TEXT_SIZE ), anchor="w")
        self.data_input.pack(side="top", fill=BOTH)

        #Allocation Factor Input
        self.symbolPreviewBut = Button(self.frame7, text="Preview Symbol Data")
        self.symbolPreviewBut.pack(side="top",fill=BOTH)

        self.editSymbolBut = Button(self.frame7, text="Edit Symbol")
        self.editSymbolBut.pack(side="top",fill=BOTH)



        self.addDataBut = Button(self.frame7, text="Add Order Data")
        self.addDataBut.pack(side="top",fill=BOTH)


        # ====================   Button in frame 3  =====================#
        #Order Data Input
        self.data_input = Label(self.frame3, text="Order Data Input:",padx = PADX_SIZE, font = ("Times",TEXT_SIZE ), anchor="w")
        self.data_input.pack(side="top",fill=BOTH)

        self.loadDataBut = Button(self.frame3, text="Load Order Data ")
        self.loadDataBut.pack(side="top",fill=BOTH)

        #Allocation Factor Input
        self.AFPreviewBut = Button(self.frame5, text="Preview Allocation Factor ")
        self.AFPreviewBut.pack(side="top",fill=BOTH)

        self.editAFBut = Button(self.frame5, text="Edit Allocation Factor")
        self.editAFBut.pack(side="top",fill=BOTH)

        # Use Last PNL AND POSITION
        self.data_input = Label(self.frame4, text="Previous PNL Info:",padx =PADX_SIZE, font = ("Times",TEXT_SIZE ), anchor="w")
        self.data_input.pack(side="top",fill=BOTH)

        self.pnl_var = IntVar(value=1)
        self.use_previous_pnl = Checkbutton(self.frame4, text="USE Previous PNL", padx =PADX_SIZE, font = ("Times",TEXT_SIZE ), anchor="w", variable=self.pnl_var)
        self.use_previous_pnl.pack(side="top", fill=BOTH)
        self.position_var = IntVar(value=1)
        self.use_previous_position = Checkbutton(self.frame4, text="USE Previous Position", padx = PADX_SIZE, font = ("Times",TEXT_SIZE ), anchor="w", variable=self.position_var )
        self.use_previous_position.pack(side="top", fill=BOTH)

        #Post Trade Allocation
        self.data_output = Label(self.frame6, text="Post Trade Allocation:",padx = PADX_SIZE, font = ("Times",TEXT_SIZE ), anchor="w")
        self.data_output.pack(side="top",fill=BOTH)

        #Save Result and Review Result

        self.recentResultButton = Button(self.frame6, text="Check Most Recent Result")
        self.recentResultButton.pack(side="top",fill=BOTH)

        self.CalculateButton = Button(self.frame6, text="Calculate")
        self.CalculateButton.pack(side="top",fill=BOTH)

        self.data_output = Label(self.frame8, text="Result:",padx = PADX_SIZE,  font = ("Times",TEXT_SIZE ), anchor="w")
        self.data_output.pack(side="top",fill=BOTH)

        #Optimization Parameters
        self.display_option = StringVar()
        self.display_option.set('allocation')

        self.display_choices = {'allocation', 'pnl', 'position', 'eod_pnl','eod_position', 'return'}
        self.display_optionMenu = OptionMenu(self.frame9, self.display_option, *self.display_choices)
        Label(self.frame9, text="Display Choice", padx=PADX_SIZE,font = ("Times",TEXT_SIZE ), anchor="w").pack(side=TOP,fill=BOTH)
        self.display_optionMenu.pack(side=TOP,fill=BOTH)

        self.pivot_var = IntVar(value=0)
        self.pivot_table = Checkbutton(self.frame9, text="PIVOT TABLE", padx = PADX_SIZE,  font = ("Times",TEXT_SIZE ), anchor="w", variable=self.pivot_var)
        self.pivot_table.pack(side=TOP, fill=BOTH)

        #Allocation Factor Input
        self.AUMPreviewBut = Button(self.frame9, text="Preview AUM ")
        self.AUMPreviewBut.pack(side="top",fill=BOTH)

        self.editAUMBut = Button(self.frame9, text="Edit AUM")
        self.editAUMBut.pack(side="top",fill=BOTH)

        self.ResultPreviewBut = Button(self.frame8, text="Preview Result Data ", compound=LEFT)
        self.ResultPreviewBut.pack(side="top", fill=BOTH)

        self.saveResultBut = Button(self.frame8, text="Save Result")
        self.saveResultBut.pack(side="top",fill=BOTH)




