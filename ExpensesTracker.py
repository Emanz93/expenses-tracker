import os
import sys
import json
import threading

# libraries
from controller import read_json, import_expences, import_payslips, re_train_classifier
from csv_lib import UnkownBankError

# tkinter
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import font
from tkinter import filedialog, messagebox

""" ExpensesTracker """

# TODO: add waiting window for both import and training
# TODO: double check the duplicates in the import phase
# TODO: do a better sampling of the data for the training 


class App:
    """ Class that represent the GUI. """
    
    def __init__(self, r):
        """ Constructor of the graphical user interface """
        self.root = r
        
        self.settings = read_json('res/settings.json')
        
        # screen settings
        self.width = 600
        self.height = 350
        self._set_screen_settings(title="Expences Management", width=self.width, height=self.height)

        # set the style
        self._set_azure_style()

        # populate the GUI
        self._create_main_frame()

    def _set_screen_settings(self, title="Title", width=500, height=500):
        """ Set the screen settings and the window placement.
            Parameters:
                title: String. Default title.
                width: Integer. Default width of the main view.
                height: Integer. Default height of the main view.
        """
        # setting title
        self.root.title(title)

        # set the icon
        self.png_icon_path = 'res/wallet.png'
        self.ico_icon_path = 'res/wallet.ico'
        if not os.path.isfile(self.png_icon_path):
            # if not locally called, use the absolute path.
            self.png_icon_path = os.path.join(os.path.dirname(sys.argv[0]), self.png_icon_path)
            self.ico_icon_path = os.path.join(os.path.dirname(sys.argv[0]), self.ico_icon_path)
        self.root.tk.call('wm', 'iconphoto', self.root._w, tk.PhotoImage(file=self.png_icon_path))

        # setting window size
        screenwidth = self.root.winfo_screenwidth()
        screenheight = self.root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height,
                                    (screenwidth - width) / 2, (screenheight - height) / 2)
        self.root.geometry(alignstr)
        self.root.resizable(width=False, height=False)

    def _set_azure_style(self):
        """ Set the Azure Style """
        self.style = ttk.Style(self.root)
        style_path = 'res/style/azure.tcl' # if locally called, use local path.
        if not os.path.isfile(style_path):
            # if not locally called, use the absolute path.
            style_path = os.path.join(os.path.dirname(sys.argv[0]), style_path)
        self.root.tk.call('source', style_path)
        self.style.theme_use('azure')

    def _create_main_frame(self):
        """ Create the main frame """
        self.margin = 20 # margin to be used across the gui

        # Title label
        highlightFont = font.Font(family='Helvetica', name='appHighlightFont', size=12, weight='bold')
        title_label = ttk.Label(self.root, text="Expences Management", font=highlightFont)
        title_height = 32
        title_label.place(x=(self.width/2)-(self.width/8), y=self.margin, width=200, height=title_height)
        
        # Label frames
        frame_height = 105
        csv_frame = ttk.Labelframe(self.root, text=' Expenses import ')
        frame_width = self.width-2*self.margin
        csv_frame.place(x=self.margin, y=self.margin+title_height, width=frame_width, height=frame_height)
        payslips_frame = ttk.Labelframe(self.root, text=' Payslips import ')
        payslips_frame.place(x=self.margin, y=2*self.margin+title_height+frame_height, width=frame_width, height=frame_height)

        # CSV selection section
        cmp_height = 32 # height of the components
        excel_label = ttk.Label(csv_frame, text="CSV file:")
        label_width = 60
        excel_label.place(x=5, y=5, width=label_width, height=cmp_height)

        self.csv_file_var = tk.StringVar(csv_frame)
        csv_entry = ttk.Entry(csv_frame, textvariable=self.csv_file_var)
        entry_width = 400
        csv_entry.place(x=label_width+10, y=5, width=entry_width, height=cmp_height)
        
        select_csv_btn = ttk.Button(csv_frame, text="Select", command=self.select_csv_btn_callback, style='Accentbutton')
        select_csv_btn.place(x=label_width+10*2+entry_width, y=5, width=70, height=cmp_height)

        import_csv_btn = ttk.Button(csv_frame, text="Import CSV", command=self.import_csv_btn_callback, style='Accentbutton')
        import_csv_btn.place(x=(frame_width/2)-130, y=15+cmp_height, width=120, height=cmp_height)

        retrain_csv_btn = ttk.Button(csv_frame, text="Retrain Classifier", command=self.retrain_btn_callback, style='Accentbutton')
        retrain_csv_btn.place(x=(frame_width/2)+10, y=15+cmp_height, width=120, height=cmp_height)

        # Payslips selection section
        cmp_height = 32 # height of the components
        excel_label = ttk.Label(payslips_frame, text="Payslip PDF file:")
        label_width = 90
        excel_label.place(x=5, y=5, width=label_width, height=cmp_height)

        self.payslip_filename_var = tk.StringVar(payslips_frame)
        payslip_entry = ttk.Entry(payslips_frame, textvariable=self.payslip_filename_var)
        entry_width = 370
        payslip_entry.place(x=label_width+10, y=5, width=entry_width, height=cmp_height)
        
        select_payslip_btn = ttk.Button(payslips_frame, text="Select", command=self.select_payslip_btn_callback, style='Accentbutton')
        select_payslip_btn.place(x=label_width+10*2+entry_width, y=5, width=70, height=cmp_height)

        import_payslip_btn = ttk.Button(payslips_frame, text="Import", command=self.import_payslip_btn_callback, style='Accentbutton')
        import_payslip_btn.place(x=(frame_width/2)-35, y=15+cmp_height, width=70, height=cmp_height)
        
        # Exit button
        exit_btn = ttk.Button(self.root, text="Exit", command=sys.exit, style='Accentbutton')
        exit_btn.place(x=(self.width/2)-40, y=self.height-cmp_height-self.margin, width=80, height=cmp_height)

    def create_waiting_window(self):
        """ Create a toplevel window to display the waiting message """
        print('create_waiting_window')
        height=50
        width=150
        self.waiting_window = tk.Toplevel(self.root)

        # create a frame container for the label
        fr = ttk.Frame(self.waiting_window, height=height, width=width)
        fr.pack()
        win_label = ttk.Label(fr, text='Please wait...')
        win_label.pack()

        # setting window size
        screenwidth = self.root.winfo_screenwidth()
        screenheight = self.root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.waiting_window.geometry(alignstr)

        # icon
        self.waiting_window.tk.call('wm', 'iconphoto', self.waiting_window._w, tk.PhotoImage(file=self.png_icon_path))

    def select_csv_btn_callback(self):
        """ Callback to set the source CSV file of the expences. """
        filename = filedialog.askopenfilename(title="Select expences file in CSV format...")
        if os.path.isfile(filename):
            self.csv_filename = filename
            self.csv_file_var.set(filename)
        else:
            messagebox.showerror("Error", "No input file selected.")

    def import_csv_btn_callback(self):
        """ Callback to the import button. It imports the expences in gspread. """
        if self.csv_file_var.get() != '':
            if self.csv_filename.endswith('csv'):
                try:
                    # start a thread to avoid the GUI to freeze
                    t = threading.Thread(target=import_expences, args=(self.csv_filename, self.settings))
                    t.start()
                    # import_expences(self.csv_filename, self.settings)

                    # start the waiting window in another thread.
                    self.create_waiting_window()
                    self.root.update()

                    # wait for the thead to be completed
                    t.join()
                    
                    # destroy the waiting window
                    self.waiting_window.destroy()

                    messagebox.showinfo(title="Info", message="Import completed.")
                except UnkownBankError:
                    messagebox.showerror(title="Error", message="The CSV format is unexpected or is coming from an unsupported bank.")
            else:
                messagebox.showerror(title="Error", message="The file is not a CSV format.")
        else:
            messagebox.showerror(title="Error", message="No expences file selected.")

    def select_payslip_btn_callback(self):
        """ Callback to set the source PDF file of the payslip. """
        filename = filedialog.askopenfilename(title="Select payslip file in PDF format...")
        if os.path.isfile(filename):
            self.payslip_filename = filename
            self.payslip_filename_var.set(filename)
        else:
            messagebox.showerror("Error", "No input file selected.")

    def import_payslip_btn_callback(self):
        """ Callback to the import button. It imports the payslisps in gspread. """
        if self.payslip_filename_var.get() != '':
            if self.payslip_filename.endswith('pdf'):
                import_payslips(self.payslip_filename, self.settings)
                messagebox.showinfo("Success", "Import completed.")
            else:
                messagebox.showerror("Error", "The file is not a PDF format.")
        else:
            messagebox.showerror("Error", "No expences file selected.")
    
    def retrain_btn_callback(self):
        """ Callback to the button for retrain the classifier. """
        try:
            re_train_classifier(self.settings)
            messagebox.showinfo("Success", "Expences classifier re-trained.")
        except Exception as e:
            messagebox.showerror("Error", "Expences classifier failed the re-train. {}".format(e))
            raise e


# **************************
#            MAIN           
# **************************
if __name__ == '__main__':
    # Init for the GUI pop-ups
    root = tk.Tk()
    app = App(root)
    root.mainloop()
