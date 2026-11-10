import os
import sys
import threading
from queue import Queue, Empty
import time

# libraries
from controller import read_json, import_expenses, import_payslips, re_train_classifier
from csv_lib import UnknownBankError

# tkinter
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import font
from tkinter import filedialog, messagebox

""" ExpensesTracker """

# TODO: the classification shall add a "needs review" flag for entries with classification with low confidence
# TODO: add multi-file import
# TODO: pre-fill the newest csv when starting the application
# TODO: automatically aggregate the same bar and restaurants entries, within the same day

# TODO: double check the duplicates in the import phase -> hex(hash(...))
# TODO: exclude certain categories from the training, e.g. miscellaneous and wedding.
# TODO: review parallelisation at import time (models files) and training

class ThreadCommunicator:
    """ Communication is the key! This class to help Threads to communicate better. """
    def __init__(self):
        self.is_done = threading.Event()   # set by the worker when finished, checked by the GUI
        self.message_queue = Queue()       # thread-safe message queue: worker puts, GUI gets
        self.error = None                  # Contains the exception raised by the worker thread
        self.traceback_str = ''            # Contains the traceback for logging

class App:
    """ Class that represent the GUI. """
    
    def __init__(self, r):
        """ Constructor of the graphical user interface """
        self.wait_label = None
        self.waiting_window = None
        self._worker_running = False  # re-entrance guard for _run_in_thread
        self.csv_filenames = []
        self.root = r
        
        if not os.path.exists('res/settings.json'):
            print('Cannot find settings file: res/settings.json')
            messagebox.showwarning(title="Error", message='Cannot find settings file: res/settings.json')
            return
        
        self.settings = read_json('res/settings.json')
        
        # screen settings
        self.width = 600
        self.height = 350
        self._set_screen_settings(title="Expenses Management", width=self.width, height=self.height)

        # set the style
        self._set_azure_style()

        # populate the GUI
        self._create_main_frame()

        # helper class
        self.communicator = ThreadCommunicator()

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
        if not os.path.isfile(self.png_icon_path): # if not locally called, use the absolute path.
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
        self.root.tk.call("set_theme", "light")
        #self.style.theme_use('azure')

    def _create_main_frame(self):
        """ Create the main frame """
        self.margin = 20 # margin to be used across the gui

        # Title label
        highlight_font = font.Font(family='Helvetica', name='appHighlightFont', size=12, weight='bold')
        title_label = ttk.Label(self.root, text="Expenses Management", font=highlight_font)
        title_height = 32
        title_label.pack(side=tk.TOP, anchor='n', pady=self.margin)

        # Version label
        version_label = ttk.Label(self.root, text="1.2.1")
        version_label.pack(side=tk.BOTTOM, anchor='se', padx=5, pady=5)
        
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

        select_csv_btn = ttk.Button(csv_frame, text="Select", command=self.select_csv_btn_callback, style='Accent.TButton')
        select_csv_btn.place(x=label_width+10*2+entry_width, y=5, width=70, height=cmp_height)

        import_csv_btn = ttk.Button(csv_frame, text="Import CSV", command=self.import_csv_btn_callback, style='Accent.TButton')
        import_csv_btn.place(x=(frame_width/2)-130, y=15+cmp_height, width=120, height=cmp_height)

        retrain_csv_btn = ttk.Button(csv_frame, text="Retrain Classifier", command=self.retrain_btn_callback, style='Accent.TButton')
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
        
        select_payslip_btn = ttk.Button(payslips_frame, text="Select", command=self.select_payslip_btn_callback, style='Accent.TButton')
        select_payslip_btn.config(state="disabled")
        select_payslip_btn.place(x=label_width+10*2+entry_width, y=5, width=70, height=cmp_height)

        import_payslip_btn = ttk.Button(payslips_frame, text="Import", command=self.import_payslip_btn_callback, style='Accent.TButton')
        import_payslip_btn.config(state="disabled")
        import_payslip_btn.place(x=(frame_width/2)-35, y=15+cmp_height, width=70, height=cmp_height)
        
        # Exit button
        exit_btn = ttk.Button(self.root, text="Exit", command=sys.exit, style='Accent.TButton')
        exit_btn.place(x=(self.width/2)-40, y=self.height-cmp_height-self.margin, width=80, height=cmp_height)

    def create_waiting_window(self):
        """ Create a toplevel window to display the waiting messages. """
        print('create_waiting_window')
        height=50
        width=150
        self.waiting_window = tk.Toplevel(self.root)

        # create a frame container for the label
        fr = ttk.Frame(self.waiting_window, height=height, width=width)
        fr.pack()
        self.wait_label = ttk.Label(fr, text='Please wait...')
        self.wait_label.pack()

        # setting window size
        screenwidth = self.root.winfo_screenwidth()
        screenheight = self.root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.waiting_window.geometry(alignstr)

        # icon
        self.waiting_window.tk.call('wm', 'iconphoto', self.waiting_window._w, tk.PhotoImage(file=self.png_icon_path))

    def _run_in_thread(self, target, args, success_message):
        """ Run target in a worker thread, updating the waiting window.
            Guarantees is_done is set even if target raises. """
        if self._worker_running:
            messagebox.showwarning("Busy", "An operation is already running. Please wait.")
            return
        self._worker_running = True
        self.communicator = ThreadCommunicator()

        def worker():
            try:
                target(*args, self.communicator)
            except Exception as e:
                import traceback
                self.communicator.traceback_str = traceback.format_exc()
                print(self.communicator.traceback_str)
                self.communicator.error = e
            finally:
                self.communicator.is_done.set()  # ALWAYS set, even on failure

        try:
            self.create_waiting_window()
            self.root.update()

            t = threading.Thread(target=worker, name='Controller-TH', daemon=True)
            t.start()

            # wait for the thread to complete
            while not self.communicator.is_done.is_set():
                try:
                    text = self.communicator.message_queue.get_nowait()
                except Empty:
                    pass  # no message available yet
                else:
                    self.wait_label['text'] = text
                self.root.update()
                time.sleep(0.05)

            # drain the messages that lost the race with is_done
            while True:
                try:
                    text = self.communicator.message_queue.get_nowait()
                except Empty:
                    break
                self.wait_label['text'] = text

            self.waiting_window.destroy()

            # report success or failure to the user
            if self.communicator.error is not None:
                if isinstance(self.communicator.error, UnknownBankError):
                    messagebox.showerror("Error", "The CSV format is unexpected or is coming from an unsupported bank.")
                else:
                    messagebox.showerror("Error", "The operation failed: {}".format(self.communicator.error))
            else:
                messagebox.showinfo("Success", success_message)
        finally:
            self._worker_running = False

    def select_csv_btn_callback(self):
        """ Callback to set the source CSV file of the expenses. """
        filenames = filedialog.askopenfilenames(
            title="Select expenses file(s) in CSV format...",
            initialdir=os.path.join(os.path.expanduser('~'), "Downloads"),
        )
        if filenames and len(filenames) > 0:
            self.csv_filenames = list(filenames)
            self.csv_file_var.set(f"{len(self.csv_filenames)} file(s) selected")
        else:
            messagebox.showerror("Error", "No input file selected.")

    def import_csv_btn_callback(self):
        """ Callback to the import button. It imports the expenses in gspread. """

        # Check if there is a file in the input entry
        if self.csv_file_var.get() == '' or len(self.csv_filenames) == 0:
            messagebox.showerror("Error", "No expenses file selected.")
            return

        # Check all files have the csv extension
        for filename in self.csv_filenames:
            if not filename.endswith('csv'):
                messagebox.showerror("Error", f"The file '{filename}' is not a CSV format.")
                return

        # run the thread: import_expenses
        self._run_in_thread(import_expenses, (self.csv_filenames, self.settings), "Import completed.")

        # clear the input entry once processed
        self.csv_file_var.set('')

    def select_payslip_btn_callback(self):
        """ Callback to set the source PDF file of the payslip. """
        filename = filedialog.askopenfilename(title="Select payslip file in PDF format...")
        if os.path.isfile(filename):
            self.payslip_filename = filename
            self.payslip_filename_var.set(filename)
        else:
            messagebox.showerror("Error", "No input file selected.")

    def import_payslip_btn_callback(self):
        """ Callback to the import button. It imports the payslips in gspread. """
        if self.payslip_filename_var.get() != '':
            if self.payslip_filename.endswith('pdf'):
                import_payslips(self.payslip_filename, self.settings, self.communicator)

                # clear the entry once the file has been processed.
                self.payslip_filename_var.set('')

                messagebox.showinfo("Success", "Import completed.")
            else:
                messagebox.showerror("Error", "The file is not a PDF format.")
        else:
            messagebox.showerror("Error", "No expenses file selected.")
    
    def retrain_btn_callback(self):
        """ Callback to the button for retrain the classifier. """
        self._run_in_thread(re_train_classifier, (self.settings,), "Expenses classifier re-trained.")


# **************************
#            MAIN           
# **************************
if __name__ == '__main__':
    # Init for the GUI pop-ups
    root = tk.Tk()
    app = App(root)
    root.mainloop()
