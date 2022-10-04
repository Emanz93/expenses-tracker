import gspread
import json
import csv
from datetime import datetime
import pandas as pd

""" ExpensesTracker """

def read_json(json_path):   
    """ Read a json file.
    Parameters:
        json_path: String. Path of the json file.
    Returns:
        d: dict. Content of the file.
    """
    with open(json_path, 'r') as f:
        d = json.load(f)
    return d


def check_header(csv_lines):
    """ Check if the header line is the expected one and understand also which bank it belongs.
    Parameters:
        cvs_lines: Lists of lists of strings. Input file.
    Returns:
        check_header_ok: boolean. True if the header is as expected. False if not or the bank is not recognized.
        bank_type: String. N26, ING or None if not recognized
    """
    # Check if the header is compliant with N26.
    N26_HEADER_LINE = settings["N26_HEADER_LINE"]
    if csv_lines[0] == list(N26_HEADER_LINE.keys()):
        print("N26")
        return True, "N26"
    else:
        i = 1
        while i < 15:
            ING_HEADER_LINE = settings["ING_HEADER_LINE"]
            if csv_lines[i] == list(ING_HEADER_LINE.keys()):
                print("ING")
                return True, "ING"
            else:
                i = i + 1
        print("Not recognized")
        return False, None


def ingest_N26_csv(csv_path):
    """ Ingest, check and pre-process a CSV file taken from N26.
    Parameters:
        csv_path: string. Path of the CSV file to ingest.
    Returns:
        generic_df: pd.DataFrame.
    Raises:
        ValueError: In case the header of the input CSV is not respecting the one for N26 ecoded in the settings. 
    """
    # read the lines
    with open(csv_path) as csv_file:
        csv_lines = list(csv.reader(csv_file, delimiter=',', quotechar='"'))
    
    # Check if the header is compliant with N26.
    N26_HEADER_LINE = settings["N26_HEADER_LINE"]
    if csv_lines[0] != list(N26_HEADER_LINE.keys()):
        raise ValueError("Header not respecting the N26 header")
    else:
        csv_lines = csv_lines[1:] # remove the first line that contains the header

    generic_lines = []
    # construct the generic lines as per GENERIC_HEADER
    # "Date", "Payee", "Transaction type", "Reference", "Category", "Amount"
    for line in csv_lines:
        generic_lines.append([line[N26_HEADER_LINE["Date"]],
                              line[N26_HEADER_LINE["Payee"]],
                              line[N26_HEADER_LINE["Payment reference"]],
                              line[N26_HEADER_LINE["Category"]],
                              line[N26_HEADER_LINE["Amount (EUR)"]],
                              ])
    
    # finally create the pandas dataframe
    c = list(settings["GENERIC_HEADER"].keys())
    generic_df = pd.DataFrame(data=generic_lines, columns=c)
    generic_df = generic_df.astype(settings["GENERIC_HEADER_DTYPE"])
    return generic_df


def ingest_ING_csv(csv_path):
    """ Ingest, check and pre-process a CSV file taken from ING.
    Parameters:
        csv_path: string. Path of the CSV file to ingest.
    Returns:
        generic_lines: List of list of strings respecting the generic header.
    Raises:
        ValueError: In case the header of the input CSV is not respecting the one for ING ecoded in the settings. 
    """
    # read the lines
    with open(csv_path) as csv_file:
        csv_lines = list(csv.reader(csv_file, delimiter=';'))

    # Check if the header is compliant with ING.
    ING_HEADER_LINE = settings["ING_HEADER_LINE"]
    if csv_lines[13][0] != "Buchung":
        raise ValueError("Header not respecting the ING header")
    else:
        csv_lines = csv_lines[14:] # remove the first line that contains the header

    generic_lines = []
    # construct the generic lines as per GENERIC_HEADER
    # "Date", "Payee", "Reference", "Category", "Amount"
    for line in csv_lines:
        generic_lines.append([line[ING_HEADER_LINE["Valuta"]],
                              line[ING_HEADER_LINE["Auftraggeber/Empfanger"]],
                              line[ING_HEADER_LINE["Verwendungszweck"]],
                              line[ING_HEADER_LINE["Buchungstext"]],
                              line[ING_HEADER_LINE["Saldo"]],
                              ])
    
    # finally create the pandas dataframe
    c = list(settings["GENERIC_HEADER"].keys())
    generic_df = pd.DataFrame(data=generic_lines, columns=c)
    return generic_df


def google_services():
    """ Connect with the google services. """
    sa = gspread.service_account(filename=settings['G_SERVICE_KEYFILE'])
    # open the google sheet and select the correct worksheet
    sh = sa.open(settings['G_SHEET_NAME'])
    print("Connected with Google Sheet: {}".format(settings['G_SHEET_NAME']))
    return sh

# **************************
#            MAIN           
# **************************
# read the settings file
settings = read_json('settings.json')

if __name__ == '__main__':
    # input cvs file
    in_cvs_path = "infiles/n26-csv-transactions.csv"
    df_in = ingest_N26_csv(in_cvs_path)

    # in_cvs_path = "infiles/Umsatzanzeige_ING.csv"
    # df_in = ingest_ING_csv(in_cvs_path)

    print("Input dataframe:")
    print(df_in.head(15))

    # get the unique values of the year from the input csv. In this way, I download only the ones that are needed
    unique_years_in = set(x[:4] for x in df_in["Date"].tolist()) # set of strings

    # open the google sheet
    sh = google_services()

    # create a dict that contains the worksheets. 
    # key: wsh title, value: pointer to the wsh
    worksheets = dict()
    for x in sh.worksheets():
        worksheets[x.title] = x

    # loop each year
    for year in sorted(unique_years_in):
        print('year={}'.format(year))
        wsh = None
        if year in worksheets.keys():
            # there is a worksheet having the same year.
            # donwload from gspread all the records in the 'year' sheet in a pandas frame
            wsh = worksheets[year]
            df_gd = pd.DataFrame(wsh.get_all_records())

            # merge the downloaded pandas frame with the input csv pandas of the records of the same year. 
            df_out = pd.merge(df_gd, df_in[df_in['Date'].str.contains(year)], on=list(settings["GENERIC_HEADER"].keys()), how="outer")
        else:
            # there are no worksheets having 'year' name, create a new one and fill the values from the input csv
            wsh = sh.add_worksheet(title=year, rows=100, cols=10)
            df_out = df_in[df_in['Date'].str.contains(year)]
            worksheets[year] = wsh
        
        # finally update the output df in the target year sheet
        wsh.update([df_out.columns.values.tolist()] + df_out.values.tolist())

    