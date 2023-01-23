import csv
import re
import pandas as pd
from datetime import datetime

""" Library for the managing of the input files in CSV format. """

class UnkownBankError(Exception):
    """ Error raised when a CSV from a unkown bank is injested. """
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def _preprocess(s):
    """ Preprocess the input string. Remove the unwanted chars.
    Parameter:
        s: String.
    Returns:
        s: String.
    """
    s = re.sub(r'\W', ' ', s) # remove special chars
    s = s.strip() # remove leading and trailing whitespaces
    s = re.sub(r'\s+', ' ', s, flags=re.I) # remove multiple whitespaces
    s = s.lower() # to lowercase
    return s


def _preprocess_date(date_str):
    """ Convert the date string from %d.%m.%y to %y-%m-%d.
    Parameter:
        date_str: String. Date in %d.%m.%y format.
    Returns:
        String. Date in %y-%m-%d format.
    """
    d = datetime.strptime(date_str, '%d.%m.%Y')
    return d.strftime('%Y-%m-%d')


def _get_month_int(d):
    """ Takes in input a string containing a date and returns the month in integer.
    Parameter:
        d: String. Date
    Returns:
        m: Integer. Month
    """
    dt = datetime.strptime(d, '%Y-%m-%d')
    return dt.month


def check_which_bank(csv_path, settings):
    """ Check if the header line is the expected one and understand also which bank it belongs.
    Parameters:
        csv_path: String. path of the input file.
    Returns:
        bank_type: String. N26, ING or None if not recognized
    Raises:
        UnkownBankError: the CSV is unrecognized.
    """
    # read the lines
    with open(csv_path) as csv_file:
        csv_lines = list(csv.reader(csv_file, delimiter=',', quotechar='"'))
    
    # Check if the header is compliant with N26.
    N26_HEADER_LINE = settings['N26_HEADER_LINE']
    ING_HEADER_LINE = settings['ING_HEADER_LINE']
    if csv_lines[0] == list(N26_HEADER_LINE.keys()):
        return 'N26'
    elif csv_lines[13][0].startswith('Buchung'): # Check if the header is compliant with ING.
        return 'ING'
    else:
        raise UnkownBankError('Header not respecting neither N26 nor the ING header')


def ingest_N26_csv(csv_path, settings):
    """ Ingest, check and pre-process a CSV file taken from N26.
    Parameters:
        csv_path: string. Path of the CSV file to ingest.
    Returns:
        generic_df: pd.DataFrame.
    """
    # read the lines
    with open(csv_path) as csv_file:
        csv_lines = list(csv.reader(csv_file, delimiter=',', quotechar='"'))
        
    # remove the first line that contains the header
    csv_lines = csv_lines[1:]

    generic_lines = []
    # construct the generic lines as per GENERIC_HEADER
    # "Date", "Payee", "Reference", "Amount"
    N26_HEADER_LINE = settings["N26_HEADER_LINE"]
    for line in csv_lines:
        date = line[N26_HEADER_LINE["Date"]]
        payee = _preprocess(line[N26_HEADER_LINE["Payee"]])
        reference = _preprocess(line[N26_HEADER_LINE["Payment reference"]])
        amount = line[N26_HEADER_LINE["Amount (EUR)"]]
        generic_lines.append([hex(hash(date + payee + reference + amount)),date,payee,reference,amount])
    
    # finally create the pandas dataframe
    c = list(settings["GENERIC_HEADER"].keys())
    generic_df = pd.DataFrame(data=generic_lines, columns=c)
    generic_df = generic_df.astype(settings["GENERIC_HEADER_DTYPE"])
    generic_df['payee'].fillna('', inplace=True)
    generic_df['reference'].fillna('', inplace=True)
    return generic_df


def ingest_ING_csv(csv_path, settings):
    """ Ingest, check and pre-process a CSV file taken from ING.
    Parameters:
        csv_path: string. Path of the CSV file to ingest.
    Returns:
        generic_lines: List of list of strings respecting the generic header. 
    """
    # read the lines
    with open(csv_path) as csv_file:
        csv_lines = list(csv.reader(csv_file, delimiter=';'))

    csv_lines = csv_lines[14:] # remove the first line that contains the header

    generic_lines = []
    # construct the generic lines as per GENERIC_HEADER
    # "Date", "Payee", "Reference", "Category", "Amount"
    # 25.10.2022 ;  25.10.2022;    TELESPAZIO GERMANY GMBH;   Gehalt/Rente;  GEHALT 10/22;  4.922,75;EUR;3.185,53;EUR
    ING_HEADER_LINE = settings["ING_HEADER_LINE"]
    for line in csv_lines:
        date = _preprocess_date(line[ING_HEADER_LINE['Valuta']])
        payee = _preprocess(line[ING_HEADER_LINE['Auftraggeber/Empfanger']])
        reference = _preprocess(line[ING_HEADER_LINE['Verwendungszweck']])
        amount = line[ING_HEADER_LINE['Betrag']]
        generic_lines.append([hex(hash(date + payee + reference + amount)),date, payee, reference, amount])
    
    # finally create the pandas dataframe
    c = list(settings['GENERIC_HEADER'].keys())
    generic_df = pd.DataFrame(data=generic_lines, columns=c)
    generic_df['amount'] = generic_df['amount'].str.replace('.', '', regex=False)
    generic_df['amount'] = generic_df['amount'].str.replace(',', '.', regex=False)
    generic_df = generic_df.astype(settings["GENERIC_HEADER_DTYPE"])
    generic_df['payee'].fillna('', inplace=True)
    generic_df['reference'].fillna('', inplace=True)
    return generic_df
