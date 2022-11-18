import csv
import re
import pandas as pd

""" Library for the managing of the input files in CSV format. """

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


def check_header(csv_lines, settings):
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
        # print("N26")
        return True, "N26"
    else:
        i = 1
        while i < 15:
            ING_HEADER_LINE = settings["ING_HEADER_LINE"]
            if csv_lines[i] == list(ING_HEADER_LINE.keys()):
                # print("ING")
                return True, "ING"
            else:
                i = i + 1
        print("Not recognized")
        return False, None


def ingest_N26_csv(csv_path, settings):
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
    # "Date", "Payee", "Reference", "Amount"
    for line in csv_lines:
        generic_lines.append([line[N26_HEADER_LINE["Date"]], # Date
                              _preprocess(line[N26_HEADER_LINE["Payee"]]), # Payee
                              _preprocess(line[N26_HEADER_LINE["Payment reference"]]), # Reference
                              line[N26_HEADER_LINE["Amount (EUR)"]], # Amount
                            ])
    
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
    Raises:
        ValueError: In case the header of the input CSV is not respecting the one for ING ecoded in the settings. 
    """
    # read the lines
    with open(csv_path) as csv_file:
        csv_lines = list(csv.reader(csv_file, delimiter=';'))

    # Check if the header is compliant with ING.
    ING_HEADER_LINE = settings['ING_HEADER_LINE']
    if csv_lines[13][0] != 'Buchung':
        raise ValueError('Header not respecting the ING header')
    else:
        csv_lines = csv_lines[14:] # remove the first line that contains the header

    generic_lines = []
    # construct the generic lines as per GENERIC_HEADER
    # "Date", "Payee", "Reference", "Category", "Amount"
    # 25.10.2022 ;  25.10.2022;    TELESPAZIO GERMANY GMBH;   Gehalt/Rente;  GEHALT 10/22;  4.922,75;EUR;3.185,53;EUR
    for line in csv_lines:
        generic_lines.append([line[ING_HEADER_LINE['Valuta']].replace('.', '-'), # Date
                              _preprocess(line[ING_HEADER_LINE['Auftraggeber/Empfanger']]), # Payee
                              _preprocess(line[ING_HEADER_LINE['Verwendungszweck']]), # Reference
                              line[ING_HEADER_LINE['Betrag']], # Amount
                            ])
    
    # finally create the pandas dataframe
    c = list(settings['GENERIC_HEADER'].keys())
    generic_df = pd.DataFrame(data=generic_lines, columns=c)
    generic_df = generic_df.astype(settings["GENERIC_HEADER_DTYPE"])
    generic_df['payee'].fillna('', inplace=True)
    generic_df['reference'].fillna('', inplace=True)
    return generic_df

