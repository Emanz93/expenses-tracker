import os
import re
import pandas as pd
from datetime import datetime

""" Library for the managing of the input files in CSV format. """

class UnknownBankError(Exception):
    """ Error raised when a CSV from an unknown bank is injected. """
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
    """ Convert the date string from %d.%m.%Y to %Y-%m-%d.
    Parameter:
        date_str: String. Date in %d.%m.%Y format.
    Returns:
        String. Date in %Y-%m-%d format.
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


def group_payee_on_same_day(df, payee_names):
    """ Group the expenses of the same payee by day, e.g. sodexo service gmbh.
    This preprocessing removes the information from the reference column.
    Parameters:
        df: DataFrame.
        payee_names: list of strings. payee_names to be grouped. e.g. sodexo service gmbh
    Returns:
        df: DataFrame.
    """
    for raw_payee in payee_names:
        payee = _preprocess(raw_payee)  # Apply the same normalization the data went through

        # Group the transactions by date
        df_sodexo = df[df['payee'] == payee].groupby("date", as_index=False).agg({"payee": lambda x:payee, "reference": lambda x: "", "amount": "sum"})
        
        # drop from the input dataframe the original transactions
        df = df.drop(df[df['payee'] == payee].index)

        # concatenate the two dataframes
        df = pd.concat([df, df_sodexo], axis=0)

    df = df.reset_index(drop=True) # reset the index
    return df


def _generic_df_preprocessing(df, settings):
    """
    Perform pre-processing operations common for all the CSV's banks.
    Parameters:
        df: DataFrame. Input DataFrame. It contains already the bank-specific changes needed.
        settings: dict.

    Returns:
        df: DataFrame. Pre-processed DataFrame.

    """
    required_columns = list(settings['GENERIC_HEADER'].keys())

    # Remove all the unnecessary columns and set them in the correct order
    df = df[required_columns]

    # Apply type conversion
    df = df.astype(settings["GENERIC_HEADER_DTYPE"])
    df['amount'] = df['amount'].round(2)

    # Fill NaN in string columns
    df['payee'] = df['payee'].fillna('')
    df['reference'] = df['reference'].fillna('')

    # round the float numbers to .2
    df['amount'] = df['amount'].round(2)

    # Apply preprocessing
    df['payee'] = df['payee'].apply(_preprocess)
    df['reference'] = df['reference'].apply(_preprocess)

    return df

def validate_dataframe(df, settings):
    """ Validate that the dataframe has required columns and correct types.
    Parameters:
        df: pd.DataFrame. DataFrame to validate.
        settings: dict.
    Returns:
        df: pd.DataFrame. Validated dataframe.
    Raises:
        ValueError: if validation fails.
    """
    required_columns = list(settings['GENERIC_HEADER'].keys())

    # Check all required columns exist
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f'Missing required columns: {missing_cols}')

    return df


def check_which_bank(csv_path, settings):
    """ Given an input file, detect which bank the csv is coming from.
    Parameters:
        csv_path: String. path of the input file.
        settings: dict.
    Returns:
        bank_type: String. N26, ING or None if not recognized
    Raises:
        UnknownBankError: the CSV is unrecognized.
    """
    # Try with N26 first: comma-delimited, utf format. Header is in the first line.
    try:
        with open(csv_path, encoding='utf-8-sig', errors='replace') as csv_file:
            first_line = csv_file.readline().strip()

        if not first_line:
            raise UnknownBankError('The CSV file is empty')

        # Check if it matches N26 header
        if first_line.replace('"','') == ','.join(settings['N26_HEADER_LINE']):
            return 'N26'

    except Exception:
        pass

    # Try ING format (semicolon-delimited, cp1252
    try:
        with open(csv_path, encoding='cp1252', errors='replace') as csv_file:
            lines = csv_file.readlines()

            if not lines:
                raise UnknownBankError('The CSV file is empty')

            # For ING, the header is in line #14
            if lines[13].strip() == ';'.join(settings['ING_HEADER_LINE']):
                return 'ING'

    except Exception:
        pass

    raise UnknownBankError('Header not respecting neither N26 nor the ING header')


def _ingest_N26_csv(csv_path, settings):
    """ Ingest, check and pre-process a CSV file taken from N26.
    Parameters:
        csv_path: string. Path of the CSV file to ingest.
        settings: dict.
    Returns:
        generic_df: pd.DataFrame, with generic columns.
    """

    # Read CSV with pandas
    df = pd.read_csv(
        csv_path,
        encoding='utf-8-sig',
        on_bad_lines='warn'  # handle malformed rows
    )

    # Map to generic columns: here we need to be consistent with the one specified in the settings.json, GENERIC_HEADER key.
    column_mapping = {
        'Booking Date': 'date',
        'Partner Name': 'payee',
        'Payment Reference': 'reference',
        'Amount (EUR)': 'amount'
    }
    df = df.rename(columns=column_mapping)

    # perform generic preprocessing operations on the data frame
    df = _generic_df_preprocessing(df, settings)

    return df


def _ingest_ING_csv(csv_path, settings):
    """ Ingest, check and pre-process a CSV file taken from ING.
    Parameters:
        csv_path: string. Path of the CSV file to ingest.
        settings: dict.
    Returns:
        generic_lines: List of list of strings respecting the generic header. 
    """
    # Read CSV with pandas
    df = pd.read_csv(
        csv_path,
        encoding='cp1252',
        delimiter=';',
        skiprows=13,  # Skip header lines
        on_bad_lines='warn'  # handle malformed rows
    )

    # Map to generic columns: here we need to be consistent with the one specified in the settings.json, GENERIC_HEADER key.
    column_mapping = {
        'Buchung': 'date',
        'Auftraggeber/Empfänger': 'payee',
        'Verwendungszweck': 'reference',
        'Betrag': 'amount'
    }
    df = df.rename(columns=column_mapping)

    # Convert date format of ING to the normal one
    df['date'] = df['date'].apply(_preprocess_date)

    # Convert German number format (1.234,56 -> 1234.56)
    df['amount'] = (
        df['amount'].astype(str)
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
        .astype(settings["GENERIC_HEADER_DTYPE"]['amount'])
    )

    # perform generic preprocessing operations on the data frame
    df = _generic_df_preprocessing(df, settings)

    return df


def ingest_csv(csv_path, settings):
    """ Unified CSV ingestion with automatic bank detection.
    Parameters:
        csv_path: String. Path of the input csv file.
        settings: Dictionary.
    Returns:
        df: pd.DataFrame. DataFrame with generic columns (date, payee, reference, amount).
        bank: String. Name of the bank that was ingested.
    Raises:
        UnknownBankError: if bank cannot be detected.
        ValueError: if CSV structure is invalid.
    """
    # check if the file exists
    if not os.path.isfile(csv_path):
        print('Input CSV is not a file')
        raise ValueError('Input CSV is not a file')

    # Detect bank
    bank = check_which_bank(csv_path, settings)

    # Dispatch to bank-specific ingestion
    ingestion_functions = {
        'N26': _ingest_N26_csv,
        'ING': _ingest_ING_csv,
        # Add new banks here: 'Revolut': _ingest_Revolut_csv,
    }

    if bank not in ingestion_functions:
        raise UnknownBankError(f'Unsupported bank: {bank}')

    # Ingest using the appropriate function
    df = ingestion_functions[bank](csv_path, settings)
    print(f'Ingested {bank} CSV')

    if df.empty:
        print('Input csv is empty')
        raise ValueError('Input csv is empty')

    # Group certain payees by date so that the number of rows is reduced
    df = group_payee_on_same_day(df, settings["FILTERED_PAYEES"])

    # Validate and return
    df = validate_dataframe(df, settings)

    return df, bank
