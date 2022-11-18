import gspread
import joblib
import json
from csv_lib import _preprocess
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from csv_lib import ingest_N26_csv, ingest_ING_csv, _preprocess

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


def google_services():
    """ Connect with the google services. """
    sa = gspread.service_account(filename=settings['G_SERVICE_KEYFILE'])
    # open the google sheet and select the correct worksheet
    sh = sa.open(settings['G_SHEET_NAME'])
    print("Connected with Google Sheet: {}".format(settings['G_SHEET_NAME']))
    return sh


def classify(df):
    """ Perform the classification task. It uses the classified model saved in a pickle file along with the 
    hot encoder used for the target label unary transformation. It adds the 'category' column to the input 
    data frame and returns it.
    Parameter:
        df: pd.DataFrame. Input dataframe with the category feature empty.
    Returns:
        df: pd.DataFrame. Output dataframe with the category feature populated.
    """
    # load the required models
    with open('models/text_classifier', 'rb') as trained_model:
        classifier = joblib.load(trained_model)
    with open('models/encoder', 'rb') as picklefile:
        encoder = joblib.load(picklefile)
    with open('models/vocaboulary', 'rb') as picklefile:
        vocaboulary = joblib.load(picklefile)

    # preprocessing
    transactions = df.copy()
    transactions = transactions.drop(['date'], axis=1)
    transactions['payee'] = transactions['payee'].apply(_preprocess)
    transactions['reference'] = transactions['reference'].apply(_preprocess)

    # Vectorizer
    for feature in ['payee', 'reference']:
        vectorizer = TfidfVectorizer(vocabulary=vocaboulary[feature])
        X = vectorizer.fit_transform(transactions[feature]).toarray()
        x_in = pd.DataFrame(X)
        x_in.columns = x_in.columns.astype(str)
        print(feature)
        transactions = transactions.drop(feature, axis=1)
        transactions = pd.concat([transactions, x_in], axis=1)
    
    # perform the prediction
    y_pred = classifier.predict(transactions)
    
    # use the encoder to revert the encoded transformation, i.e. from uninay representation to labelized.
    df['category'] = encoder.inverse_transform(y_pred)
    return df


# **************************
#            MAIN           
# **************************
# read the settings file
settings = read_json('settings.json')

if __name__ == '__main__':
    # input cvs file
    in_cvs_path = "infiles/n26-csv-transactions.csv"
    df_in = ingest_N26_csv(in_cvs_path, settings)

    # in_cvs_path = "infiles/Umsatzanzeige_ING2.csv"
    # df_in = ingest_ING_csv(in_cvs_path)

    print("Input dataframe len={}".format(len(df_in)))

    # classify the input transactions in categories
    df_class = classify(df_in)

    # get the unique values of the year from the input csv. In this way, I download only the ones that are needed
    unique_years_in = set(x[:4] for x in df_class["date"].tolist()) # set of strings

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
            df_out = pd.merge(df_gd, df_class[df_class['date'].str.contains(year)], on=list(df_class.columns), how="outer")
        else:
            # there are no worksheets having 'year' name, create a new one and fill the values from the input csv
            wsh = sh.add_worksheet(title=year, rows=1000, cols=100)
            df_out = df_class[df_class['date'].str.contains(year)]
            worksheets[year] = wsh
        
        # finally update the output df in the target year sheet
        wsh.update([df_out.columns.values.tolist()] + df_out.values.tolist())

    