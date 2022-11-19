import gspread
import joblib
import pandas as pd

# scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

# personal libraries
from csv_lib import ingest_N26_csv, ingest_ING_csv, _preprocess, _get_month_int

def google_services(settings):
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
    with open('res/models/text_classifier', 'rb') as trained_model:
        classifier = joblib.load(trained_model)
    with open('res/models/encoder', 'rb') as picklefile:
        encoder = joblib.load(picklefile)
    with open('res/models/vocaboulary', 'rb') as picklefile:
        vocaboulary = joblib.load(picklefile)

    # preprocessing
    transactions = df.copy()
    transactions = transactions.drop(['date'], axis=1)
    transactions['payee'] = transactions['payee'].apply(_preprocess)
    transactions['reference'] = transactions['reference'].apply(_preprocess)

    # Vectorizer
    for feature in ['payee', 'reference']:
        vectorizer = TfidfVectorizer(vocabulary=vocaboulary[feature])
        X = vectorizer.fit_transform(transactions[feature])
        x_in = pd.DataFrame(X.toarray())
        x_in.columns = x_in.columns.astype(str)
        print(feature)
        transactions = transactions.drop(feature, axis=1)
        transactions = pd.concat([transactions, x_in], axis=1)
    
    # perform the prediction
    y_pred = classifier.predict(transactions)
    
    # use the encoder to revert the encoded transformation, i.e. from uninay representation to labelized.
    df['category'] = encoder.inverse_transform(y_pred)
    return df


def import_expences(in_cvs_path, settings):
    """ Import the CSV in gspread.
    Parameters:
        in_csv_path: String. Path of the input csv file.
        settings: Dictionary.
    """
    df_in = ingest_N26_csv(in_cvs_path, settings)

    #print("Input dataframe len={}".format(len(df_in)))

    # classify the input transactions in categories
    df_class = classify(df_in)

    # add the extra fields required for the visualization in gspread
    df_class['month'] = df_class['month'].apply(_get_month_int) # add a column with the month in integer format

    # get the unique values of the year from the input csv. In this way, I download only the ones that are needed
    unique_years_in = set(x[:4] for x in df_class['date'].tolist()) # set of strings

    # open the google sheet
    sh = google_services(settings)

    # create a dict that contains the worksheets that are currently in gspread. 
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


def re_train_classifier():
    print('re_train_classifier')

def import_payslips(in_cvs_path, settings):
    print('import_payslips')