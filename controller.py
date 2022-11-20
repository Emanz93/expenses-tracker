import gspread
import joblib
import pandas as pd
import re

# scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# personal libraries
from csv_lib import ingest_N26_csv, ingest_ING_csv, _preprocess, _get_month_int

def google_services(settings):
    """ Connect with the google services. """
    sa = gspread.service_account(filename=settings['G_SERVICE_KEYFILE'])
    # open the google sheet and select the correct worksheet
    sh = sa.open(settings['G_SHEET_NAME'])
    print("Connected with Google Sheet: {}".format(settings['G_SHEET_NAME']))
    return sh


def classify(df, settings):
    """ Perform the classification task. It uses the classified model saved in a pickle file along with the 
    hot encoder used for the target label unary transformation. It adds the 'category' column to the input 
    data frame and returns it.
    Parameter:
        df: pd.DataFrame. Input dataframe with the category feature empty.
    Returns:
        df: pd.DataFrame. Output dataframe with the category feature populated.
    """
    # load the required models
    with open(settings['MODEL_CLASSIFIER_PATH'], 'rb') as classifierfile:
        classifier = joblib.load(classifierfile)
    with open(settings['MODEL_ENCODER_PATH'], 'rb') as encoderfile:
        encoder = joblib.load(encoderfile)
    with open(settings['MODEL_VOCABOULARY_PATH'], 'rb') as vacfile:
        vocaboulary = joblib.load(vacfile)

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


def train(transactions, settings):
    """ Perform the train of the classifier.
    Parameters:
        transactions: pd.DataFrame. Input transactions with the classified
    """
    # preprocessing
    for feature in list(transactions.columns):
        if feature not in ['payee', 'reference', 'category']:
            # drop the useless features
            transactions = transactions.drop(feature, axis=1)
    
    transactions['payee'] = transactions['payee'].fillna('')
    transactions['payee'] = transactions['payee'].apply(_preprocess)

    transactions['reference'] = transactions['reference'].fillna('')
    transactions['reference'] = transactions['reference'].apply(_preprocess)

    # separate the category from the rest of the dataset
    tr_x = transactions.drop('category', axis=1) # transactions inputs
    tr_y = transactions['category'].copy() # transactions category to be guessed

    # remove the labels and append them as columns to the dataframe
    encoder = LabelBinarizer()
    tr_cat_1hot = encoder.fit_transform(tr_y)
    tr_cat_prepared = pd.DataFrame(data=tr_cat_1hot, columns=encoder.classes_)

    # convert the words in bag of words (CountVectorizer)
    #vectorizer = CountVectorizer(max_features=1500, min_df=1, max_df=0.7)

    # convert the words in TFIDF (TfidfVectorizer)
    vectorizer = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7)
    # X = tfidfconverter.fit_transform(list(tr_num['Payee'])).toarray()

    # vectorize the textual fields. The original ones need to be dropped
    vocabulary = dict()
    for feature in ['payee', 'reference']:
        vectorizer = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7)
        X = vectorizer.fit_transform(tr_x[feature]).toarray()
        vocabulary[feature] = vectorizer.vocabulary_
        print(feature)
        x_in = pd.DataFrame(X)
        x_in.columns = x_in.columns.astype(str)
        tr_x = tr_x.drop(feature, axis=1)
        tr_x = pd.concat([tr_x, x_in], axis=1)
    
    # random_state allows to set the seed number
    X_train, X_test, y_train, y_test = train_test_split(tr_x, tr_cat_prepared, test_size=0.2)

    # train a random forest classifier
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=2500)
    classifier.fit(X_train, y_train)

    # predict the test
    y_pred = classifier.predict(X_test)

    # evaluate the model
    # print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred, target_names=encoder.classes_))
    print(accuracy_score(y_test, y_pred))

    # save the model of the random forest:
    with open(settings['MODEL_CLASSIFIER_PATH'], 'wb') as picklefile:
        joblib.dump(classifier, picklefile)

    # save the vocaboulary
    with open(settings['MODEL_VOCABOULARY_PATH'], 'wb') as picklefile:
        joblib.dump(vocabulary, picklefile)

    # save the vocaboulary
    with open(settings['MODEL_ENCODER_PATH'], 'wb') as picklefile:
        joblib.dump(encoder, picklefile)


def get_all_worksheets(sh, only_years=False):
    """ Return a dictionary of worksheets from the gsheet.
    Parameter:
        sh: Gspread sheet.
    Returns:
        worksheets: dictionary. <key: worksheet name (string). Value: pointer to the worksheet>
    """
    worksheets = dict()
    if only_years: # if only the years are in
        for x in sh.worksheets():
            if re.search('^(20[0-9]{2})$', x) != None:
                worksheets[x.title] = x
    else:
        for x in sh.worksheets():
            worksheets[x.title] = x
    return worksheets


def import_expences(in_cvs_path, settings):
    """ Import the CSV in gspread.
    Parameters:
        in_csv_path: String. Path of the input csv file.
        settings: Dictionary.
    """
    df_in = ingest_N26_csv(in_cvs_path, settings)

    #print("Input dataframe len={}".format(len(df_in)))

    # classify the input transactions in categories
    df_class = classify(df_in, settings)

    # add the extra fields required for the visualization in gspread
    df_class['month'] = df_class['month'].apply(_get_month_int) # add a column with the month in integer format

    # get the unique values of the year from the input csv. In this way, I download only the ones that are needed
    unique_years_in = set(x[:4] for x in df_class['date'].tolist()) # set of strings

    # open the google sheet
    sh = google_services(settings)

    # create a dict that contains the worksheets that are currently in gspread. 
    # key: wsh title, value: pointer to the wsh
    worksheets = get_all_worksheets(sh)

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


def re_train_classifier(settings):
    # open the google sheet
    sh = google_services(settings)

    # create a dict that contains the worksheets that are currently in gspread. 
    # key: wsh title, value: pointer to the wsh
    years_worksheets = get_all_worksheets(sh, only_years=True)
    print('re_train_classifier')

    df = pd.DataFrame()
    for year in sorted(years_worksheets.keys()):
        # donwload from gspread all the records in the 'year' sheet in a pandas frame
        wsh = years_worksheets[year]
        if df.empty:
            df = pd.DataFrame(wsh.get_all_records())
        else:
            df = pd.merge(df, pd.DataFrame(wsh.get_all_records()), on=list(df.columns), how="outer")
    
    train(df, settings)

def import_payslips(in_cvs_path, settings):
    print('import_payslips')