import pandas as pd
import pickle
import joblib
from csv_lib import _preprocess

# scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# read the file
transactions = pd.read_csv('./infiles/n26-csv-transactions_cat.csv')

# preprocessing
transactions = transactions.drop(['Date', 'Amount (Foreign Currency)', 'Type Foreign Currency', 'Exchange Rate', 'Account number', 'Transaction type'], axis=1)
transactions.rename(columns = {'Date':'date', 'Payee':'payee', 'Payment reference':'reference', 'Amount (EUR)':'amount', 'Category':'category'}, inplace = True)
transactions['payee'].fillna('', inplace=True)
transactions['reference'].fillna('', inplace=True)
transactions['payee'] = transactions['payee'].apply(_preprocess)
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
    

tr_cat_prepared.info()
# random_state allows to set the seed number
X_train, X_test, y_train, y_test = train_test_split(tr_x, tr_cat_prepared, test_size=0.2, random_state=42)

# train a random forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=2500, random_state=0)
classifier.fit(X_train, y_train)

# predict the test
y_pred = classifier.predict(X_test)

# evaluate the model
# print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred, target_names=encoder.classes_))
print(accuracy_score(y_test, y_pred))

# reverse the encoding of the classification feature to get the actual classification values
# encoder.inverse_transform(y_pred)

# save the model of the random forest:
with open('text_classifier', 'wb') as picklefile:
    joblib.dump(classifier, picklefile)

with open('vocaboulary', 'wb') as picklefile:
    joblib.dump(vocabulary, picklefile)