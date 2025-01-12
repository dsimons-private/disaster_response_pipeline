import re
import nltk
import time
import pickle
import pandas
import pandas as pd
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine, text

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


def load_data(path_to_db: Path, table_name: str) -> (pandas.Series, pandas.DataFrame):
    """
    load data from db

    :param path_to_db: define path to database
    :param table_name: sql table name in db
    :return X, Y: pandas objects (messages and columns)
    """

    # load data from database
    engine = create_engine('sqlite:///{}'.format(path_to_db), echo=True)
    connection = engine.connect()

    # Execute SQL query to select table and create df
    df = pd.read_sql_query("SELECT * FROM {}".format(table_name), connection)

    # Close the database connection
    connection.close()

    # messages column
    X = df.message.values

    # columns
    Y = df[df.columns[4:]]  # skip the first three columns

    return X, Y


def tokenize(text):
    """
    NLP function for tokenizing and lemmatizing text (using nltk lib)

    :param text: input string / text
    :return: tokens
    """
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_ml_model() -> GridSearchCV:
    """
    Define a ML pipeline and parameters
    """

    # Create a Random Forest Classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(n_estimators=1)))
    ])

    # define parameters
    params = {
        'vect__max_df': [0.3, 0.4],
        'clf__estimator__n_estimators': [2, 3]
    }

    cv = GridSearchCV(estimator=pipeline, param_grid=params)

    return cv


def test_ml_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    for i, col in enumerate(y_test.columns):
        print(f'message category: {col}\n')
        print(classification_report(y_test.iloc[:, i], y_pred[:, i]))


def save_ml_model(model, filepath_model: Path) -> None:
    """
    Save ML model as .pkl (pickle file)

    :param model: trained model
    :param filepath_model: defined filepath to .pkl file
    :return: None
    """
    with open(filepath_model, 'wb') as file:
        pickle.dump(model, file)

    print("Model saved to file")


def main():
    """
    Main function for complete ML pipeline:

    Steps:
    1) load Data from SQL database
    2) build ML pipeline and create model
    3) split data into training and test fraction. Fit data to model
    4) test and evaluate model
    5) save model to file (.pkl)
    """

    print("Step 1: Load data from database")
    try:
        X, Y = load_data(path_to_db=Path(r"..\2_data\Messages.db"),
                         table_name='ETL_Messages')
    except FileNotFoundError as e:
        print("File not found: {}".format(e))

    print('Step 2: Build the ML model')
    model = build_ml_model()

    print("Step 3: Split data and train ML model")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    start_fit_pipeline = time.time()

    model.fit(X_train, y_train)

    end_fit_pipeline = time.time()
    measured_time = end_fit_pipeline - start_fit_pipeline
    print(f"time elapsed for training {measured_time}s")

    print("Step 4: Test ML model")
    test_ml_model(model, X_test, y_test)

    print("Step 5 : Save to ML model file")
    save_ml_model(model=model,
                  filepath_model=Path(r"model.pkl"))


if __name__ == '__main__':
    main()




