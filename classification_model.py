from cgi import test
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocessing import preprocess_text
from tqdm import tqdm
import pandas as pd
import joblib

tqdm.pandas()

def load_data(file_path):
    # Load Pickled data
    df = pd.read_pickle(file_path)

    df['preprocessed_text'] = df['text'].progress_apply(lambda x: ' '.join(map(str, preprocess_text(x)))) # type: ignore

    df.to_pickle('data/essays_preprocessed.pkl')

    return df

def extract_features(test, label):
    # Feature extraction implementation

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(test, label, test_size=0.2, random_state=42)

    # Vectorize the text data
    vectorizer = TfidfVectorizer()

    # save the vectorizer
    joblib.dump(vectorizer, 'models/vectorizer.pkl')

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def train_logistic_model(X_train, y_train):
    # Model training implementation
    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model


def predict_labels(model, X_test):
    # Prediction implementation
    return model.predict(X_test)


def evaluate_model(y_test, y_pred):
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return accuracy, precision, recall, f1

def pipeline():
    df = load_data('data/essays.pkl')
    
    trait = ['extroversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']

    results = {}

    for t in trait:
        X_train, X_test, y_train, y_test = extract_features(df['preprocessed_text'], df[t])

        # Train the model
        model = train_logistic_model(X_train, y_train)

        # Predict the labels
        y_pred = predict_labels(model, X_test)

        # Evaluate the model
        accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)
        results[t] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    # Save the model
    joblib.dump(model, f'models/{t}_model.pkl')

if __name__ == "__main__":
    # Example usage or testing

    test_sentence = "I am a very happy person"

    # Load the model
    model = joblib.load('models/extroversion_model.pkl')

    # Load the vectorizer
    vectorizer = joblib.load('models/vectorizer.pkl')

    # Preprocess the text
    test_sentence = preprocess_text(test_sentence) 
    test_sentence = ' '.join(map(str, test_sentence)) # type: ignore

    # Vectorize the text
    test_sentence = vectorizer.transform(test_sentence)

    # Predict the label
    prediction = model.predict(test_sentence)

    print(prediction)
    

    

