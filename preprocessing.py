import nltk 
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import opinion_lexicon
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
# lemmitization
from nltk.stem import WordNetLemmatizer

import string
import pandas as pd
from tqdm import tqdm

nltk.download('opinion_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def write_to_file(filename, data):
    with open(filename, 'w') as f:
        if isinstance(data, dict):
            for key, values in data.items():
                f.write(f'{key}: {" ".join(values)}\n')
        else:
            f.write(' '.join(data) if isinstance(data, list) else str(data))

def extract_sentiment_terms(sentence):
    # Tokenize words and tag part of speech
    words = word_tokenize(sentence)
    stop_words = set(stopwords.words('english'))
    tagged_words = pos_tag(words)
    sentiment_terms = set()

    for word, tag in tagged_words:
        if word.lower() not in stop_words and word not in string.punctuation:
            if tag.startswith('JJ') or tag.startswith('RB'):
                if word.lower() in opinion_lexicon.negative() or word.lower() in opinion_lexicon.positive():
                    sentiment_terms.add(word)
            
    return sentiment_terms

def preprocess_corpus(corpus: pd.DataFrame, text_column: str):
    # Implement preprocessing steps here

    print("reading corpus...")
    samples = [row[text_column] for _, row in corpus.iterrows() if isinstance(row[text_column], str) and len(row[text_column]) > 0]
    
    # get sentences from each review
    print("tokenising sentences...")
    sentences = [sentence.lower() for sentence in samples]

    # stop words and numbers
    stop_words = set(stopwords.words('english'))
    sentences = [' '.join([word for word in word_tokenize(sentence) if word not in stop_words and not word.isdigit()]) for sentence in tqdm(sentences)]

    # lemmisation
    lemmatizer = WordNetLemmatizer()
    sentences = [' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(sentence)]) for sentence in tqdm(sentences)]

    print("extracting sentiment terms -> sentiment terms")

    sentiment_terms = set()
    for sentence in tqdm(sentences):
        sentiment_terms.update(extract_sentiment_terms(sentence))

    print("tokenising words")
    processed_corpus = [word_tokenize(sentence) for sentence in tqdm(sentences)]

    return processed_corpus, sentiment_terms

def preprocess_text(text: str):
    # Check if text is a string and not empty
    if not isinstance(text, str) or len(text) == 0:
        return None

    # Convert to lower case
    sentence = text.lower()

    # Remove stop words and numbers
    stop_words = set(stopwords.words('english'))
    sentence = ' '.join([word for word in word_tokenize(sentence) if word not in stop_words and not word.isdigit()])

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    sentence = ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(sentence)])

    # Extract sentiment terms
    sentiment_terms = extract_sentiment_terms(sentence)

    # Tokenize words
    processed_text = word_tokenize(sentence)

    return processed_text, sentiment_terms

if __name__ == "__main__":

    # Load the dataset
    df = pd.read_csv('data/essays.csv')
    processed_corpus, sentiment_terms = preprocess_corpus(df, 'TEXT')

    # Write the processed corpus and sentiment terms to files
    write_to_file('output/processed_corpus.txt', str(processed_corpus))
    write_to_file('output/sentiment_terms.txt', str(sentiment_terms))

