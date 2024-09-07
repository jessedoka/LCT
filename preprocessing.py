import nltk 
from nltk.corpus import opinion_lexicon
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

import string
import pandas as pd
from tqdm import tqdm
import json

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
    punctuation = set(string.punctuation)
    tagged_words = pos_tag(words)
    sentiment_terms = set()

    # Convert opinion lexicon to sets for faster lookup
    negative_lexicon = set(opinion_lexicon.negative())
    positive_lexicon = set(opinion_lexicon.positive())

    for word, tag in tagged_words:
        lower_word = word.lower()
        if lower_word not in stop_words and word not in punctuation:
            # Check if word is an adjective or adverb
            if tag.startswith('JJ') or tag.startswith('RB'):
                if lower_word in negative_lexicon or lower_word in positive_lexicon:
                    sentiment_terms.add(word)
            
    return sentiment_terms

def preprocess_corpus(corpus: pd.DataFrame, text_column: str):
    """Preprocess a corpus of text data assumed to be in a column of a pandas dataframe. 
    
    Keyword arguments:
    corpus -- the pandas dataframe containing the corpus
    text_column -- the name of the column containing the text data
    return: processed_corpus, sentiment_terms
    """
    
    # Implement preprocessing steps here

    print("reading corpus...")
    samples = [row[text_column] for _, row in corpus.iterrows() if isinstance(row[text_column], str) and len(row[text_column]) > 0]
    
    # get sentences from each review
    print("tokenising sentences...")
    sentences = [sentence.lower() for sentence in samples]

    print("removing stop words and numbers -> lemmatisation")
    # stop words and numbers
    stop_words = set(stopwords.words('english'))
    sentences = [' '.join([word for word in word_tokenize(sentence) if word not in stop_words and not word.isdigit()]) for sentence in tqdm(sentences)]

    print("lemmatisation")

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


def invert_dict(d: dict[str, list[str]]) -> dict[str, list[str]]:
    # Invert a dictionary
    inverted_dict = {}
    for key, values in tqdm(d.items()):
        for value in values:
            if value in inverted_dict:
                inverted_dict[value].append(key)
            else:
                inverted_dict[value] = [key]
        
    return {key: list(value) for key, value in inverted_dict.items()}

if __name__ == "__main__":

    with open('output/lexicon.json', 'r') as f:
        lexicon = json.load(f)

    lexicon = invert_dict(lexicon)
    write_to_file('output/inverted_lexicon.json', json.dumps(lexicon, indent=4))


