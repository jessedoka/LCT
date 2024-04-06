from gensim.models import Word2Vec
from gensim import downloader as api
import networkx as nx
import pandas as pd
from collections import defaultdict

import nltk
from nltk.corpus import sentiwordnet as swn

import json
from tqdm import tqdm
from preprocessing import preprocess_corpus, preprocess_text, write_to_file

import joblib

nltk.download('opinion_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')

def get_synonyms(word):
    synonyms = set()

    for syn in swn.senti_synsets(word):
        for lemma in syn.synset.lemmas():
            synonyms.add(lemma.name())
    return synonyms

def learn_word_embeddings(processed_corpus):
    # Train a Word2Vec model on the processed corpus
    # model = Word2Vec(sentences=processed_corpus,
    #                  vector_size=100, window=5, min_count=1, workers=4)
    # w2v
    model = api.load('word2vec-google-news-300')
    return model

def expand_seeds(seeds, model, Tc, sentiment_terms):
    print("expanding seeds")
    similarities = defaultdict(dict)
    
    # Pre-calculate intersections
    vocab = set(model.wv.index_to_key)
    seeds_in_vocab = vocab.intersection(seeds)
    sentiment_terms_in_vocab = vocab.intersection(sentiment_terms)

    for word in tqdm(vocab):
        if word not in sentiment_terms_in_vocab:
            continue
        for seed in seeds:

            if seed in seeds_in_vocab:
                # Calculate similarity between seed and word
                similarities[seed][word] = model.wv.similarity(seed, word)
            else:

                # Get synonyms of seed
                synonyms = set(get_synonyms(seed)).intersection(vocab)

                for synonym in synonyms:
                    similarities[synonym][word] = model.wv.similarity(synonym, word)

                if not synonyms:
                    similarities[seed][word] = 0

    C = set()
    for seed, similar_words in tqdm(similarities.items()):
        for word, similarity in similar_words.items():
            if similarity >= Tc:
                C.add((seed, word))     
    return C, seeds

def build_semantic_graph(C, model):
    G = nx.Graph()
    print("building semantic graph")
    for word_pair in tqdm(C):
        Si, Wj = word_pair

        G.add_edge(Si, Wj, weight=model.wv.similarity(Si, Wj))
    return G

def label_propagation(G, seeds, max_iterations=100):
    # Initialize labels based on seeds
    labels = {node: "objective" for node in G.nodes()}  # Default label
    for seed, label in seeds.items():
        labels[seed] = label  # label for each seed

    for _ in range(max_iterations):
        prev_labels = labels.copy()
        for node in G.nodes():
            if node not in seeds:  # Don't update seed labels
                neighbor_labels = [labels[neighbor] for neighbor in G.neighbors(node)]

                labels[node] = max(set(neighbor_labels), key=neighbor_labels.count)

        # Check for convergence
        if prev_labels == labels:
            break
        
    return labels

def classify_corpus(df):
    # Load the model
    model = joblib.load('models/extroversion_model_cv.pkl')

    # Preprocess the text
    df['preprocessed_text'] = df['review_text'].apply(preprocess_text)

    prediction = model.predict(df['preprocessed_text'])
    
    # Classify the corpus
    df['personality_trait'] = prediction

    return df

def build_lexicon(labels):
    lexicon = defaultdict(set)
    print("building lexicon")
    for word, label in tqdm(labels.items()):
        if label is not None:
            lexicon[label].add(word)
    return lexicon

def main(corpus, seeds, Tc):
    processed_corpus, sentiment_terms = preprocess_corpus(corpus, 'review_text')

    model = learn_word_embeddings(processed_corpus)

    C, seeds = expand_seeds(seeds, model, Tc, sentiment_terms)

    G = build_semantic_graph(C, model)
    # class_corpus = classify_corpus(corpus)

    # labels = label_propagation(G, seeds)
    # lexicon = build_lexicon(labels)
    lexicon = None 
    return lexicon, G, C


if __name__ == "__main__":
    # Example usage
    corpus = pd.read_csv('data/sample.csv')
    seeds = pd.read_csv('data/seeds.csv')
    liwc_seeds = pd.read_csv('data/liwc_lexicon.csv')

    # OCEAN traits
    # seeds = {word: trait for trait in seeds.columns for word in seeds[trait].dropna().tolist()}

    # LIWC
    seeds = {word: category for word, category in zip(liwc_seeds['word'], liwc_seeds['categories'])}


    Tc = 0.9  # Threshold for similarity
    lexicon, G, C = main(corpus, seeds, Tc)

    # lexicon = {key: list(value) for key, value in lexicon.items()}

    # write_to_file('output/lexicon.json', json.dumps(lexicon, indent=4))
    write_to_file('output/graph.txt', G.edges())
    write_to_file('output/candidate.txt', C)

