from os import write
import numpy as np
from gensim.models import Word2Vec
import networkx as nx
import pandas as pd
from collections import defaultdict

import matplotlib.pyplot as plt

import nltk
from nltk.corpus import opinion_lexicon
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

from nltk.corpus import sentiwordnet as swn

import string
import json
from tqdm import tqdm

nltk.download('opinion_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')


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

def get_synonyms(word):
    synonyms = set()

    for syn in swn.senti_synsets(word):
        for lemma in syn.synset.lemmas():
            synonyms.add(lemma.name())
    return synonyms


def preprocess_corpus(corpus, text_column):
    # Implement preprocessing steps here

    print("reading corpus...")
    df = pd.read_csv(corpus, chunksize=1000)
    samples = [sample for chunk in df for sample in chunk[text_column] if isinstance(sample, str) and len(sample) > 0]
    
    # get sentences from each review
    print("tokenising sentences...")
    sentences = [sentence.lower() for sentence in samples]

    # stop words and numbers
    stop_words = set(stopwords.words('english'))
    sentences = [' '.join([word for word in word_tokenize(sentence) if word not in stop_words and not word.isdigit()]) for sentence in tqdm(sentences)]

    print("extracting sentiment terms -> sentiment terms")

    sentiment_terms = set()
    for sentence in tqdm(sentences):
        sentiment_terms.update(extract_sentiment_terms(sentence))

    print("tokenising words")
    processed_corpus = [word_tokenize(sentence) for sentence in tqdm(sentences)]

    # Extract sentiment terms from the corpus
    

    return processed_corpus, sentiment_terms


def learn_word_embeddings(processed_corpus):
    # Train a Word2Vec model on the processed corpus
    model = Word2Vec(sentences=processed_corpus,
                     vector_size=100, window=5, min_count=1, workers=4)
    return model


def expand_seeds(seeds, model, Tc, sentiment_terms):

    print("expanding seeds")
    similarities = defaultdict(dict)
    for word in tqdm(model.wv.index_to_key):
        for seed in seeds:
            if seed in model.wv.index_to_key and word in sentiment_terms:
                similarities[seed][word] = model.wv.similarity(seed, word)
            else:
                synonyms = get_synonyms(seed)
                for synonym in synonyms:
                    if synonym in model.wv.index_to_key:
                        similarities[synonym][word] = model.wv.similarity(synonym, word)
                else:
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

    labels = label_propagation(G, seeds)
    lexicon = build_lexicon(labels)
    return lexicon, G, C


# Example usage
corpus = 'data/sample.csv'

# OCEAN personality traits
seeds = pd.read_csv('data/seeds.csv')

# seeds = {word: trait for trait in seeds.columns for word in seeds[trait].dropna().tolist()}

seeds = {'creative': 1, 'sad': -1, 'happy': 1, 'angry': -1, 'calm': 1, 'anxious': -1, 'energetic': 1, 'lazy': -1}

Tc = 0.5  # Threshold for similarity
lexicon, G, C = main(corpus, seeds, Tc)


def write_to_file(filename, data):
    with open(filename, 'w') as f:
        if isinstance(data, dict):
            for key, values in data.items():
                f.write(f'{key}: {" ".join(values)}\n')
        else:
            f.write(' '.join(data) if isinstance(data, list) else str(data))

lexicon = {key: list(value) for key, value in lexicon.items()}
write_to_file('output/lexicon.json', json.dumps(lexicon, indent=4))

write_to_file('output/lexicon.txt', lexicon)
write_to_file('output/graph.txt', G.edges())

write_to_file('output/candidate.txt', C)

