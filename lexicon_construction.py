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

from nltk.corpus import wordnet
import string

nltk.download('opinion_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def initialize_lexicon():
    C, Ld, Ln = set(), set(), set()
    return C, Ld, Ln


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
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms


def preprocess_corpus(corpus):
    # Implement preprocessing steps here
    # This might include tokenization, removing stop words, stemming, etc.

    # read the csv file only return the review_text
    df = pd.read_csv(corpus)
    reviews =  df['review_text'].tolist()
    
    # get sentences from each review
    sentences = []
    for review in reviews:
        sentences.extend(nltk.sent_tokenize(review))

    
    processed_corpus = [word_tokenize(sentence) for sentence in sentences]

    # Extract sentiment terms from the corpus
    sentiment_terms = set()
    for sentence in sentences:
        sentiment_terms.update(extract_sentiment_terms(sentence))

    print(f"sentiment_terms: {sentiment_terms} -> {len(sentiment_terms)} words")

    # flat_corpus = [word for sublist in processed_corpus for word in sublist]

    return processed_corpus, sentiment_terms


def learn_word_embeddings(processed_corpus):
    # Train a Word2Vec model on the processed corpus
    model = Word2Vec(sentences=processed_corpus,
                     vector_size=100, window=5, min_count=1, workers=4)
    return model


def expand_seeds(seeds, model, Tc):
    # TODO: Implement the expansion of seeds using wordnet 

    # if model cannot find word in seed then use wordnet to find synonyms 

    # Pre-compute a dictionary of similarities 
    similarities = defaultdict(dict)
    for word in model.wv.index_to_key:
        for seed in seeds:
            if seed in model.wv.index_to_key:
                similarities[seed][word] = model.wv.similarity(seed, word)
            else:
                synonyms = get_synonyms(seed)
                for synonym in synonyms:
                    if synonym in model.wv.index_to_key:
                        similarities[synonym][word] = model.wv.similarity(synonym, word)
                else:
                    similarities[seed][word] = 0
    print(f"similarities: {similarities}")

    C = set()
    for seed, similar_words in similarities.items():
        print(f"seed: {seed}")
        for word, similarity in similar_words.items():
            if similarity >= Tc:
                print(f"word: {word} similarity: {similarity}")
                C.add((seed, word))     
    print(f"C: {C} -> {len(C)} pairs")
    return C, seeds


def build_semantic_graph(C, model):
    G = nx.Graph()
    for word_pair in C:
        # Assuming word_pair is a tuple (Si, Wj)
        Si, Wj = word_pair
        # Add an edge between Si and Wj with weight as their similarity
        G.add_edge(Si, Wj, weight=model.wv.similarity(Si, Wj))

    print(f"graph edges: {G.edges()} -> {len(G.edges())} edges")
    return G

def label_propagation(G, seeds, max_iterations=100):
    # Initialize labels based on seeds
    labels = {node: 0 for node in G.nodes()}  # Default label
    for seed, label in seeds.items():
        labels[seed] = label  # label for each seed

    for _ in range(max_iterations):
        prev_labels = labels.copy()
        for node in G.nodes():
            if node not in seeds: 
                # Don't update seed labels

                # Update label based on the weighted average of neighbor labels
                labels[node] = int(np.mean([labels[neighbor]
                                   for neighbor in G.neighbors(node)]))

        # Check for convergence
        if prev_labels == labels:
            break

    print(f"labels: {labels}")
    return labels


def build_lexicon(labels, sentiment_terms):
    Ld, Ln = set(), set()
    for word, label in labels.items():
        if word in sentiment_terms:
            if label < 0 and abs(label) > 0.5:
                Ld.add(word)
            elif label > 0 and abs(label) > 0.5:
                Ln.add(word)
    return Ld, Ln


def main(corpus, seeds, Tc):
    C, Ld, Ln = initialize_lexicon()
    processed_corpus, sentiment_terms = preprocess_corpus(corpus)
    model = learn_word_embeddings(processed_corpus)
    C, seeds = expand_seeds(seeds, model, Tc)

    G = build_semantic_graph(C, model)

    labels = label_propagation(G, seeds)
    Ld, Ln = build_lexicon(labels, sentiment_terms)
    L = Ld.union(Ln)
    return L, Ld, Ln, G, C
    # return None, None, None, None, None


# Example usage
corpus = 'data/sample.csv' 

# Seeds for different emotions
seeds = {"anger": -1, "fear": -1, "joy": 1, "sadness": -1, "surprise": 1, "disgust": -1, "trust": 1, "anticipation": 1}

Tc = 0.5  # Threshold for similarity
L, Ld, Ln, G, C = main(corpus, seeds, Tc)

def write_to_file(filename, data):
    with open(filename, 'w') as f:
        f.write(' '.join(data) if isinstance(data, list) else str(data))

write_to_file('output/lexicon.txt', L)
write_to_file('output/depressive_lexicon.txt', Ld)
write_to_file('output/non_depressive_lexicon.txt', Ln)
write_to_file('output/graph.txt', G.edges())
write_to_file('output/candidate.txt', C)

