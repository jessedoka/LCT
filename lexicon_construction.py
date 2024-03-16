import numpy as np
from gensim.models import Word2Vec
import networkx as nx
import pandas as pd

import matplotlib.pyplot as plt

import nltk
from nltk.corpus import opinion_lexicon
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

from nltk.corpus import wordnet
import string

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
                if word.lower() in opinion_lexicon.positive() or word.lower() in opinion_lexicon.negative():
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

    # Extract sentiment terms from each sentence
    sentiment_terms = set()
    for sentence in sentences:
        sentiment_terms.update(extract_sentiment_terms(sentence))

    # Add the sentiment terms to the processed corpus
    processed_corpus = [word_tokenize(term) for term in sentiment_terms]

    return processed_corpus


def learn_word_embeddings(processed_corpus):
    # Train a Word2Vec model on the processed corpus
    model = Word2Vec(sentences=processed_corpus,
                     vector_size=100, window=5, min_count=1, workers=4)
    
    print(model.wv.similarity('cry', 'good'))
    return model


def expand_seeds(seeds, model, Tc):
    # TODO: Implement the expansion of seeds using wordnet 

    # if model cannot find word in seed then use wordnet to find synonyms 

    # Pre-compute a dictionary of similarities 
    similarities = {seed: {} for seed in seeds}
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
    print(similarities)
    C = set()
    for seed, similar_words in similarities.items():
        for word, similarity in similar_words.items():
            if similarity >= Tc:
                C.add((seed, word))     
                     
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
            if node not in seeds:  # Don't update seed labels
                # Update label based on the weighted average of neighbor labels
                # print neighbors
                labels[node] = int(np.mean([labels[neighbor]
                                   for neighbor in G.neighbors(node)]))

        # Check for convergence
        if prev_labels == labels:
            break

    print(f"labels: {labels}")
    return labels


def build_lexicon(labels):
    Ld, Ln = set(), set()
    for word, label in labels.items():
        if label < 0 and abs(label) > 0.5:
            Ld.add(word)
        elif label > 0 and abs(label) > 0.5:
            Ln.add(word)
    return Ld, Ln


def main(corpus, seeds, Tc):
    C, Ld, Ln = initialize_lexicon()
    processed_corpus = preprocess_corpus(corpus)
    model = learn_word_embeddings(processed_corpus)
    # C, seeds = expand_seeds(seeds, model, Tc)

    # G = build_semantic_graph(C, model)

    # labels = label_propagation(G, seeds)
    # Ld, Ln = build_lexicon(labels)
    # L = Ld.union(Ln)
    # return L, Ld, Ln, G, C
    return None, None, None, None, None


# Example usage
corpus = 'data/tinysample.csv' 
# big five personality traits
seeds = {"anger": -1, "fear": -1, "joy": 1, "sadness": -1, "surprise": 1}

Tc = 0.5  # Threshold for similarity
L, Ld, Ln, G, C = main(corpus, seeds, Tc)

# with open('output/lexicon.txt', 'w') as f:
#     f.write(' '.join(L))

# with open('output/depressive_lexicon.txt', 'w') as f:
#     f.write(' '.join(Ld))

# with open('output/non_depressive_lexicon.txt', 'w') as f:
#     f.write(' '.join(Ln))

# with open('output/graph.txt', 'w') as f: 
#     f.write(str(G.edges()))

# with open('output/candidate.txt', 'w') as f: 
#     f.write(str(C))

