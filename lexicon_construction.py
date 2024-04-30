
from re import sub
from gensim.models import Word2Vec
from gensim import downloader as api

import networkx as nx
import pandas as pd
from collections import defaultdict

import nltk
from nltk.corpus import sentiwordnet as swn

import gzip
from tqdm import tqdm
import numpy as np

from preprocessing import preprocess_text, preprocess_corpus
from sklearn.neighbors import NearestNeighbors

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

# This will crash computer!
class CorpusStreamer(object):
    def __init__(self, gzip_file_path, preprocess_text=preprocess_text):
        self.gzip_file_path = gzip_file_path
        self.preprocess_text = preprocess_text

    def __iter__(self):
        with gzip.open(self.gzip_file_path, 'rt') as file:
            print("Training...")
            for line in tqdm(file):
                processed_text, sentiment_terms = self.preprocess_text(line)
                yield processed_text, sentiment_terms

def learn_word_embeddings(gzip_file_path):
    print("learning word embeddings")
    processed_corpus, sentiment_terms = CorpusStreamer(gzip_file_path)
    model = Word2Vec(sentences=processed_corpus,
                     vector_size=100, window=5, min_count=1, workers=4)
    return model.wv, sentiment_terms


def expand_seeds(seeds, model, Tc, sentiment_terms):
    print("expanding seeds")
    similarities = defaultdict(dict)

    # Just get seed words
    seeds = set(seeds.keys())

    # Pre-calculate intersections
    vocab = set(model.index_to_key)
    seeds_in_vocab = vocab.intersection(seeds)
    sentiment_terms_in_vocab = vocab.intersection(sentiment_terms)

    index_to_term = list(seeds_in_vocab) + list(sentiment_terms_in_vocab)
    # term_to_index = {term: index for index, term in enumerate(index_to_term)}

    vectors = np.array([model[term] for term in index_to_term])

    neighbors = NearestNeighbors(n_neighbors=len(vectors), metric='cosine')
    neighbors.fit(vectors)
    
    print("finding neighbors")
    # Find neighbors for each vector
    for i, vector in tqdm(enumerate(vectors)):
        distances, indices = neighbors.kneighbors([vector]) #type: ignore

        # Iterate over neighbors
        for distance, index in zip(distances[0], indices[0]):
            # Only consider neighbors with cosine similarity > Tc
            if 1 - distance > Tc:
                term1 = index_to_term[i]
                term2 = index_to_term[index]
                
                # Make sure the two terms are not the same
                if term1 != term2:
                    similarities[term1][term2] = 1 - distance


    C = []
    for seed, terms in tqdm(similarities.items()):
        for term, similarity in terms.items():
            C.append((seed, term))


    return C

def build_semantic_graph(C, model):
    G = nx.Graph()
    print("building semantic graph")
    for word_pair in tqdm(C):
        Si, Wj = word_pair
        if Si != Wj:
            G.add_edge(Si, Wj, weight=model.similarity(Si, Wj))
    return G

def multi_label_propagation(G, seeds, max_iterations=100):
    # Initialize labels for all nodes
    labels = {node: [] for node in G.nodes()}
    
    # Assign seed labels
    for node, label in seeds.items():
        labels[node] = label

    # Propagate labels
    print("propagating labels")
    for _ in tqdm(range(max_iterations)):
        new_labels = labels.copy()
        for node in G.nodes():
            if node not in seeds:  

                # Gather labels from neighbors
                neighbor_labels = [labels[neighbor] for neighbor in G.neighbors(node)]
                neighbor_labels = [item for sublist in neighbor_labels for item in sublist] 
                
                # Assign the most common labels
                if neighbor_labels:
                    unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
                    common_labels = unique_labels[np.where(counts == np.max(counts))]
                    new_labels[node] = list(common_labels)
        labels = new_labels
    return labels

def build_lexicon(labels):
    lexicon = defaultdict(set)
    print("building lexicon")
    
    # Group words by label
    for word, categories in tqdm(labels.items()):
        for category in categories:
            lexicon[category].add(word)
    return {key: list(value) for key, value in lexicon.items()}

def construct(corpus, seeds, Tc, text):
    # _, sentiment_terms = preprocess_corpus(corpus, text)
    model, sentiment_terms = learn_word_embeddings(corpus) 
    # model = api.load("word2vec-google-news-300")
    C = expand_seeds(seeds, model, Tc, sentiment_terms)

    G = build_semantic_graph(C, model)

    labels = multi_label_propagation(G, seeds)
    lexicon = build_lexicon(labels)

    return lexicon, G, C



