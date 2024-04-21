
from re import sub
from gensim.models import Word2Vec
from gensim import downloader as api

import networkx as nx
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

import ast
import nltk
from nltk.corpus import sentiwordnet as swn
import liwc

import json
from torch import le
from tqdm import tqdm
import numpy as np

from preprocessing import preprocess_corpus, write_to_file, invert_dict

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

def expand_seeds(seeds, model, Tc, sentiment_terms):
    print("expanding seeds")
    similarities = defaultdict(dict)

    # seeds is a dictionary with words as keys and categories as values

    # just get the words
    seeds = set(seeds.keys())
    
    # Pre-calculate intersections
    vocab = set(model.index_to_key)
    seeds_in_vocab = vocab.intersection(seeds)
    sentiment_terms_in_vocab = vocab.intersection(sentiment_terms)

    for seed in tqdm(seeds_in_vocab):
        for term in sentiment_terms_in_vocab:
            similarity = model.similarity(seed, term)
            if similarity > Tc:
                similarities[seed][term] = similarity

    # Sort by similarity
    C = []
    for seed, terms in similarities.items():
        for term, similarity in terms.items():
            C.append((seed, term))

    return C

def build_semantic_graph(C, model):
    G = nx.Graph()
    print("building semantic graph")
    for word_pair in tqdm(C):
        Si, Wj = word_pair

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

def construct(corpus, seeds, Tc):
    _, sentiment_terms = preprocess_corpus(corpus, 'review_text')

    model = api.load("word2vec-google-news-300")

    C = expand_seeds(seeds, model, Tc, sentiment_terms)

    G = build_semantic_graph(C, model)

    labels = multi_label_propagation(G, seeds)
    lexicon = build_lexicon(labels)

    return lexicon, G, C


if __name__ == "__main__":
    # Example usage
    corpus = pd.read_csv('data/sample.csv')
    ocean = pd.read_csv('data/seeds.csv')
    liwc = pd.read_csv('data/liwc_lexicon.csv')

    # OCEAN traits
    ocean = {word: trait for trait in ocean.columns for word in ocean[trait].dropna().tolist()}

    liwc = {word: ast.literal_eval(category) for word, category in zip(liwc['word'], liwc['categories'])}

    # Create a new dictionary that only includes words present in both dictionaries
    seeds = {word: [ocean[word]] + liwc[word] for word in ocean if word in liwc}

    Tc = 0.7  # Threshold for similarity
    lexicon, G, C = construct(corpus, seeds, Tc)

    print(len(lexicon), len(invert_dict(lexicon)), len(G.edges()), len(C))

    # show me a graph as a png

    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_size=20, node_size=50, edge_color='gray', alpha=0.5, width=0.5, font_color='black')

    plt.savefig('output/graph.png')


