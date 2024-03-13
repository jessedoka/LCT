import numpy as np
from gensim.models import Word2Vec
import networkx as nx
import pandas as pd
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
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(sentence)
    tagged_words = pos_tag(words)
    sentiment_terms = set()

    for word, tag in tagged_words:
        if word.lower() not in stop_words and word not in string.punctuation:
            if tag.startswith('JJ') or tag.startswith('RB'):
                if word.lower() in opinion_lexicon.positive() or word.lower() in opinion_lexicon.negative():
                    sentiment_terms.add(word)

    return sentiment_terms


def preprocess_corpus(corpus):
    # Implement preprocessing steps here
    # This might include tokenization, removing stop words, stemming, etc.

    # read the csv file only return the review_text
    df = pd.read_csv(corpus)
    processed_corpus = df['review_text'].tolist()
    
    # tokenization
    processed_corpus = [nltk.word_tokenize(review) for review in processed_corpus]

    # remove stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))

    for i in range(len(processed_corpus)):
        processed_corpus[i] = [word for word in processed_corpus[i] if word not in stop_words]

    # stemming
    stemmer = nltk.stem.PorterStemmer()

    for i in range(len(processed_corpus)):
        processed_corpus[i] = [stemmer.stem(word) for word in processed_corpus[i]]

    # # extract sentiment terms
    # for i in range(len(processed_corpus)):
    #     processed_corpus[i] = list(extract_sentiment_terms(' '.join(processed_corpus[i])))

    return processed_corpus


def learn_word_embeddings(processed_corpus):
    # Train a Word2Vec model on the processed corpus
    model = Word2Vec(sentences=processed_corpus,
                     vector_size=100, window=5, min_count=1, workers=4)
    return model


def expand_seeds(seeds, model, Tc):
    C = set()
    for Si in seeds:
        for Wj in model.wv.index_to_key:
            if model.wv.similarity(Si, Wj) >= Tc:
                # Adding both the seed and the similar word to the candidate set
                C.add((Si, Wj))
    return C


def build_semantic_graph(C, model):
    G = nx.Graph()
    for word_pair in C:
        # Assuming word_pair is a tuple (Si, Wj)
        Si, Wj = word_pair
        # Add an edge between Si and Wj with weight as their similarity
        G.add_edge(Si, Wj, weight=model.wv.similarity(Si, Wj))
    return G


def label_propagation(G, seeds, max_iterations=100):

    # Initialize labels based on seeds
    labels = {node: 0 for node in G.nodes()}  # Default label
    for seed in seeds:
        labels[seed] = 1  # Assuming 1 for depressive, -1 for non-depressive

    for _ in range(max_iterations):
        prev_labels = labels.copy()
        for node in G.nodes():
            if node not in seeds:  # Don't update seed labels
                # Update label based on the weighted average of neighbor labels
                labels[node] = int(np.mean([labels[neighbor] for neighbor in G.neighbors(node)]))
                # print(labels[node])

        # Check for convergence
        if prev_labels == labels:
            break
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
    C = expand_seeds(seeds, model, Tc)

    G = build_semantic_graph(C, model)

    labels = label_propagation(G, seeds)
    Ld, Ln = build_lexicon(labels)
    L = Ld.union(Ln)
    return L, Ld, Ln, G, C


# Example usage
corpus = 'data/reviews.csv' 
# big five personality traits
seeds = ["disappointed", "sad", "happy", "pleased"]

Tc = 0.5  # Threshold for similarity
L, Ld, Ln, G, C = main(corpus, seeds, Tc)

with open('output/lexicon.txt', 'w') as f:
    f.write(' '.join(L))

with open('output/depressive_lexicon.txt', 'w') as f:
    f.write(' '.join(Ld))

with open('output/non_depressive_lexicon.txt', 'w') as f:
    f.write(' '.join(Ln))

with open('output/graph.txt', 'w') as f: 
    f.write(str(G.edges()))

with open('output/candidate.txt', 'w') as f: 
    f.write(str(C))