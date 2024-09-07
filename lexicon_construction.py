from gensim.models import Word2Vec
import gensim.downloader as api

import networkx as nx
from collections import defaultdict

import nltk
from nltk.corpus import sentiwordnet as swn

import os
import gzip
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import psutil   
from loguru import logger

from preprocessing import preprocess_text, preprocess_corpus
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

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
    """CorpusStreamer class to stream a corpus from a gzip file.
    
    assumes the gzip file contains json objects with a 'review_text' key.
    
    Keyword arguments:
    gzip_file_path -- path to the gzip file
    preprocess_text -- function to preprocess text (default: preprocess_text)
    Return: yields processed text and sentiment terms
    """
    
    def __init__(self, gzip_file_path: str, preprocess_text=preprocess_text):
        self.gzip_file_path = gzip_file_path
        self.preprocess_text = preprocess_text

    def __iter__(self):
        with gzip.open(self.gzip_file_path, 'rt', encoding="utf-8") as file:
            for line in file:
                try:
                    review = json.loads(line.strip())
                    if review['review_text']: # Check if review_text is not empty
                        text = review['review_text']
                    result = self.preprocess_text(text)
                    if result is None:
                        logger.warning(f"Skipping line due to preprocessing error: {text}")
                        continue
                    processed_text, sentiment_terms = result
                    yield processed_text, sentiment_terms
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping line due to JSON error: {e}")
                    continue
                
def process_and_save_reviews(gzip_path, output_pickle_path, chunk_size=200_000):
    """
    Processes reviews from a large JSON file in a memory-efficient manner and saves them in chunks.

    Args:
        gzip_path (str): Path to the gzip file containing the JSON data.
        output_pickle_path (str): Path to save the processed DataFrame as a pickle file.
        chunk_size (int): Number of rows to process and save in each chunk.
    """
    streamer = CorpusStreamer(gzip_path)
    chunk = []
    temp_files = []
    
    logger.info(f"Processing and saving reviews in chunks of {chunk_size}")
    try:
        for i, (processed_text, sentiment_terms) in tqdm(enumerate(streamer)):
            chunk.append((processed_text, sentiment_terms))
            
            # Save in chunks of specified size to avoid high memory usage
            if (i + 1) % chunk_size == 0:
                temp_file = f"{output_pickle_path}_part_{i // chunk_size}.pkl"
                df = pd.DataFrame(chunk, columns=['processed_text', 'sentiment_terms'])
                df.to_pickle(temp_file, protocol=4)
                temp_files.append(temp_file)
                chunk = []  # Reset chunk to release memory

                # Log memory usage
                memory_usage = psutil.virtual_memory().percent
                logger.info(f"Memory usage after saving chunk: {memory_usage}%")

        # Save any remaining reviews that didn't fit the last chunk
        if chunk:
            temp_file = f"{output_pickle_path}_part_{len(temp_files)}.pkl"
            df = pd.DataFrame(chunk, columns=['processed_text', 'sentiment_terms'])
            logger.info(f"Saving the remaining reviews to {temp_file}")
            df.to_pickle(temp_file, protocol=4)
            temp_files.append(temp_file)

        # Concatenate all temporary pickle files into the final output pickle file
        logger.info(f"Concatenating all chunks into {output_pickle_path}")
        all_dfs = [pd.read_pickle(temp_file) for temp_file in temp_files]
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_pickle(output_pickle_path, protocol=4)

    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            os.remove(temp_file)
            logger.info(f"Removed temporary file {temp_file}")
        
def expand_seeds(corpus_path, seeds, model, Tc):
    """
    Expands seeds with a model and computes cosine similarity efficiently.

    Args:
        corpus_path (str): Path to the file with processed data.
        seeds (dict): Seed words with initial values.
        model (object): Pre-trained model for embeddings.
        Tc (float): Cosine similarity threshold.
    """
    
    logger.info("Expanding seeds with a model and computing cosine similarity efficiently")
    similarities = defaultdict(dict)

    # Initialize nearest neighbors model once
    vocab = set(model.index_to_key)
    seeds_in_vocab = vocab.intersection(seeds)
    index_to_term = list(seeds_in_vocab)

    vectors = np.array([model[term] for term in index_to_term])
    neighbors = NearestNeighbors(n_neighbors=len(vectors), metric='cosine')
    neighbors.fit(vectors)

    # Processing seeds without collecting terms in memory
    with pd.read_pickle(corpus_path) as df_chunk:
        logger.info("Processing seeds without collecting terms in memory")
        for _, chunk in tqdm(df_chunk):
            sentiment_terms = set(chunk)
            sentiment_terms_in_vocab = vocab.intersection(sentiment_terms)
            index_to_term = list(seeds_in_vocab) + list(sentiment_terms_in_vocab)
            vectors = np.array([model[term] for term in index_to_term])

            neighbors.fit(vectors)
            for i, vector in enumerate(vectors):
                distances, indices = neighbors.kneighbors(np.array([vector]), n_neighbors=len(vectors))

                # Store similarities without holding all terms
                for distance, index in zip(distances[0], indices[0]):
                    if 1 - distance > Tc:
                        term1 = index_to_term[i]
                        term2 = index_to_term[index]
                        similarities[term1][term2] = 1 - distance

    return similarities

def build_semantic_graph(C, model):
    G = nx.Graph()
    logger.info("Building semantic graph")
    for word_pair in tqdm(C):
        Si, Wj = word_pair
        if Si != Wj:
            G.add_edge(Si, Wj, weight=model.similarity(Si, Wj))
    return G

def multi_label_propagation(G, seeds, max_iterations=100):
    # Initialize labels for all nodes
    labels = {node: [] for node in G.nodes()}
    
    # Assign seeds with their labels
    for node, label in seeds.items():
        labels[node] = label

    # Propagate labels
    logger.info("Propagating labels")
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
    logger.info("Building lexicon")
    
    # Group words by label
    for word, categories in tqdm(labels.items()):
        for category in categories:
            lexicon[category].add(word)
    return {key: list(value) for key, value in lexicon.items()}

def visualize_embeddings(model):
    # Get the word vectors
    word_vectors = model.wv.vectors

    # Perform PCA
    pca = PCA(n_components=2)
    result = pca.fit_transform(word_vectors)

    # Create a scatter plot of the projection
    plt.scatter(result[:, 0], result[:, 1])
    
    words = list(model.wv.key_to_index)
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    plt.show()

def construct(corpus, seeds, Tc):
    output_corpus = 'output/processed_corpus.pkl'
    model = api.load("word2vec-google-news-300")
    process_and_save_reviews(corpus, output_corpus)
    
    C = expand_seeds(output_corpus, seeds, model, Tc)

    G = build_semantic_graph(C, model)

    labels = multi_label_propagation(G, seeds)
    lexicon = build_lexicon(labels)

    return lexicon, G, C



