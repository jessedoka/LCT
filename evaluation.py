from numpy import invert
import requests
import json
import random

import matplotlib.pyplot as plt
import networkx as nx

from preprocessing import invert_dict, write_to_file
from lexicon_construction import construct
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tqdm import tqdm
import pandas as pd

import liwc

from collections import Counter

import nltk 
import ast

def liwc_comparison(lexicon, reviews, max_iter=10):
    # Load the LIWC dictionary
    parse, category_names = liwc.load_token_parser('data/LIWC2007_English100131.dic')

    # Tokenise the reviews
    tokenised_reviews = [nltk.word_tokenize(review) for review in reviews]
    accuracies = []
    # Count the number of words in each category for each review and store in a dictionary with the review position as the key
    for review in tokenised_reviews:
        test = Counter(category for token in review for category in parse(token))

        # get the categories that are in the lexicon from each token in review
        categories = Counter(category for token in review for category in parse(token) if category in lexicon)

        # compare categories with test and get the intersection of the two
        intersection = test & categories

        accuracy = sum(intersection.values()) / sum(test.values()) if sum(test.values()) > 0 else 0
        
        accuracies.append(accuracy)

        reviews = reviews['review_text'].sample(len(reviews) // 10)
  

        for _ in range(max_iter):
            try:
                v = sum(accuracies) / len(accuracies)
            except ZeroDivisionError:
                v = 0
            v_avg += v
        
    return v_avg / max_iter # 88.59520391834048%


def sentence_generation(words: list) -> list:
    templates = [
        "I feel {} today.",
        "This is a very {} thing.",
        "Why does it always seem so {}?"
    ]

    sentences = []
    for word in words:
        for template in templates:
            sentence = template.format(word)
            sentences.append(sentence)
    return sentences

def llm_evaluate(sentences: list, categories: list) -> None:
    responses = []
    data = {
            "model": 'llama2',
            "stream": False
        }
    url = 'http://localhost:11434/api/generate'

    for sentence in tqdm(sentences):
        data["prompt"] = f"""For the following categories: {categories}, determine  which category the following sentence belongs to: {sentence}, \
    
        Give me the categories that the sentence belongs to only if the sentence belongs to a category. If the sentence does not belong to any category, respond with 'None'."""

        response = requests.post(url, json=data)
        response = response.json()['response']
        responses.append(response)

    # Evaluate the responses
    for sentence, response in zip(sentences, responses):
        print(f"Sentence: {sentence}")
        print(f"Response: {response}")
        print("\n")

    return None

def show_graph(G):
    plt.figure(figsize=(100, 100))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=50, font_size=10, font_color='black', font_weight='bold', edge_color='gray', node_color='skyblue', linewidths=0.5)

    plt.savefig('output/graph.png')



def consistency_check(lexicon, C):
    """
    Review if similar words have been grouped under consistent categories and if the classifications make sense logically. For instance, check if synonyms or related words consistently share categories.
    """

    # Check if similar words have been grouped under consistent categories
    for seed, term in C:
        if seed in lexicon and term in lexicon and seed != term:
            # similarity percentage between seed and term 
            similarity = len(set(lexicon[seed]) & set(lexicon[term])) / len(set(lexicon[seed]) | set(lexicon[term]))
            
            print(f"Similarity between {seed} and {term}: {similarity}")

    return None


if __name__ == "__main__":

    # Example usage
    corpus = pd.read_csv('data/sample3.csv').sample(60000)
    ocean = pd.read_csv('data/ocean.csv')
    liwc = pd.read_csv('data/liwc.csv')

    # creating seeds... 
    ocean = {word: trait for trait in ocean.columns for word in ocean[trait].dropna().tolist()}
    liwc = {word: ast.literal_eval(category) for word, category in zip(liwc['word'], liwc['categories'])}

    # Create a new dictionary that only includes words present in both dictionaries
    seeds = {word: [ocean[word]] + liwc[word] for word in ocean if word in liwc}

    # words that are unique to liwc
    liwc_unique = {word: liwc[word] for word in liwc if word not in ocean}
    ocean_unique = {word: ocean[word] for word in ocean if word not in liwc} 

    Tc = 0.7  # Threshold for similarity
    lexicon, G, C = construct(corpus, seeds, Tc)

    print(len(lexicon), len(invert_dict(lexicon)), len(G.edges), len(C))
    write_to_file('output/lexicon.json', json.dumps(lexicon, indent=4))
    consistency_check(lexicon, C)
    show_graph(G)
