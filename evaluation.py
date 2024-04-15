from os import write
import requests
import json
from preprocessing import invert_dict, write_to_file
from tqdm import tqdm
import pandas as pd
import liwc
from collections import Counter
import nltk



def liwc_comparison(lexicon, reviews):
    # Load the LIWC dictionary
    parse, category_names = liwc.load_token_parser('data/LIWC2007_English100131.dic')

    # Tokenise the reviews
    tokenised_reviews = [nltk.word_tokenize(review) for review in reviews]
    accuracies = []
    # Count the number of words in each category for each review and store in a dictionary with the review position as the key
    for review in tokenised_reviews:
        test = Counter(category for token in review for category in parse(token))

        # get the categories that are in the lexicon from each token in revieww
        categories = Counter(category for token in review for category in parse(token) if category in lexicon)

        # compare categories with test and get the intersection of the two
        intersection = test & categories

        accuracy = sum(intersection.values()) / sum(test.values()) if sum(test.values()) > 0 else 0
        
        accuracies.append(accuracy)
    return sum(accuracies) / len(accuracies)


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

def main():
    with open('output/lexicon.json', 'r') as f:
        lexicon = json.load(f)

    reviews = pd.read_csv('data/sample.csv')

    inverted_lexicon = invert_dict(lexicon)
    words = list(inverted_lexicon.keys())

    # print(llm_evaluate(sentences, categories))

    reviews = reviews['review_text'].sample(len(reviews) // 10)
    max_iter = 10
    v_avg = 0
    for _ in range(max_iter):
        v = liwc_comparison(lexicon, reviews)
        v_avg += v
    print(v_avg / max_iter) # 88.59520391834048%


    



if __name__ == "__main__":
    main()