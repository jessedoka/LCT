import requests
import json
from preprocessing import invert_dict
from tqdm import tqdm

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

def evaluate(sentences: list, categories: list) -> None:
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

    inverted_lexicon = invert_dict(lexicon)
    words = list(inverted_lexicon.keys())

    sentences = sentence_generation(words)
    categories = list(lexicon.keys())

    print(evaluate(sentences, categories))

if __name__ == "__main__":
    main()