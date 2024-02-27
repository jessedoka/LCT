# goal: scrape text data from google scholars
import random
import spacy
import argparse
# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# # Sample text
argparser = argparse.ArgumentParser()
argparser.add_argument('text', help='text to be processed')

args = argparser.parse_args()
fileloc = args.text

def character_tracker(file: str, doc=None) -> dict:
    """
    file: str
    doc: spacy doc object
    returns a dictionary of characters, with their associated count and context.
    
    This is function that takes a spacy doc object or a file if the doc is not entered 
    and returns a dictionary of characters, with their associated count and context. 
    """

    if doc is None:
        with open(file, "r") as f:
            text = f.read()
            doc = nlp(text)
    
    characters = {}

    for entity in doc.ents:
        # print(entity.text, entity.label_)
        # entity need to appear more than 5 times to be classed a character
        if entity.label_ == "PERSON" and text.count(entity.text) > 5:
            # add list of characters as an object
            if entity.text not in characters:
                characters[entity.text] = {
                    "count": text.count(entity.text),
                    "context": [entity.sent.text]
                }
            else:
                characters[entity.text]["context"].append(entity.sent.text)

    return characters

print(character_tracker(fileloc))

