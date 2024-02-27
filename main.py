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

# Read the file
with open(fileloc, "r") as file:
    text = file.read()

# Process the text
doc = nlp(text)

characters = {
}

# Identify and track the character
for entity in doc.ents:
    print(entity.text, entity.label_)
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

# output into a json file
import json

with open("characters_track.json", "w") as file:
    json.dump(characters, file, indent=4)

print("Characters extracted and saved to characters.json")