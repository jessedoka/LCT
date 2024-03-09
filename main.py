# building a lexicon of positive and negative words

import spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
import json
import pandas as pd
from typing import List, Tuple

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("spacy_wordnet", after='tagger')

# seed attributes
# https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html

personality_traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
emotional_states = ["anger", "fear", "joy", "sadness", "surprise"]
interpersonal_skills = ["empathy", "assertiveness", "conflict", "leadership", "teamwork"]

seed_lexicon = {
    "PersonalityTraits": personality_traits,
    "EmotionalStates": emotional_states,
    "InterpersonalSkills": interpersonal_skills
}

review = 'data/reviews.pkl' 

# use wordnet to create seed lexicon

seed_emotions_lexicon = {}

def init_lexicon():
    for key in seed_lexicon:
        seed_emotions_lexicon[key] = {}
        for word in seed_lexicon[key]:
            seed_emotions_lexicon[key][word] = {"synonyms": [], "antonyms": []}

def get_synonyms_antonyms(word) -> Tuple[List[str], List[str]]:
    synonyms = []
    antonyms = []
    for syn in word._.wordnet.synsets():
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
            if lemma.antonyms():
                antonyms.append(lemma.antonyms()[0].name())
    return synonyms, antonyms

def get_definitions(word):
    definitions = []
    for syn in word._.wordnet.synsets():
        definitions.append(syn.definition())
    return definitions

def polarity_score(word):
    # get the polarity score of the word
    pass



def create_lexicon():
    for key in seed_lexicon:
        for word in seed_lexicon[key]:
            synonyms, antonyms = get_synonyms_antonyms(nlp(word)[0])
            seed_emotions_lexicon[key][word]["synonyms"] = synonyms
            seed_emotions_lexicon[key][word]["antonyms"] = antonyms
            seed_emotions_lexicon[key][word]["definitions"] = get_definitions(nlp(word)[0])

def main():
    init_lexicon()
    create_lexicon()
    with open('data/seed_emotions_lexicon.json', 'w') as f:
        json.dump(seed_emotions_lexicon, f)
    print(seed_emotions_lexicon)

if __name__ == "__main__":
    main()
    
                                                                                                               