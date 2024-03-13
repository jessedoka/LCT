import nltk
from nltk.corpus import wordnet
import json
from typing import List, Tuple

class LexiconBuilder:
    def __init__(self, review):
        nltk.download('wordnet')

        self.personality_traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
        self.emotional_states = ["anger", "fear", "joy", "sadness", "surprise"]
        self.interpersonal_skills = ["empathy", "assertiveness", "conflict", "leadership", "teamwork"]

        self.seed_lexicon = {
            "PersonalityTraits": self.personality_traits,
            "EmotionalStates": self.emotional_states,
            "InterpersonalSkills": self.interpersonal_skills
        }

        self.review = review 
        self.seed_hla_lexicon = {}

    def init_lexicon(self):
        for key in self.seed_lexicon:
            self.seed_hla_lexicon[key] = {}
            for word in self.seed_lexicon[key]:
                self.seed_hla_lexicon[key][word] = {"synonyms": [], "antonyms": []}

    def get_synonyms_antonyms(self, word) -> Tuple[List[str], List[str]]:
        synonyms = []
        antonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
                if lemma.antonyms():
                    antonyms.append(lemma.antonyms()[0].name())
        return synonyms, antonyms

    def get_definitions(self, word):
        definitions = []
        for syn in wordnet.synsets(word):
            definitions.append(syn.definition())
        return definitions
    
    def polarity_score(self, word):
        # get the polarity score of the word
        pass

    def create_lexicon(self):
        for key in self.seed_lexicon:
            for word in self.seed_lexicon[key]:
                
                synonyms, antonyms = self.get_synonyms_antonyms(word)
                self.seed_hla_lexicon[key][word]["synonyms"] = synonyms
                self.seed_hla_lexicon[key][word]["antonyms"] = antonyms
                self.seed_hla_lexicon[key][word]["definitions"] = self.get_definitions(word)
    
    
    def save_lexicon(self):
        with open('data/seed_hla_lexicon.json', 'w') as f:
            json.dump(self.seed_hla_lexicon, f)

    def print_lexicon(self):
        print(self.seed_hla_lexicon)

if __name__ == "__main__":
    lexicon_builder = LexiconBuilder('data/reviews.pkl')
    lexicon_builder.init_lexicon()
    lexicon_builder.create_lexicon()
    lexicon_builder.save_lexicon()
    lexicon_builder.print_lexicon()