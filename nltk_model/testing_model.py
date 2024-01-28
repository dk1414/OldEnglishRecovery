import dill as pickle
import re
from nltk.lm import NgramCounter
import math

# Load the trained n-gram model
with open('results.pkl', 'rb') as model_file:
    trained_model = pickle.load(model_file)

# Read the masked text from "wiki.train.txt"
with open('wiki.test.txt', 'r', encoding='utf-8') as masked_file:
    masked_text = masked_file.read()

masked_words = re.findall(r'\b\w+\b', masked_text)

predicted_words = []
for word in masked_words:
    if word == '[MASK]':
        context = [] 
        predicted_word = trained_model.generate(text_seed=context, random_seed=42)  # Use the model to predict the next word
        predicted_words.append(predicted_word)
    else:
        predicted_words.append(word)

# comparing raw words with text
with open('wiki.test.raw', 'r', encoding='utf-8') as raw_file:
    raw_text = raw_file.read()

raw_words = re.findall(r'\b\w+\b', raw_text)


