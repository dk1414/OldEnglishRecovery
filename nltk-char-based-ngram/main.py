import dill as pickle
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import models, api
from nltk.lm import *
from nltk.util import *

TEST = "corpi/test.txt"
TRAIN = "corpi/train.txt"

NGRAM_ORDER = 3

def train_model(model: api.LanguageModel, path_to_save: str):
    text = []
    with open(TRAIN, "r", encoding="utf-8") as train_file:
        for line in train_file:
            line = line.strip()
            text.append([*line])

    train, vocab = padded_everygram_pipeline(NGRAM_ORDER, text)
    lm = model(NGRAM_ORDER)
    lm.fit(train, vocab)

    with open(path_to_save, "wb") as fout:
        pickle.dump(lm, fout)

def perplexity(model: api.LanguageModel, path_to_file: str):
    gram = []
    with open(path_to_file, "r", encoding="utf-8") as test_file:
        for line in test_file:
            line = line.strip()
            chars = [*line]
            ngram = ngrams(chars, NGRAM_ORDER, pad_left=True, left_pad_symbol="<s>", pad_right=True, right_pad_symbol="</s>")
            gram.extend(ngram)
    #print(gram[0])

    return model.perplexity(gram)

if __name__ == '__main__':
    # train_model(KneserNeyInterpolated, 'models/kneserney-3gram.pkl')

    with open('models/kneserney-3gram.pkl', 'rb') as fin:
        model = pickle.load(fin)
        print(len(model.vocab))
        print("".join(model.generate(text_seed=[*'displacemen'])))
        print("train perplexity", perplexity(model, TRAIN))
        print("test  perplexity", perplexity(model, TEST))