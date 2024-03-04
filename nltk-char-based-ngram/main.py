import dill as pickle

import re

import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import models, api
from nltk.lm import *
from nltk.util import *

TEST = "../data/original/wiki.test.raw"
TRAIN = "../data/original/wiki.train.raw"

NGRAM_ORDER = 4

def tokenize(line: str) -> list[list[str]] | None:
    """Preprocesses data in the wikitext-2 corpus format for an nltk model. Tokenizes by character.
    Removes excess whitespace and recovers punctuation

    Args:
        line (str): The line to be processed

    Returns:
        list[list[str]]:  A list of sentences, where each sentence is tokenized by character
    """

    line = line.strip()
    if not line or line.startswith('='):
        return None
    sentences = re.split(r" \.", line)
    output = []
    for l in sentences:
        l = l.strip()
        if not l:
            continue
        l = re.sub(r" @(.)@ ", r"\1", l) # replace '@,@' '@-@' etc. with ',' '-'
        l = l.replace(" 's", "'s")
        l = l.replace('( ', '(')
        l = l.replace(' )', ')')
        l = l.replace(' ;', ';')
        l = l.replace(' ,', ',')
        l = l.replace(' " ', '')

        output.append([*l])

    return output

def train_model(model: api.LanguageModel, path_to_save: str) -> None:
    text = []
    with open(TRAIN, "r", encoding="utf-8") as train_file:
        for line in train_file:
            line = tokenize(line)
            if line is not None:
                text.extend(line)

    train, vocab = padded_everygram_pipeline(NGRAM_ORDER, text)
    print('training on this many sentences:\t', len(text))
    lm: api.LanguageModel = model(NGRAM_ORDER)
    lm.fit(train, vocab)

    with open(path_to_save, "wb") as fout:
        pickle.dump(lm, fout)

def perplexity(model: api.LanguageModel, path_to_file: str) -> float:
    gram = []
    with open(path_to_file, "r", encoding="utf-8") as test_file:
        for line in test_file:
            sentences = tokenize(line)
            if sentences is None:
                continue
            for s in sentences:
                ngram = ngrams(s, NGRAM_ORDER, pad_left=True, left_pad_symbol="<s>", pad_right=True, right_pad_symbol="</s>")
                gram.extend(ngram)
    #print(gram[0])

    return model.perplexity(gram)

if __name__ == '__main__':
    # train_model(KneserNeyInterpolated, 'models/kneserney-bigram.pkl')

    with open('models/kneserney-bigram.pkl', 'rb') as fin:
        model = pickle.load(fin)
        print('calculating pp on model of vocab size:\t', len(model.vocab))
        print("".join(model.generate(text_seed=[*'displacemen'])))
        # print("train perplexity", perplexity(model, TRAIN))
        # print("test  perplexity", perplexity(model, TEST))