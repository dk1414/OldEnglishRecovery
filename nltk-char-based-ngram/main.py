import dill as pickle
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.util import *

TEST = "corpi/test.txt"
TRAIN = "corpi/train.txt"

NGRAM_ORDER = 2

def train_model(path_to_save: str):
    text = []
    with open(path_to_save, "r", encoding="utf-8") as train_file:
        for line in train_file:
            line = line.strip()
            text.append([*line])

    train, vocab = padded_everygram_pipeline(NGRAM_ORDER, text)
    lm = MLE(NGRAM_ORDER)
    lm.fit(train, vocab)

    with open(path_to_save, "wb") as fout:
        pickle.dump(lm, fout)

def perplexity(model: MLE, path_to_file: str):
    bigram = []
    with open(path_to_file, "r", encoding="utf-8") as test_file:
        for line in test_file:
            line = line.strip()
            chars = pad_sequence([*line], n=NGRAM_ORDER, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')
            bigram.append(list(bigrams(chars)))
    print(bigram[0])
    return model.perplexity(bigram)

if __name__ == '__main__':
    train_model(TRAIN)

    with open('models/mle-bigram.pkl', 'rb') as fin:
        model = pickle.load(fin)
        print("".join(model.generate(30)))
        print(perplexity(model, TEST))