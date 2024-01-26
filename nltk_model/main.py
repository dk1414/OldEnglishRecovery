import dill as pickle
import nltk
from nltk.lm import MLE, Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline


TRAIN_TXT = "wiki.train.txt"
NGRAM_ORDER = 2

def train_ngram_model(path_to_save: str, train_data: str, n: int):
    text = []
    with open(train_data, "r", encoding="utf-8") as train_file:
        for line in train_file:
            tokens = line.strip().split()  
            text.append(tokens)

    train, vocab = padded_everygram_pipeline(n, text)
    model = Laplace(n)  
    model.fit(train, vocab)

    with open(path_to_save, "wb") as fout:
        pickle.dump(model, fout)

if __name__ == '__main__':
    train_ngram_model('results.pkl', "wiki.train.raw", NGRAM_ORDER)
