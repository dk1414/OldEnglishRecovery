import string

punc = string.punctuation

SOS = '<s>'
EOS = '</s>'
UNK = '<unk>'
YEAR = '<year>'
NUM = '<num>'
TOKENS = [SOS, EOS, UNK, YEAR, NUM]

EPS = 1e-8

def is_year(s):
    try:
        int(s)
        return int(s) in range(1400, 2100)
    except ValueError:
        return False


def is_number(s):
    return any(char.isdigit() for char in s)

def num(word):
    if is_number(word):
        return NUM
    else:
        return word


def year(word):
    if is_year(word):
        return YEAR
    else:
        return word

def isalpha(word):
    word_copy = word.replace("-", "")
    word_copy = word_copy.replace(".", "")
    return word_copy.isalpha()

def process_word(word, lower=True, mask_year=True, mask_nums=True, filter_punc=True):
    if filter_punc and word in punc:
        word = ""
    if lower:
        word = word.lower()
    if mask_year:
        word = year(word)  # Turns into <year> if applicable.
    if mask_nums:
        word = num(word)   # Turns into <num> if applicable.
    if word not in TOKENS and not isalpha(word):
        word = ""
    return word

def process_sentence(sentence):
    sentence = sentence.replace(" @-@ ", "-")
    sentence = sentence.replace(" @.@ ", ".")

    words = [process_word(word) for word in sentence.split() if word not in punc]

    return words
