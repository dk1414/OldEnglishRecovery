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
    word_copy = word_copy.replace("'", "")
    return word_copy.isalpha()

def process_word(word, lower=True, mask_year=True, mask_nums=True, filter_punc=True):
    """Performs relevant manipulations at the word level

    Args:
        word (str): word to be manipulated
        lower (bool, optional): Convert word to lowercase. Defaults to True.
        mask_year (bool, optional): Turn all years into a special YEAR token. Defaults to True.
        mask_nums (bool, optional): Turn all words containing digits into a special NUM token. Defaults to True.
        filter_punc (bool, optional): Removes all punctuation. Defaults to True.

    Returns:
        str: The manpulated word
    """
    if filter_punc and word in punc:
        word = ""
    if lower:
        word = word.lower()
    if mask_year:
        word = year(word)  # Turns into <year> if applicable.
    if mask_nums:
        word = num(word)   # Turns into <num> if applicable.
    # if word not in TOKENS and not isalpha(word):
    #     word = ""
    return word

def process_sentence(sentence, **kwargs):
    sentence = sentence.replace(" @-@ ", "-")
    sentence = sentence.replace(" @.@ ", ".")
    sentence = sentence.replace(" @,@ ", ",")
    sentence = sentence.replace(" 's", "'s")

    words = [process_word(word, **kwargs) for word in sentence.split() if word not in punc]

    return words
