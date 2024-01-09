import os
from tqdm import tqdm
import string
from utils import SOS, EOS, UNK, process


punc = string.punctuation


# if procesing training set, update the vocabulary
def preprocess(path,training,vocab):
    # vocab = set()
    all_sents = []
    # loop through paragraphs, skip headers, and split into sentences
    file = open(path, "r")
    for line in file.readlines():
        end_sent = False
        line = line.strip()
        if not line:
            continue
        elif line.count('=') >= 2:
            continue
        elif line.startswith('='):
            continue
        else:
            # sentence = [SOS]
            sentence = []                
            for word in line.split():
                # dont include puncuation
                if word not in punc:
                    p_word = process(word, lower=True)
                    sentence.append(p_word)
                
            # update vocab if training set
            if training:
                vocab.update(sentence)
            final_sent = [word if word in vocab else UNK for word in sentence]
            # add end token 
            # final_sent.append(EOS)
            all_sents.append(final_sent)
            sentence = []
                    
            
    return all_sents,vocab




if __name__ == '__main__':

    newtraincorpus,t_vocab = preprocess('data/original/wiki.train.raw', True,set())
    newtestcorpus,vocab = preprocess('data/original/wiki.test.raw',False,t_vocab)
    
    with open('newtraincorpus.txt', "w") as file1:
        file1.truncate(0)
        for sentence in newtraincorpus:
            sent = " ".join(sentence)
            file1.write(sent)       
                
    with open('newtestcorpus.txt', "w") as file2:
        file2.truncate(0)
        for sentence in newtraincorpus:
            file2.write(" ".join(sentence))        
            
    print(newtraincorpus[:100])

