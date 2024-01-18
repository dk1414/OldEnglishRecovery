import os
from tqdm import tqdm
from utils import SOS, EOS, UNK, process_sentence

# if procesing training set, update the vocabulary
def preprocess(path, training, vocab):
    # vocab = set()
    all_sents = []
    # loop through paragraphs, skip headers, and split into sentences
    file = open(path, "r", encoding="utf-8")
    lines = file.read().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        elif line.count('=') >= 2:
            continue
        elif line.startswith('='):
            continue
        else:
            words = process_sentence(line)
            if training:
                vocab.update(words)
            final_sent = [word if word in vocab else UNK for word in words]
            # add end token 
            # final_sent.append(EOS)
            all_sents.append(final_sent)
                      
    return all_sents, vocab

if __name__ == '__main__':

    newtraincorpus, t_vocab = preprocess('data/original/wiki.train.raw', True, set())
    newtestcorpus, vocab = preprocess('data/original/wiki.test.raw', False, t_vocab)
    
    with open('newtraincorpus.txt', "w", encoding="utf-8") as file1:
        file1.truncate(0)
        for sentence in newtraincorpus:
            sent = " ".join(sentence)
            file1.write(sent+'\n')       
                
    with open('newtestcorpus.txt', "w", encoding="utf-8") as file2:
        file2.truncate(0)
        for sentence in newtestcorpus:
            file2.write(" ".join(sentence)+'\n')        
            
    # print(file1])

