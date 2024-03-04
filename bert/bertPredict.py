from transformers import AutoTokenizer, BertForMaskedLM
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, BertForMaskedLM
import torch
import torch.nn.functional as F

import sys
sys.path.append('c:\\Users\\declan\\Documents\\OldEnglishRecovery')
from LanguageModelHelper import LanguageModelHelper

import numpy as np
import pickle


class BertDataset(Dataset):
    def __init__(self, lm, mask_token, percent_masks_per_line=0.03):
        super(BertDataset, self).__init__()
        self.mask_token = mask_token
        self.unmasked = lm.get_dataset()
        self.masked, self.mask_indices = lm.get_masked_dataset(percent_masks_per_line)
        self.mask_indices = set([(i[0], i[1]) for i in self.mask_indices]) #only need word level indices not char level

        
    def __len__(self):
        return len(self.unmasked)
    
    def __getitem__(self, index):
        
        word_masked = self.unmasked[index].copy()
        mask_indices = []

        for j, word in enumerate(word_masked):
            if (index,j) in self.mask_indices:
                word_masked[j] = self.mask_token
                mask_indices.append(j)
 

        true = self.unmasked[index]
        char_masked = self.masked[index]

        return (word_masked, char_masked, true, mask_indices)
    
#calculates pseudo perplexity of a sentence by masking each token in a sentence - https://arxiv.org/abs/1910.14659

def perp_score(model, tokenizer, sentence):
    tensor_input = tokenizer.encode(sentence, return_tensors='pt').to(device)
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1, device=device).diag(1)[:-2]  # Move mask tensor to the same device
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)
    with torch.inference_mode():
        loss = model(masked_input, labels=labels).loss
    # Remove tensors from GPU memory
    del tensor_input, repeat_input, mask, masked_input, labels
    torch.cuda.empty_cache()
    return np.exp(loss.item())



#make predictions for each word with missing characters in dataset
def predict(model, tokenizer, lm, dataset):
    




    preds = []
    true = []
    m = [] #also keep track of masked words for later analysis

    progress = 0 #keep track of where we are for terminal print out
    for example in dataset:
        

        if progress % 1000 == 0:
            print(f'Starting line {progress}')
        
        progress += 1

        word_masked, char_masked, unmasked, mask_indices = example

        #mask_indices will contain all the indices of mask tokens, the number of mask tokens in a sentence is the number of preds we need to make
        for i in mask_indices:
            #get all candidate words for this word
            candidates = lm.find_candidates(char_masked[i])

            if len(candidates):
                lowest_perplexity = 10000000000 #dumb but idc
                pred = candidates[0] #just init to first one

                for candidate in candidates:
                    #fill in mask token with possible word
                    sentence = word_masked
                    sentence[i] = candidate
                    sentence = ' '.join(sentence)

                    #get pseudo perplexity
                    score = perp_score(model, tokenizer, sentence)

                    #keep track of best guess
                    if score < lowest_perplexity:
                        lowest_perplexity = score
                        pred = candidate
                #add best guess to preds      
                preds.append(pred)
                true.append(unmasked[i])
                m.append(char_masked[i])
            else:
                #dont think this should ever execute, there should always be at least one candidate
                #otherwise something weird is going on
                raise Exception(f"No candidates found for {char_masked[i]}")
    
    return preds, true, m



def count_char_occurrences(string, char):
    count = 0
    for c in string:
        if c == char:
            count += 1
    return count

def analyze_results(preds, true, m, masked_char = '¿'):

    total_correct = 0

    correct_by_num_missing = dict()
    correct_by_char_count = dict()
    total_by_char_count = dict()
    total_by_num_missing = dict()

    for i in range(len(preds)):
        pred = preds[i]
        label = true[i]
        masked = m[i]

        num_masked_chars = count_char_occurrences(masked, masked_char)

        if pred == label:
            total_correct += 1
            
            if num_masked_chars in correct_by_num_missing:
                correct_by_num_missing[num_masked_chars] += 1
            else:
                correct_by_num_missing[num_masked_chars] = 1
            
            if len(label) in correct_by_char_count:
                correct_by_char_count[len(label)] += 1
            else:
                correct_by_char_count[len(label)] = 1
        
        if num_masked_chars in total_by_num_missing:
            total_by_num_missing[num_masked_chars] += 1
        else:
            total_by_num_missing[num_masked_chars] = 1

        if len(label) in total_by_char_count:
            total_by_char_count[len(label)] += 1
        else:
            total_by_char_count[len(label)] = 1

    print(f'Total acc: {total_correct / len(preds)}')

    for key in correct_by_num_missing:
        print(f'Acc for words with {key} chars missing: {correct_by_num_missing[key] / total_by_num_missing[key]}')

    for key in correct_by_char_count:
        print(f'Acc for words with {key} chars in total: {correct_by_char_count[key] / total_by_char_count[key]}')
    
    print(f'# words with x chars missing: {total_by_num_missing}')




if __name__ == "__main__":

    #running this file 4 times for whole dataset cause my gpu runs out of memory and I have to restart the kernel

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = BertForMaskedLM.from_pretrained("bert-base-uncased")

    #load text and create vocabulary. Just going to say words need to be present at least 5 times to be in vocab
    lm = LanguageModelHelper(file_path='../data/newtraincorpus1-4.txt', min_freq=5, unk_token = tokenizer.unk_token)

    #this is gonna be pretty slow without cuda. Probably like 3 min for 5 lines. With cuda only about a second per line
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert.to(device)
    bert.eval()




    #mask x% of characters in each line of text
    dataset = BertDataset(lm,tokenizer.mask_token, percent_masks_per_line=0.2)

    #make preds
    output = predict(bert, tokenizer, lm, dataset)

    #get some stats
    analyze_results(output[0], output[1], output[2])

    #organize
    results = {'preds': output[0],
               'true': output[1],
               'masked': output[2]}
    
    #save results to file
    save_path = 'bert_results_20_percent.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    



