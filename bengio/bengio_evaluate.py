import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
import multiprocessing
from tqdm import tqdm
import numpy as np
import pickle
from collections import defaultdict

EMBEDDING_DIM = 50
CONTEXT_SIZE = 4
BATCH_SIZE = 256
H = 200

device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_trigram_state_dict: dict = torch.load('./models/best_model_3.dat', map_location=torch.device(device))
best_fivegram_state_dict: dict = torch.load('./models/best_fivegram_model_3.dat', map_location=torch.device(device))
best_fivegram_char_state_dict: dict = torch.load('./models/best_char_fivegram_model_7.dat', map_location=torch.device(device))


def create_dataloader(path: str, use_char_map = False) -> tuple[DataLoader, int]:
    context_array = []
    target_array = []

    maps_path = './word2idmaps/5gram.pkl' if not use_char_map else './word2idmaps/5chargram.pkl'
    with open(maps_path, 'rb') as map_pkl:
        loaded_dict: dict = pickle.load(map_pkl)
        unknown_word_id = loaded_dict['<UNK>']
        # unknown tokens get mapped to id of <UNK>
        word_to_id_mappings = defaultdict(lambda: unknown_word_id, loaded_dict)

    vocab_size = len(word_to_id_mappings.keys())

    with open(path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc='Reading in corpus'):
            sentence = line.strip().split()
            if use_char_map:
                # turn list of word tokens into list of char tokens
                sentence = [*"".join(sentence)]
            for i, _ in enumerate(sentence):
                if i+CONTEXT_SIZE >= len(sentence):
                    break

                context_extract = [word_to_id_mappings[sentence[j]] for j in range(i, i+CONTEXT_SIZE)]
                target_extract = [word_to_id_mappings[sentence[i+CONTEXT_SIZE]]]

                context_array.append(context_extract)
                target_array.append(target_extract)

    context_array = np.array(context_array)
    target_array = np.array(target_array)
    total_set = np.concatenate((context_array, target_array), axis=1)
    
    available_workers = multiprocessing.cpu_count()
    loader = DataLoader(total_set, batch_size=BATCH_SIZE, num_workers=available_workers)

    return loader, vocab_size

# see https://abhinavcreed13.github.io/blog/bengio-trigram-nplm-using-pytorch/
class BengioModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, h):
        super(BengioModel, self).__init__()
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, h)
        self.linear2 = nn.Linear(h, vocab_size, bias = False)

    def forward(self, inputs):
        # compute x': concatenation of x1 and x2 embeddings
        embeds = self.embeddings(inputs).view((-1,self.context_size * self.embedding_dim))
        # compute h: tanh(W_1.x' + b)
        out = torch.tanh(self.linear1(embeds))
        # compute W_2.h
        out = self.linear2(out)
        # compute y: log_softmax(W_2.h)
        log_probs = F.log_softmax(out, dim=1)
        # return log probabilities
        # BATCH_SIZE x len(vocab)
        return log_probs

def log_probs_to_accuracy(log_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    predicted_labels = torch.argmax(log_probs, dim=1)
    accuracy = (predicted_labels == labels).float().mean()
    return accuracy

def evaluate(state_dict: dict, data_path: str, device, **kwargs) -> tuple[float, float]:
    dataloader, vocab_size = create_dataloader(data_path, **kwargs)

    model = BengioModel(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE, H)
    model.load_state_dict(state_dict)
    model.eval()

    loss = nn.NLLLoss()

    acc_sum, loss_sum = 0, 0
    count = 0
    with torch.no_grad():
        for i, data_tensor in tqdm(enumerate(dataloader), desc='Evaluating loss and accuracy'):
            context_tensor: torch.LongTensor = data_tensor[:,0:CONTEXT_SIZE].type(torch.LongTensor)
            target_tensor: torch.LongTensor = data_tensor[:,CONTEXT_SIZE].type(torch.LongTensor)
            context_tensor, target_tensor = context_tensor.to(device), target_tensor.to(device)

            log_probs: torch.Tensor = model(context_tensor)
            loss_val: torch.Tensor = loss(log_probs, target_tensor)
            loss_sum += loss_val.item()
            acc_sum += log_probs_to_accuracy(log_probs, target_tensor).item()
            count += 1
    
    acc_mean = acc_sum / count
    loss_mean = loss_sum / count
    return acc_mean, loss_mean



if __name__ == '__main__':
    # path = sys.argv[1]

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # state_dict: dict = torch.load(path, map_location=torch.device(device))

    acc, loss = evaluate(best_fivegram_char_state_dict, '../Srilm/newtestcorpus.txt', device, use_char_map=True)
    print("Accuracy: {}; Loss: {}".format(acc, loss))