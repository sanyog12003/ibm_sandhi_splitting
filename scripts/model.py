import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.nn.functional as F
from torcheval.metrics.text import Perplexity
import os, glob, sys
import numpy as np
import pandas as pd
import random
import math
from sklearn.model_selection import train_test_split
import gc
import pylcs

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dout):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.encoder_module = nn.LSTM(embed_dim, hidden_dim, num_layers=2, bidirectional=True, dropout = dout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, inputs):
        embs = self.embed(inputs)
        enc_outs, hidden = self.encoder_module(embs)
        h_n_2 = torch.cat((hidden[0][-2, :, :], hidden[0][-1, :, :]), dim = 1)
        h_n_1 = torch.cat((hidden[0][0, :, :], hidden[0][1, :, :]), dim = 1)
        h_n = self.fc1(torch.stack((h_n_1, h_n_2), dim = 0))
        c_n_2 = torch.cat((hidden[1][-2, :, :], hidden[1][-1, :, :]), dim=1)
        c_n_1 = torch.cat((hidden[1][0, :, :], hidden[1][1, :, :]), dim = 1)
        c_n = self.fc2(torch.stack((c_n_1, c_n_2), dim = 0))
        enc_outs = (enc_outs[:, :, :self.hidden_dim] + enc_outs[:, :, self.hidden_dim:])
        return enc_outs, (h_n, c_n)

class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim, method = "general"):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        
        if method == 'dot':
            pass
        elif method == 'general':
            self.w = nn.Linear(hidden_dim, hidden_dim)
        elif method == 'concat':
            self.w = nn.Linear(hidden_dim*2, hidden_dim)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_dim))
        elif method == 'bahdanau':
            """
            self.q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            """
            self.q_linear = nn.Linear(embed_dim, hidden_dim, bias = False)
            self.k_linear = nn.Linear(hidden_dim, hidden_dim, bias = False)
            self.v_linear = nn.Linear(hidden_dim, hidden_dim, bias = False)
        else:
            raise NotImplementedError
    
    def forward(self, emb, enc_outs, mask = None):
        if self.method == 'dot':
            attn_energies = self.dot(enc_outs, enc_outs)
        elif self.method == 'general':
            attn_energies = self.general(enc_outs, enc_outs)
        elif self.method == 'concat':
            attn_energies = self.concat(enc_outs, enc_outs)
        elif self.method == 'bahdanau':
            attn_energies = self.bahdanau(emb, enc_outs)
        if mask is not None:
            if(self.method == 'bahdanau'):
                attn_energies = attn_energies.squeeze(1).transpose(1, 0)
                attn_energies = attn_energies.masked_fill(mask, -float('inf'))
            else:
                attn_energies = attn_energies.masked_fill(mask, -float('inf'))
        attn_energies = F.softmax(attn_energies, -1).transpose(1, 0).unsqueeze(1)
        enc_outs = enc_outs.permute(1, 0, 2)
        if self.method != 'bahdanau':
            weighted = torch.bmm(attn_energies, enc_outs)
        else:
            weighted = torch.bmm(attn_energies, self.v_linear(enc_outs))
        return weighted.permute(1, 0, 2)

    def dot(self, emb, enc_outs):
        return torch.sum(emb*enc_outs, dim=2)

    def general(self, emb, enc_outs):
        energy = self.w(enc_outs)
        return torch.sum(emb*energy, dim=2)

    def concat(self, emb, enc_outs):
        emb = emb.expand(enc_outs.shape[0], -1, -1)
        energy = torch.cat((emb, enc_outs), 2)
        return torch.sum(self.v * self.w(energy).tanh(), dim=2)

    def bahdanau(self, q, k): 
        q = self.q_linear(q).permute(1, 0, 2)
        k = self.k_linear(k).permute(1, 0, 2)
        out = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        return out

class Decoder(nn.Module):
    def __init__(self, vocab_size, tgt_vocab_size, embed_dim, hidden_dim, dout, attention):
        super(Decoder, self).__init__()
        self.attention = attention
        self.vocab_size = vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim)
        self.decoder_module = nn.LSTM(3 * hidden_dim + embed_dim, hidden_dim, num_layers = 2, dropout = dout)
        self.fc_out = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, inp, encoder_output, hidden = None, mask = None, flag = True):
        inp = inp.unsqueeze(0)
        embeddings = self.embed(inp)
        weighted = self.attention(embeddings, encoder_output, mask)
        if(flag):
            output = torch.cat((embeddings, weighted, weighted), dim = 2)
            output, hidden = self.decoder_module(output)
        else:
            output = torch.cat((embeddings, weighted, hidden[0][0].unsqueeze(0), hidden[0][1].unsqueeze(0)), dim = 2)
            output, hidden = self.decoder_module(output, hidden)
        prediction = self.fc_out(output)
        return F.log_softmax(prediction.squeeze(0), -1), hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, location_decoder, split_decoder, device):
        super().__init__()
        self.encoder = encoder
        self.location_decoder = location_decoder
        self.split_decoder = split_decoder
        self.device = device

    def forward(self, src, tgt, mask = None, dataset = "train", phase = "location_decoder"):
        batch_size = src.shape[1]
        tgt_len = tgt.shape[0]
        outputs = []

        encoder_output, hidden = self.encoder(src)
        inp = tgt[0]
        flag = False
        for i in range(1, tgt_len):
            if(phase == "location_decoder"):
                output, hidden = self.location_decoder(inp, encoder_output, hidden, mask, flag)
            else:
                output, hidden = self.split_decoder(inp, encoder_output, hidden, mask, flag)
            flag = False
            outputs.append(output)
            top_guess = output.argmax(1)
            if(dataset == "train"):
                inp = tgt[i]
            else:
                inp = top_guess
        return torch.stack(outputs)

with open("../dataset/iitd_dataset_single_split.txt", "r") as fread:
    lines = fread.readlines()

dataset = []
for line in lines:
    comps = line.split(',')
    dataset.append([comps[0].strip(), comps[1].split('\n')[0].strip()])

train_dataset, test_dataset = train_test_split(dataset, test_size = 0.2, random_state = 0)
train_dataset, valid_dataset = train_test_split(train_dataset, test_size = 0.1, random_state = 0)

X_train = []
Y_train_location = []
Y_train_split = []
char_vocab = set()

def getSplitLocationArr(input, output):
    location_arr = np.zeros(len(input), dtype=int)
    res = pylcs.lcs_sequence_idx(output, input)
    prev_val = res[0]
    if(res[0] == -1):
        location_arr[0] = 1
    prev = res[0]
    for i in range(1, len(res)):
        if(res[i] == -1):
            if(i + 1 == len(res)):
                if(prev_val + 1 == len(input)):
                    location_arr[prev_val] = 1
                else:
                    for j in range(prev_val + 1, len(input)):
                        location_arr[j] = 1
                    """
                    location_arr[prev_val + 1] = 1
                    """
            if(prev == -1):
                continue
            else:
                prev_val = prev
        else:
            if(prev == -1):
                if(prev_val + 1 == res[i]):
                   location_arr[prev_val] = 1
                else: 
                    for j in range(prev_val + 1, res[i]):
                        location_arr[j] = 1
                    """
                    location_arr[prev_val + 1] = 1
                    """
            prev_val = res[i]
        prev = res[i]
    return location_arr

train_reject_counter = 0
train_size_reject_counter = 0
folder_name="iitd_single_split"

with open("../dataset/" + folder_name + "/train.src", "w") as fwrite_src, open("../dataset/" + folder_name + "/train.tgt", "w") as fwrite_tgt, open("../dataset/" + folder_name + "/train.tgt_location", "w") as fwrite_tgt_location:
    for data in train_dataset:
        input = data[0]
        if(len(input) == 0 or len(input) > 150):
            train_size_reject_counter = train_size_reject_counter + 1
            continue
        if(len(input) - 3 >= len(data[1])):
            train_reject_counter = train_reject_counter + 1
            continue
        fwrite_src.write(data[0] + "\n")
        fwrite_tgt.write(data[1] + "\n")
        output = '<' + data[1] + '>'
        location_arr = getSplitLocationArr(data[0], data[1])
        fwrite_tgt_location.write(''.join(map(str, location_arr)) + "\n")
        location_arr = '<' + ''.join(map(str, location_arr)) + '>'
        X_train.append(input)
        Y_train_location.append(location_arr)
        Y_train_split.append(output)
        for char in input:
            if char not in char_vocab:
                char_vocab.add(char)
        for char in output:
            if char not in char_vocab:
                char_vocab.add(char)

#char_vocab.add('0')
#char_vocab.add('1')
char_vocab = sorted(list(char_vocab))
char_vocab.remove('>')
char_vocab.remove('<')
char_vocab.insert(0, '?')
char_vocab.insert(0, '>')
char_vocab.insert(0, '<')
char_vocab.insert(0, '*')
#char_vocab[char_vocab.index('0')] = 0
#char_vocab[char_vocab.index('1')] = 1

vocab_size = len(char_vocab)
token_index = dict([(char, i) for i, char in enumerate(char_vocab)])
print(token_index)
tgt_location_vocab = list()
tgt_location_vocab.append('*')
tgt_location_vocab.append('<')
tgt_location_vocab.append('>')
tgt_location_vocab.append('?')
tgt_location_vocab.append('0')
tgt_location_vocab.append('1')
tgt_location_vocab_size = len(tgt_location_vocab)
location_token_index = dict([(char, i) for i, char in enumerate(tgt_location_vocab)])
print(location_token_index)

def tokenizeDataset(X_train, Y_train_location, Y_train_split):
    X_train_tokenized = []
    Y_train_location_tokenized = []
    Y_train_split_tokenized = []
    for input, output_location, output_split in zip(X_train, Y_train_location, Y_train_split):
        X_train_tokenized.append(list(input))
        Y_train_location_tokenized.append(list(output_location))
        Y_train_split_tokenized.append(list(output_split))
    return X_train_tokenized, Y_train_location_tokenized, Y_train_split_tokenized

X_train, Y_train_location, Y_train_split = tokenizeDataset(X_train, Y_train_location, Y_train_split)

X_valid = []
Y_valid_location = []
Y_valid_split = []

valid_reject_counter = 0
valid_size_reject_counter = 0
with open("../dataset/" + folder_name + "/valid.src", "w") as fwrite_src, open("../dataset/" + folder_name + "/valid.tgt", "w") as fwrite_tgt, open("../dataset/" + folder_name + "/valid.tgt_location", "w") as fwrite_tgt_location:
    for data in valid_dataset:
        input = data[0]
        if(len(input) == 0 or len(input) > 150):
            valid_size_reject_counter = valid_size_reject_counter + 1
            continue
        if(len(input) - 3 >= len(data[1])):
            valid_reject_counter = valid_reject_counter + 1
            continue
        fwrite_src.write(data[0] + "\n")
        fwrite_tgt.write(data[1] + "\n")
        output = '<' + data[1] + '>'
        location_arr = getSplitLocationArr(data[0], data[1])
        fwrite_tgt_location.write(''.join(map(str, location_arr)) + "\n")
        location_arr = '<' + ''.join(map(str, location_arr)) + '>'
        X_valid.append(input)
        Y_valid_location.append(location_arr)
        Y_valid_split.append(output)

X_valid, Y_valid_location, Y_valid_split = tokenizeDataset(X_valid, Y_valid_location, Y_valid_split)

X_test = []
Y_test_location = []
Y_test_split = []

test_reject_counter = 0
test_size_reject_counter = 0
with open("../dataset/" + folder_name + "/test.src", "w") as fwrite_src, open("../dataset/" + folder_name + "/test.tgt", "w") as fwrite_tgt, open("../dataset/" + folder_name + "/test.tgt_location", "w") as fwrite_tgt_location:
    for data in test_dataset:
        input = data[0]
        if(len(input) == 0 or len(input) > 150):
            test_size_reject_counter = test_size_reject_counter + 1
            continue
        if(len(input) - 3 >= len(data[1])):
            test_reject_counter = test_reject_counter + 1
            continue
        fwrite_src.write(data[0] + "\n")
        fwrite_tgt.write(data[1] + "\n")
        output = '<' + data[1] + '>'
        location_arr = getSplitLocationArr(data[0], data[1])
        fwrite_tgt_location.write(''.join(map(str, location_arr)) + "\n")
        location_arr = '<' + ''.join(map(str, location_arr)) + '>'
        X_test.append(input)
        Y_test_location.append(location_arr)
        Y_test_split.append(output)

X_test, Y_test_location, Y_test_split = tokenizeDataset(X_test, Y_test_location, Y_test_split)

class customDataset(Dataset):
    def __init__(self, X, Y_location, Y_split):
        super(customDataset, self).__init__()
        self.X = X
        self.Y_location = Y_location
        self.Y_split = Y_split

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y_location = self.Y_location[idx]
        y_split = self.Y_split[idx]
        x = list((pd.Series(x)).map(token_index))
        y_location = list((pd.Series(y_location)).map(location_token_index))
        y_split = list((pd.Series(y_split)).map(token_index))
        for i in range(len(y_split)):
            if math.isnan(y_split[i]):
                y_split[i] = token_index['?']
        return torch.tensor(x), torch.tensor(y_location), torch.tensor(y_split)

train_dataset = customDataset(X_train, Y_train_location, Y_train_split)
valid_dataset = customDataset(X_valid, Y_valid_location, Y_valid_split)
test_dataset = customDataset(X_test, Y_test_location, Y_test_split)

class MyCollate:
    def __init__(self, pad_idx, tgt_pad_idx):
        self.pad_idx = pad_idx
        self.tgt_pad_idx = tgt_pad_idx
    
    def __call__(self, batch):
        batch_size = len(batch)
        X = [item[0] for item in batch]
        Y_location = [item[1] for item in batch]
        Y_split = [item[2] for item in batch] 
        X = pad_sequence(X, batch_first=False, padding_value = self.pad_idx)
        Y_location = pad_sequence(Y_location, batch_first=False, padding_value = self.tgt_pad_idx)
        Y_split = pad_sequence(Y_split, batch_first=False, padding_value = self.pad_idx)
        return X, Y_location, Y_split

dout = 0.3
hidden_dim = 512
embed_dim = 128
batch_size = 64

train_dataloader = DataLoader(train_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, collate_fn = MyCollate(pad_idx = token_index['*'], tgt_pad_idx = location_token_index['*']))
valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, collate_fn = MyCollate(pad_idx = token_index['*'], tgt_pad_idx = location_token_index['*']))
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, collate_fn = MyCollate(pad_idx = token_index['*'], tgt_pad_idx = location_token_index['*']))

method = "bahdanau"
attn = Attention(embed_dim, hidden_dim, method)
encoder = Encoder(vocab_size, embed_dim, hidden_dim, dout)
location_decoder = Decoder(vocab_size, tgt_location_vocab_size, embed_dim, hidden_dim, dout, attn)
split_decoder = Decoder(vocab_size, vocab_size, embed_dim, hidden_dim, dout, attn)

model = Seq2Seq(encoder, location_decoder, split_decoder, device)
model.to(device)
optimizer = optim.SGD(model.parameters(), lr = 1.0)
location_criterion = nn.NLLLoss(ignore_index = location_token_index['*'])
split_criterion = nn.NLLLoss(ignore_index = token_index['*'])
metric1 = Perplexity(ignore_index = location_token_index['*'])
metric2 = Perplexity(ignore_index = location_token_index['*'])
metric3 = Perplexity(ignore_index = token_index['*'])
metric4 = Perplexity(ignore_index = token_index['*'])

print("number of parameters : {}".format(sum(p.numel() for p in model.parameters())))
print("dataset : train : {} || valid : {} || test : {}".format(len(train_dataset), len(valid_dataset), len(test_dataset)))
print("train : size_reject_counter : {} || reject_counter : {}".format(train_size_reject_counter, train_reject_counter))
print("valid : size_reject_counter : {} || reject_counter : {}".format(valid_size_reject_counter, valid_reject_counter))
print("test : size_reject_counter : {} || reject_counter : {}".format(test_size_reject_counter, test_reject_counter))
print(model)

def getAccuracyCounter(preds, expec, phase):
    counter = 0
    preds = preds.transpose(1, 0)
    expec = expec.transpose(1, 0)
    for i in range(preds.shape[0]):
        if(phase == "location_decoder"):
            index = (expec[i] == torch.tensor(location_token_index['>'])).nonzero().flatten().tolist()[0]
        else:
            index = (expec[i] == torch.tensor(token_index['>'])).nonzero().flatten().tolist()[0]
        if(torch.equal(preds[i, :(index + 1)], expec[i, :(index + 1)])):
            counter = counter + 1
    return counter;

def generateSourceMask(x):
    mask = None
    mask = (x != token_index["*"]) + 0.0
    mask = ~mask.type(torch.bool)
    return mask

def train(model, loader, optimizer, criterion, phase):
    model.train()
    epoch_loss = 0
    counter = 0
    accuracy_counter = 0
    samples = 0
    if(phase == "location_decoder"):
        for i, (x, y_location, y_split) in enumerate(loader):
            optimizer.zero_grad()
            x = x.to(device).long()
            mask = generateSourceMask(x).to(device)
            y_location = y_location.to(device).long()
            expected_location = y_location[1:]
            y_split = y_split
            output = model(x, y_location, mask, "train", phase)
            predicted_location = torch.argmax(output, dim = 2)
            if(i % 250 == 0):
                print("expectation : ", expected_location[:, 0])
                print("prediction : ", predicted_location[:, 0])
            output = output.view(-1, output.shape[2])
            y_location = y_location[1:].view(-1)
            y_location = y_location.cpu()
            output = output.cpu()
            loss = criterion(output, y_location)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            epoch_loss = epoch_loss + loss.item()
            counter = counter + 1
            accuracy_counter = accuracy_counter + getAccuracyCounter(predicted_location, expected_location, phase)
            samples = samples + x.shape[1]
    else:
        for i, (x, y_location, y_split) in enumerate(loader):
            optimizer.zero_grad()
            x = x.to(device).long()
            mask = generateSourceMask(x).to(device)
            y_split = y_split.to(device).long()
            y_location = y_location
            output = model(x, y_split, mask, "train", phase)
            expected_split = y_split[1:]
            predicted_split = torch.argmax(output, dim = 2)
            if(i % 250 == 0):
                print("expectation : ", expected_split[:, 0])
                print("prediction : ", predicted_split[:, 0])
            output = output.view(-1, output.shape[2])
            y_split = y_split[1:].view(-1).cpu()
            output = output.cpu()
            loss = criterion(output, y_split)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            epoch_loss = epoch_loss + loss.item()
            counter = counter + 1
            accuracy_counter = accuracy_counter + getAccuracyCounter(predicted_split, expected_split, phase)
            samples = samples + x.shape[1]
    return epoch_loss/counter, accuracy_counter/samples

def evaluate(model, loader, criterion, first_metric, second_metric, dataset, phase):
    model.eval()
    epoch_loss = 0
    counter = 0
    accuracy_counter = 0
    samples = 0
    first_metric.reset()
    if(phase == "location_decoder"):
        for i, (x, y_location, y_split) in enumerate(loader):
            x = x.to(device).long()
            y_location = y_location.to(device).long()
            mask = generateSourceMask(x).to(device)
            output = model(x, y_location, mask, dataset, phase)
            output_dim = output.shape[-1]
            expected_location = y_location[1:]
            perp_location = output
            predicted_location = torch.argmax(output, dim = 2)
            if(i % 80 == 0):
                print("expectation : ", expected_location[:, 0])
                print("prediction : ", predicted_location[:, 0])
            output = output.view(-1, output_dim)
            y_location = y_location[1:].view(-1)
            y_location = y_location.cpu()
            output = output.cpu()
            loss = criterion(output, y_location)
            epoch_loss += loss.item()
            counter = counter + 1
            accuracy_counter = accuracy_counter + getAccuracyCounter(predicted_location, expected_location, phase)
            if(dataset == "valid"):
                perp_location = perp_location.cpu()
                expected_location = expected_location.cpu()
                second_metric.update(perp_location, expected_location)
                first_metric.merge_state([second_metric])
                second_metric.reset()
            samples = samples + x.shape[1]
    else:
        for i, (x, y_location, y_split) in enumerate(loader):
            x = x.to(device).long()
            y_location = y_location
            mask = generateSourceMask(x).to(device)
            y_split = y_split.to(device).long()
            output = model(x, y_split, mask, dataset, phase)
            expected_split = y_split[1:]
            perp_split = output
            predicted_split = torch.argmax(output, dim = 2)
            if(i % 80 == 0):
                print("expectation : ", expected_split[:, 0])
                print("prediction : ", predicted_split[:, 0])
            output = output.view(-1, output.shape[-1])
            y_split = y_split[1:].view(-1).cpu()
            output = output.cpu()
            loss = criterion(output, y_split)
            epoch_loss += loss.item()
            counter = counter + 1
            accuracy_counter = accuracy_counter + getAccuracyCounter(predicted_split, expected_split, phase)
            if(dataset == "valid"):
                perp_split = perp_split.cpu()
                expected_split = expected_split.cpu()
                second_metric.update(perp_split, expected_split)
                first_metric.merge_state([second_metric])
                second_metric.reset()
            samples = samples + x.shape[1]
    if(dataset == "valid"):
        return epoch_loss/counter, accuracy_counter/samples, first_metric.compute().item()
    else:
        return epoch_loss/counter, accuracy_counter/samples

"""
for i, (x, y_location, y_split) in enumerate(test_dataloader):
    seq_len, batch_size = y_split.shape
    for j in range(seq_len):
        for k in range(batch_size):
            if(y_split[j, k] < 0):
                print(j, k, y_split[j, k])
"""

N_EPOCHS = 20
min_location_perp_metric = float('inf')
min_split_perp_metric = float('inf')
for epoch in range(N_EPOCHS):
    if(epoch < (N_EPOCHS/2)):
        phase = "location_decoder"
        criterion = location_criterion
        first_metric = metric1
        second_metric = metric2
        if(epoch == 0):
            min_perp_metric = min_location_perp_metric
    else:
        phase = "split_decoder"
        criterion = split_criterion
        first_metric = metric3
        second_metric = metric4
        if(epoch == N_EPOCHS/2):
            optimizer.param_groups[0]['lr'] = 1.0
            min_perp_metric = min_split_perp_metric
    train_loss, train_accuracy = train(model, train_dataloader, optimizer, criterion, phase)
    print("training completed .... ")
    valid_loss, valid_accuracy, perp_metric = evaluate(model, valid_dataloader, criterion, first_metric, second_metric, "valid", phase)
    if(min_perp_metric < perp_metric):
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*0.5;
    else:
        min_perp_metric = perp_metric
    print("Epoch : {} || train : loss : {} || accuracy : {} || valid:  loss : {} || min perplexity : {} || perplexity : {} || accuracy : {}".format(epoch, train_loss, train_accuracy * 100, valid_loss, min_perp_metric, perp_metric, valid_accuracy * 100))
    if(epoch == (N_EPOCHS/2) - 1):
        test_location_loss, test_location_accuracy = evaluate(model, test_dataloader, location_criterion, metric1, metric2, "test", "location_decoder")
        print("test location:  loss : {} || accuracy : {}".format(test_location_loss, test_location_accuracy * 100)) 
        print("\n")
        print("\n")
        print("Starting split decoder training")
    if(epoch == (N_EPOCHS - 1)):
        test_split_loss, test_split_accuracy = evaluate(model, test_dataloader, split_criterion, metric3, metric4, "test", "split_decoder")
        print("test split : loss : {} || accuracy : {}".format(test_split_loss, test_split_accuracy * 100))

