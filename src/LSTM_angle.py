# coding: utf-8
import pandas as pd
import numpy as np
import re
import logging
import torch
from torchtext import data
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pickle
import io
import time
import sys
import datahelper

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

torch.manual_seed(42)

test_mode = 0  # 0 for train+test 1 for test
device = 0 # 0 for gpu, -1 for cpu

batch_size = 32
embedding_dim = 300
hidden_dim = 300
out_dim = 1

epochs = 1
print_every = 20
bidirectional = False



print('Reading data..')
ID = data.Field(sequential=False, batch_first=True, use_vocab=False)
TEXT = data.Field(sequential=True, lower=True, eos_token='<EOS>', init_token='<BOS>',
                  pad_token='<PAD>', fix_length=None, batch_first=True, use_vocab=True)
LABEL = data.Field(sequential=False, batch_first=True, use_vocab=False)

train = data.TabularDataset(
        path='../data/quora/train.tsv', format='tsv',
        fields=[('Label', LABEL), ('Text1', TEXT), ('Text2', TEXT), ('Id', ID)], skip_header=True)
valid = data.TabularDataset(
        path='../data/quora/dev.tsv', format='tsv',
        fields=[('Label', LABEL), ('Text1', TEXT), ('Text2', TEXT), ('Id', ID)], skip_header=True)
test = data.TabularDataset(
        path='../data/quora/test.tsv', format='tsv',
        fields=[('Label', LABEL), ('Text1', TEXT), ('Text2', TEXT), ('Id', ID)], skip_header=True)

TEXT.build_vocab(train)
print('Building vocabulary Finished.')


word_matrix = datahelper.wordlist_to_matrix("../data/quora/wordvec.txt", TEXT.vocab.itos, device, embedding_dim)

train_iter = data.BucketIterator(dataset=train, batch_size=batch_size, sort_key=lambda x: len(x.Text1) + len(x.Text2), device=device, shuffle=True, repeat=False)
valid_iter = data.Iterator(dataset=valid, batch_size=batch_size, device=device, shuffle=False, repeat=False)
test_iter = data.Iterator(dataset=test, batch_size=batch_size, device=device, shuffle=False, repeat=False)


train_dl = datahelper.BatchWrapper(train_iter, ["Text1", "Text2", "Label"])
valid_dl = datahelper.BatchWrapper(valid_iter, ["Text1", "Text2", "Label"])
test_dl = datahelper.BatchWrapper(test_iter, ["Text1", "Text2", "Label"])
print('Reading data done.')

train_dl = datahelper.BatchWrapper(train_iter, ["Text1", "Text2", "Label"])
valid_dl = datahelper.BatchWrapper(valid_iter, ["Text1", "Text2", "Label"])
test_dl = datahelper.BatchWrapper(test_iter, ["Text1", "Text2", "Label"])
print('Reading data done.')
def predict_on(model, data_dl, loss_func, device ,model_state_path=None):
    if model_state_path:
        model.load_state_dict(torch.load(model_state_path))
        print('Start predicting...')

    model.eval()
    res_list = []
    label_list = []
    loss = 0

    
    for text1, text2, label in data_dl:
        hidden_init = model.init_hidden(label.size()[0], device)
        y_pred = model(text1, text2, hidden_init)
        loss += loss_func(y_pred, label).data.cpu()
        y_pred = y_pred.data.max(1)[1].cpu().numpy()
        res_list.extend(y_pred)
        label_list.extend(label.data.cpu().numpy())
        
    acc = accuracy_score(res_list, label_list)
    Precision = precision_score(res_list, label_list)
    Recall = recall_score(res_list, label_list)
    F1 = f1_score(res_list, label_list)

    with open("res_list.txt", 'w') as fw:
        for item in res_list:
            fw.write('{}\n'.format(item))
    
    return loss, (acc, Precision, Recall, F1)


class LSTM_angel(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size, wordvec_matrix, bidirectional):
        super(LSTM_angel, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        self.dist = nn.PairwiseDistance(2)
    
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.word_embedding.weight.data.copy_(wordvec_matrix)
        self.word_embedding.weight.requires_grad = False
        
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim//2 if bidirectional else hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.lstm2 = nn.LSTM(embedding_dim, hidden_dim//2 if bidirectional else hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.linear1 = nn.Linear(2, 200)
        self.linear2 = nn.Linear(200, 2)
        
    def forward(self, text1, text2, hidden_init) :
        text1_word_embedding = self.word_embedding(text1)
        text2_word_embedding = self.word_embedding(text2)
#         print(text1)
#         print(text1_word_embedding[0:3])
        text1_seq_embedding = self.lstm_embedding(self.lstm1, text1_word_embedding, hidden_init)
        text2_seq_embedding = self.lstm_embedding(self.lstm2, text2_word_embedding, hidden_init)
#         print("------")
#         print(text1_seq_embedding[0][0:10])
#         print(text2_seq_embedding[0][0:10])
#         print("------")
        dot_value = torch.bmm(text1_seq_embedding.view(text1.size()[0], 1, self.hidden_dim), text2_seq_embedding.view(text1.size()[0], self.hidden_dim, 1))
        dot_value = dot_value.view(text1.size()[0], 1)
        dist_value = self.dist(text1_seq_embedding, text2_seq_embedding).view(text1.size()[0], 1)
#         print(dot_value)
#         print(dist_value)
#         feature_vec = torch.cat((text1_seq_embedding,text2_seq_embedding), dim=1)
        feature_vec = torch.cat((dot_value,dist_value), dim=1)
#         print(feature_vec)
#         sys.exit()
        linearout_1 = self.linear1(feature_vec)
        linearout_1 = F.relu(linearout_1)
        linearout_2 = self.linear2(linearout_1)
        return F.log_softmax(linearout_2, dim=1)
    
    def lstm_embedding(self, lstm, word_embedding ,hidden_init):
        lstm_out,(lstm_h, lstm_c) = lstm(word_embedding, None)
        if self.bidirectional:
            seq_embedding = torch.cat((lstm_h[0], lstm_h[1]), dim=1)
        else:
            seq_embedding = lstm_h.squeeze(0)
        return seq_embedding

    def init_hidden(self, batch_size, device) :
        layer_num = 2 if self.bidirectional else 1
        if device == -1:
            return (Variable(torch.randn(layer_num, batch_size, self.hidden_dim//layer_num)),Variable(torch.randn(layer_num, batch_size, self.hidden_dim//layer_num)))  
        else:
            return (Variable(torch.randn(layer_num, batch_size, self.hidden_dim//layer_num).cuda()),Variable(torch.randn(layer_num, batch_size, self.hidden_dim//layer_num).cuda()))  


print('Initialing model..')
MODEL = LSTM_angel(len(TEXT.vocab), embedding_dim, hidden_dim, batch_size, word_matrix, bidirectional)
if device == 0:
    MODEL.cuda()
    
# print(MODEL.state_dict())

# sys.exit()
best_state = None
max_metric = 0

# Train
if not test_mode:
    loss_func = nn.NLLLoss()
    parameters = list(filter(lambda p: p.requires_grad, MODEL.parameters()))
    optimizer = optim.Adam(parameters, lr=1e-3)
    print('Start training..')

    train_iter.create_batches()
    batch_num = len(list(train_iter.batches))

    batch_start = time.time()
    for i in range(epochs) :
        train_iter.init_epoch()
        batch_count = 0
        for text1, text2, label in train_dl:
            MODEL.train()
            hidden_init = MODEL.init_hidden(label.size()[0], device)
            y_pred = MODEL(text1, text2, hidden_init)
            loss = loss_func(y_pred, label)
            MODEL.zero_grad()
            loss.backward()
            optimizer.step()
            batch_count += 1
            if batch_count % print_every == 0:
                loss, (acc, Precision, Recall, F1) = predict_on(MODEL, valid_dl, loss_func, device)
                batch_end = time.time()
                if acc > max_metric:
                    best_state = MODEL.state_dict()
                    max_metric = acc
                    print("Saving model..")
                    torch.save(best_state, '../model_save/LSTM_angel.pth')           
                print('Finish {}/{} batch, {}/{} epoch. Time consuming {}s. acc is {}, Loss is {}'.format(batch_count, batch_num, i+1, epochs, round(batch_end - batch_start, 2), acc, float(loss)))



loss, (acc, Precision, Recall, F1) = predict_on(MODEL, test_dl, nn.NLLLoss(), device, '../model_save/LSTM_angel.pth')

print("=================")
print("Evaluation results on test dataset:")
print("Loss: {}.".format(float(loss)))
print("Accuracy: {}.".format(acc))
print("Precision: {}.".format(Precision))
print("Recall: {}.".format(Recall))
print("F1: {}.".format(F1))
print("=================")                                    