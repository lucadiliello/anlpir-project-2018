#!/usr/bin/env python3

## IMPORT DATA
import sys
import gensim
import torch
import torch.nn as nn
from torch import optim
from itertools import chain
from data_loader import loader
from termcolor import cprint

def sprint(stringa, indent):
    color = {
        1: 'green',
        2: 'yellow',
        3: 'blue',
        4: 'red',
    }
    stringa = '@@@ ' + '- ' * indent + stringa

    cprint(stringa, color[indent])

################################################################################
### LOAD DATASET
################################################################################

sprint('Loading datasets', 1)
train, valid, test = loader.Loader('WikiQA').load()


################################################################################
### REMOVE USELESS ENTRIES
################################################################################

def has_each_label(ds):
    res = []
    for entry in ds:

        has_one = False
        has_zero = False
        for x in entry['candidates']:
            if x['label'] == 1:
                has_one = True
            else:
                has_zero = True
        if has_one and has_zero:
            res.append(entry)
    return res

sprint('Removing useless entries', 1)

## REMOVING QUESTIONS WITH ONLY POSIVITE OR ONLY NEGATIVE ANSWERS
sprint("Working with: %d training elements, %d validation elements and %d test elements" % (len(train),len(valid),len(test)), 2)

train = has_each_label(train)
sprint('Train done', 2)
valid = has_each_label(valid)
sprint('Validation done', 2)
test = has_each_label(test)
sprint('Test done', 2)

sprint("After label filtering: %d training elements, %d validation elements and %d test elements" % (len(train),len(valid),len(test)), 2)

################################################################################
### CLEAN SENTENCES
################################################################################

sprint('Cleaning datasets', 1)

def clean_dataset(ds):
    res = []
    for entry in ds:
        tmp = {}
        tmp['question'] = gensim.utils.simple_preprocess(entry['question'])
        tmp['candidates'] = []
        for cand in entry['candidates']:
            tmp['candidates'].append({'sentence': gensim.utils.simple_preprocess(cand['sentence']), 'label': cand['label']})

        res.append(tmp)
    return res

train = clean_dataset(train)
sprint('Train done', 2)
valid = clean_dataset(valid)
sprint('Validation done', 2)
test = clean_dataset(test)
sprint('Test done', 2)


################################################################################
### VOCABULARY CREATION
################################################################################

sprint('Vocabulary creation', 1)

def get_sentences(dataset):
    for row in dataset:
        yield row['question']
        for answer in row['candidates']:
            yield answer['sentence']

documents = list(chain(get_sentences(train), get_sentences(valid), get_sentences(test)))

def create_vocabulary(sentences):
    vocab = dict()
    vocab['<PAD>'] = 0
    index = 1
    for sent in sentences:
        for token in sent:
            if token not in vocab:
                vocab[token] = index
                index += 1
    return vocab

vocab = create_vocabulary(documents)


################################################################################
### MAPPING WORD TO IDX
################################################################################

sprint('Mapping words to vocabulary indexes', 1)

def word2idx(sequence):
    return [vocab[word] for word in sequence]

def get_dataset_embedding(ds):
    res = []
    for entry in ds:
        tmp = {}
        tmp['question'] = word2idx(entry['question'])
        tmp['candidates'] = []
        for cand in entry['candidates']:
            tmp['candidates'].append({'sentence': word2idx(cand['sentence']), 'label': cand['label']})

        res.append(tmp)
    return res

train = get_dataset_embedding(train)
sprint('Train done', 2)
valid = get_dataset_embedding(valid)
sprint('Validation done', 2)
test = get_dataset_embedding(test)
sprint('Test done', 2)


################################################################################
### HYPERPARAMETERS
################################################################################

sprint('Initializing Hyperparameters', 1)
k = 3 # 3, 5, 7
word_embedding_size = 300
convolutional_filters = 400
batch_size = 20
initial_learning_rate = 1.1
loss_margin = 0.5
training_epochs = 20
n_threads = 8
word_embedding_window = 3

def get_device():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            if torch.cuda.get_device_capability(i) >= (3,5):
                return torch.device('cuda:%d' % i)
    return torch.device('cpu:0')

device = get_device()

sprint('Will train on %s' % (torch.cuda.get_device_name(device.index) if device.type.startswith('cuda') else device.type), 2)


################################################################################
### WORD EMBEDDING word2vec CREATION
################################################################################

sprint('Training the word2vec encoder', 1)
## sudo pip3 install cython - to speed up embedding
## # TODO: save training model to disk to speed up starting
model = gensim.models.Word2Vec(documents, size=word_embedding_size, window=word_embedding_window, min_count=0, workers=n_threads)
model.train(documents, total_examples=len(documents), epochs=100)

sprint('Trained', 2)
word2vec_embedding_matrix = torch.FloatTensor(model.wv.syn0).to(device)
sprint('Done', 2)


################################################################################
### Z VECTOR CREATION & PADDING
################################################################################

sprint('Getting max Q/A length', 1)

def get_M_L(ds):
    M = 0
    L = 0
    for entry in ds:
        M = max(M, len(entry['question']))
        for cand in entry['candidates']:
            L = max(L, len(cand['sentence']))
    return (M, L)

tup = (get_M_L(train), get_M_L(valid), get_M_L(test))
M_len, L_len = (max([x[0] for x in tup]), max([x[1] for x in tup]))

sprint("Max question length: %d, max answer length: %d" % (M_len, L_len), 2)


################################################################################
### DATASETS PADDING
################################################################################

sprint('Padding datasets to %d for questions and %d for answers' % (M_len, L_len), 1)

def pad_dataset(ds):
    res = []
    for entry in ds:
        tmp = {}
        tmp['question'] = entry['question'] + ([0] * (M_len - len(entry['question'])) if M_len - len(entry['question']) > 0 else [])
        tmp['candidates'] = []
        for cand in entry['candidates']:
            tmp['candidates'].append({'sentence': cand['sentence'] + ([0] * (L_len - len(cand['sentence'])) if L_len - len(cand['sentence']) > 0 else []), 'label': cand['label']})
        res.append(tmp)
    return res


train = pad_dataset(train)
sprint('Train done', 2)
valid = pad_dataset(valid)
sprint('Validation done', 2)
test = pad_dataset(test)
sprint('Test done', 2)


################################################################################
### BATCH GENERATORS CREATION
################################################################################

sprint("Batch generators creation",1)

from support_data_structures import datasets

train_ds = datasets.BatchCreator(train, batch_size, device)
sprint("Done train",2)
valid_ds = datasets.BatchCreator(valid, 50, device)
sprint("Done validation",2)
test_ds = datasets.BatchCreator(test, batch_size, device)
sprint("Done test",2)


################################################################################
### NEURAL NETWORK
################################################################################

sprint("Neural network creation",1)

from support_data_structures import networks

net = networks.AttentivePoolingNetwork((M_len, L_len), len(vocab), word_embedding_size, word2vec_embedding_matrix, device, type_of_nn=sys.argv[1], convolutional_filters=convolutional_filters, context_len=k).to(device)
#print(net)
sprint("NN Instantiated", 2)

################################################################################
### TRAINING NETWORK
################################################################################

sprint("Training NN",1)

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=initial_learning_rate)
net.train()

#print([x.size() for x in list(net.parameters())])

from time import time
starting_time = time()
sprint('Batch size: %d' % batch_size, 2)

sprint("Starting",2)
for epoch in range(training_epochs):
    questions, inputs_pos, inputs_neg = train_ds.next()

    optimizer.zero_grad()   # zero the gradient buffers

    outputs_pos = net(questions, inputs_pos)
    outputs_neg = net(questions, inputs_neg)

    sum = loss_margin + outputs_neg - outputs_pos
    ## to do the max between the results and 0 we place a tensor filled with zero side by side with the results tensor
    res = torch.stack([sum, torch.zeros(batch_size, requires_grad=False).to(device)]).transpose(0,1).view(1,batch_size,2)

    ## maxpooling on each tuple (res[i], 0)
    maxpoller = torch.nn.MaxPool1d(2)

    loss = (maxpoller(res)**2).mean()

    #trials
    #loss = ((outputs_pos - 1.)**2 + (outputs_neg)**2).sum()
    #loss = sum((outputs_pos - 1)**2 + (outputs_neg)**2)

    sprint("Epoch %d, loss: %2.3f" % (epoch+1, loss.item()), 3)
    loss.backward(retain_graph=True)
    optimizer.step()    # Does the update

    ## Validation
    thres = 0
    quest, valid_pos, valid_neg = valid_ds.next()

    valid_outputs_pos = net(quest, valid_pos) > thres
    valid_outputs_neg = net(quest, valid_neg) > thres
    #print(valid_outputs_pos)
    #print(valid_outputs_neg)
    true_pos = torch.nonzero(valid_outputs_pos).size(0)
    false_neg = valid_outputs_pos.size(0) - torch.nonzero(valid_outputs_pos).size(0)
    true_neg = torch.nonzero(valid_outputs_neg).size(0)
    false_pos = valid_outputs_pos.size(0) - torch.nonzero(valid_outputs_neg).size(0)
    #print(true_pos, true_neg, false_pos, false_neg)

    sprint('Accuracy: %2.2f - Precision: %2.2f - Recall: %2.2f' % ((true_pos+true_neg) / (true_pos+true_neg+false_neg+false_pos), true_pos/ (true_pos+false_pos), true_pos/ (true_pos+false_neg) ), 4)

sprint('Training took %.2f seconds' % (time()-starting_time), 2)

################################################################################
### TESTING THE NETWORK
################################################################################

sprint("Testing NN", 1)

starting_time = time()
accu_array = []
recall_array = []
prec_array = []

round = 1
sprint('Starting', 2)
while True:
    input = test_ds.ordered_next()
    if input is None:
        break
    questions, inputs_pos, inputs_neg = input

    valid_outputs_pos = net(quest, valid_pos) > thres
    valid_outputs_neg = net(quest, valid_neg) > thres

    true_pos = torch.nonzero(valid_outputs_pos).size(0)
    false_neg = valid_outputs_pos.size(0) - torch.nonzero(valid_outputs_pos).size(0)
    true_neg = torch.nonzero(valid_outputs_neg).size(0)
    false_pos = valid_outputs_pos.size(0) - torch.nonzero(valid_outputs_neg).size(0)

    accu_array.append((true_pos+true_neg) / (true_pos+true_neg+false_neg+false_pos))
    recall_array.append(true_pos/ (true_pos+false_neg))
    prec_array.append(true_pos/ (true_pos+false_pos))

    sprint('Round %d done! (%d elements tested)' % (round, len(questions)), 3)
    round += 1


import numpy
sprint('Accuracy: %2.2f - Precision: %2.2f - Recall: %2.2f' % (numpy.mean(accu_array), numpy.mean(prec_array), numpy.mean(recall_array)), 2)

sprint('Testing took %.2f seconds' % (time()-starting_time), 2)
