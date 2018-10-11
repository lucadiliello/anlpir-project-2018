#!/usr/bin/env python3

### IMPORT DATA
import gensim
import torch
import torch.nn as nn
from torch import optim
import argparse
from utilities import networks, metrics, sprint, datasets, loader, custom, losses
import numpy



################################################################################
### PARSE COMMAND LINE ARGUMENTS
################################################################################

parser = argparse.ArgumentParser(description='Create an AP network for Question Answering')

parser.add_argument("-d", help="dataset to use, either TrecQA or WikiQA", type=str, default='TrecQA', dest='dataset_name', choices=['TrecQA','WikiQA'])
args = parser.parse_args()
dataset_name = args.dataset_name



################################################################################
### HYPERPARAMETERS
################################################################################

sprint.p('Initializing Hyperparameters', 1)

k = 2 # 3, 5, 7
word_embedding_size = 300
word_embedding_window = 5
convolutional_filters = 4000
batch_size = 20
learning_rate = 0.05
loss_margin = 0.009
training_epochs = 1000
test_rounds = 100



################################################################################
### LOADING DATASET
################################################################################

sprint.p('Loading datasets', 1)
datasets_tupla = loader.Loader(dataset_name).load()
sprint.p('Datasets loaded', 2)



################################################################################
### DOCUMENTS FOR VOCABULARY CREATION
################################################################################

def get_sentences(ds):
    for row in ds:
        yield gensim.utils.simple_preprocess(row['question'])
        for answer in row['candidates']:
            yield gensim.utils.simple_preprocess(answer['sentence'])

documents = []
for ds in datasets_tupla:
    documents += list(get_sentences(ds))



################################################################################
### LOADING WORD EMBEDDINGS MODEL
################################################################################

sprint.p('Loading the Word Embedding model', 1)

we_model = gensim.models.KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
#we_model = gensim.models.KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300-SLIM.bin', binary=True)
word_embedding_size = we_model.wv.syn0.shape[1]

sprint.p('Done', 2)



################################################################################
### CREATING/EXTRACTING VOCABULARY
################################################################################

def create_vocabulary(docs):
    vocab = dict()
    index = 1
    for sent in docs:
        for token in sent:
            if token not in vocab:
                vocab[token] = index
                index += 1
    return vocab

vocabulary = {key: (value.index + 1) for (key, value) in we_model.wv.vocab.items()} if we_model else create_vocabulary(documents)



################################################################################
### LOADING DATASET
################################################################################

sprint.p('Creating batch manager', 1)
dataset = datasets.DatasetManager(datasets_tupla, batch_size, vocabulary)
sprint.p('Done', 2)


################################################################################
### NEURAL NETWORK
################################################################################

sprint.p("Neural network creation",1)
net = networks.AttentivePoolingNetwork((len(vocabulary), word_embedding_size), word_embedding_model=we_model, convolutional_filters=convolutional_filters, context_len=k)
#print(net)
sprint.p("NN Instantiated", 2)

def weights_initializer(m):
    if m.data:
        torch.nn.init.normal_(m.data, mean=0, std=0.1)
    elif:
        if m.weight:
            torch.nn.init.normal_(m.weight, mean=0, std=0.1)
        if m.bias:
            torch.nn.init.uniform_(m.bias, a=0, b=0.1)

net.apply(weights_initializer)

################################################################################
### TRAINING NETWORK
################################################################################

sprint.p("Training NN",1)

net.train()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)


sprint.p('Batch size: %d' % batch_size, 2)
sprint.p("Starting",2)

criterion = losses.ObjectiveHingeLoss(loss_margin)

for epoch in range(training_epochs):
    optimizer.zero_grad()   # zero the gradient buffers

    loss = torch.zeros(1)
    for _ in range(batch_size):
        questions, answers, targets = dataset.next_train()
        outputs = net(questions, answers)
        loss += criterion(outputs, targets)

    loss.backward()
    optimizer.step()    # Does the update

    #print([x.sum() if x.grad is not None else 'nograd' for x in net.parameters()])
    #print([x.grad.sum() if x.grad is not None else 'nograd' for x in net.parameters()])

    sprint.p("Epoch %d, AVG loss: %2.3f" % (epoch+1, loss.item()/batch_size), 3)

sprint.p('Training done', 2)


################################################################################
### TESTING THE NETWORK
################################################################################

sprint.p("Testing NN", 1)

res_outputs = []
res_targets = []

sprint.p('Starting', 2)

for round in range(test_rounds):
    questions, answers, targets = dataset.next_test()
    outputs = net(questions, answers)

    res_outputs.append(outputs)
    res_targets.append(targets)

    sprint.p('Round %d' % (round+1), 3)

sprint.p('MRR: %2.2f, MAP: %2.2f' % (metrics.MRR(res_outputs, res_targets), metrics.MAP(res_outputs, res_targets)), 2)
#metrics.MAP(results)
