#!/usr/bin/env python3

### IMPORT DATA
import gensim
import torch
import torch.nn as nn
from torch import optim
from time import time
import argparse
from utilities import networks, metrics, sprint, datasets, loader, custom, losses
import numpy
from prettytable import PrettyTable



################################################################################
### PARSE COMMAND LINE ARGUMENTS
################################################################################

parser = argparse.ArgumentParser(description='Create an AP network for Question Answering')

parser.add_argument("-n", help="type of the network, either CNN or biLSTM", type=str, default='CNN', dest='network_type', choices=['CNN','biLSTM'])
parser.add_argument("-d", help="dataset to use, either TrecQA or WikiQA", type=str, default='TrecQA', dest='dataset_name', choices=['TrecQA','WikiQA'])
parser.add_argument("-m", help="specify which embedding model should be used", type=str, default='Google', dest='model_type', choices=['Google', 'GoogleRed', 'LearnGensim', 'LearnPyTorch'])
args = parser.parse_args()
network_type = args.network_type
dataset_name = args.dataset_name
model_type = args.model_type



################################################################################
### HYPERPARAMETERS
################################################################################

sprint.p('Initializing Hyperparameters', 1)

k = 3 # 3, 5, 7
word_embedding_size = 300
word_embedding_window = 5
convolutional_filters = 400
batch_size = 20
negative_answer_count_training = 20
learning_rate = 0.01
loss_margin = 0.5
training_epochs = 400
test_rounds = 300

device = torch.device('cpu')
#device = torch.device('cuda') # Uncomment this to run on GPU

sprint.p('Will train on %s' % (torch.cuda.get_device_name(device) if device.type == 'cuda' else device.type), 2)



################################################################################
### LOADING DATASET
################################################################################

sprint.p('Loading datasets', 1)
loader = loader.Loader(dataset_name)
training_set, validation_set, test_set = loader.load()
sprint.p('Datasets loaded', 2)



################################################################################
### LOADING WORD EMBEDDINGS MODEL
################################################################################

sprint.p('Loading the Word Embedding model, you chose %s' % model_type, 1)

starting_time = time()
if model_type == 'Google':
    we_model = gensim.models.KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
    word_embedding_size = we_model.wv.syn0.shape[1]
elif model_type == 'GoogleRed':
    we_model = gensim.models.KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300-SLIM.bin', binary=True)
    word_embedding_size = we_model.wv.syn0.shape[1]
elif model_type == 'LearnGensim':
    we_model = custom.word_embedding_model(documents, word_embedding_size, word_embedding_window, n_threads)
elif model_type == 'LearnPyTorch':
    we_model = None
else:
    print('Wrong WE model argument')
    exit()

sprint.p('Loading took %d seconds' % (time()-starting_time), 2)
sprint.p('Done', 2)



################################################################################
### CREATING/EXTRACTING VOCABULARY
################################################################################

vocabulary = {key: (value.index + 1) for (key, value) in we_model.wv.vocab.items()} if we_model else loader.get_vocabulary()



################################################################################
### LOADING DATASET
################################################################################

sprint.p('Creating batch manager', 1)
sprint.p('Train', 2)
training_dataset = datasets.DatasetManager(training_set, vocabulary, device, hard_negative_training=True, negative_answer_count=negative_answer_count_training)
sprint.p('Valid', 2)
validation_dataset = datasets.DatasetManager(validation_set, vocabulary, device)
sprint.p('Test', 2)
test_dataset = datasets.DatasetManager(test_set, vocabulary, device)
sprint.p('Done', 2)



################################################################################
### STATISTICS ON THE DATASET
################################################################################

sprint.p("Statistics on the 3 datasets", 1)
t = PrettyTable(['Dataset', 'AVG # pos answers', 'AVG # neg answers', 'AVG question len', 'AVG answer len'])
t.add_row(['TRAIN'] + ['%2.2f' % x for x in training_dataset.get_statistics()])
t.add_row(['VALIDATION'] + ['%2.2f' % x for x in validation_dataset.get_statistics()])
t.add_row(['TEST'] + ['%2.2f' % x for x in test_dataset.get_statistics()])
print(t)



################################################################################
### NEURAL NETWORK
################################################################################

sprint.p("Neural network creation",1)
net = networks.AttentivePoolingNetwork(len(vocabulary), word_embedding_size, word_embedding_model=we_model, type_of_nn=network_type, convolutional_filters=convolutional_filters, context_len=k).to(device)
#net = networks.ClassicQANetwork(len(vocabulary), word_embedding_size, word_embedding_model=we_model, convolutional_filters=convolutional_filters, context_len=3).to(device)

sprint.p("NN Instantiated", 2)



################################################################################
### TRAINING NETWORK
################################################################################

sprint.p("Training NN",1)

net.train()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criterion = losses.ObjectiveHingeLoss(loss_margin)

def adjust_learning_rate(epo):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate / epo
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

sprint.p('Batch size: %d' % batch_size, 2)
sprint.p("Starting",2)

starting_time = time()

for epoch in range(training_epochs):
    optimizer.zero_grad()   # zero the gradient buffers

    loss = []
    for _ in range(batch_size):
        questions, answers, targets = training_dataset.next()
        outputs = net(questions, answers)
        loss.append(criterion(outputs, targets))

    loss = sum(loss)
    loss.backward()
    optimizer.step()    # Does the update

    #print([x.sum() if x.grad is not None else 'nograd' for x in net.parameters()])
    #print([x.grad.sum() if x.grad is not None else 'nograd' for x in net.parameters()])

    sprint.p("Epoch %d, AVG loss: %2.3f" % (epoch+1, loss.item()/batch_size), 3)

sprint.p('Training done, it took %.2f seconds' % (time()-starting_time), 2)



################################################################################
### TESTING THE NETWORK
################################################################################

sprint.p("Testing NN", 1)
starting_time = time()

res_outputs = []
res_targets = []

sprint.p('Starting', 2)

for round in range(test_rounds):
    questions, answers, targets = test_dataset.next()
    outputs = net(questions, answers)

    res_outputs.append(outputs)
    res_targets.append(targets)

    sprint.p('Round %d' % (round+1), 3)

sprint.p('MRR: %2.2f, MAP: %2.2f' % (metrics.MRR(res_outputs, res_targets), metrics.MAP(res_outputs, res_targets)), 2)

sprint.p('Testing done, it took %.2f seconds' % (time()-starting_time), 2)
