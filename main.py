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
parser.add_argument("-m", help="specify which embedding model should be used", type=str, default='LearnPyTorch', dest='model_type', choices=['Google', 'GoogleRed', 'LearnGensim', 'LearnPyTorch'])
parser.add_argument("-p", help="use powerful cuda nvidia gpu", dest='use_gpu', action='store_true')
args = parser.parse_args()

network_type = args.network_type
dataset_name = args.dataset_name
model_type = args.model_type
use_cuda = args.use_gpu



################################################################################
### HYPERPARAMETERS
################################################################################

sprint.p('Initializing Hyperparameters', 1)

k = 2 # 3, 5, 7
word_embedding_size = 300
word_embedding_window = 5
convolutional_filters = 2000
batch_size = 1 ## at most 8 on a GPU with 3GB
negative_answer_count_training = 20
learning_rate = 0.05
loss_margin = 0.009
training_epochs = 30

device = torch.device('cuda') if use_cuda and torch.cuda.is_available() else torch.device('cpu')

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
    we_model = custom.word_embedding_model(loader.get_documents(), word_embedding_size, word_embedding_window, 8)
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

sprint.p('Creating batch manager on dataset %s' % dataset_name, 1)
sprint.p('Train', 2)
training_dataset = datasets.DatasetManager(training_set, vocabulary, device, batch_size=batch_size, hard_negative_training=True, max_negative_answer_count=negative_answer_count_training)
sprint.p('Valid', 2)
validation_dataset = datasets.DatasetManager(validation_set, vocabulary, device, batch_size=batch_size)
sprint.p('Test', 2)
test_dataset = datasets.DatasetManager(test_set, vocabulary, device, batch_size=batch_size)
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
net = networks.ClassicQANetwork((training_dataset.max_question_len, training_dataset.max_answer_len),len(vocabulary), word_embedding_size, word_embedding_model=we_model, convolutional_filters=convolutional_filters, context_len=k).to(device)
#net = networks.ClassicQANetwork(len(vocabulary), word_embedding_size, word_embedding_model=we_model, convolutional_filters=convolutional_filters, context_len=3).to(device)
sprint.p("NN Instantiated", 2)



################################################################################
### SETUP FOR TRAINING
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


### trains on a batch and returns the sum of the losses
def train_batch(batch):
    sizes, questions, answers, targets = batch
    assert len(questions) == len(answers) == len(targets)
    outputs = net(questions, answers)

    loss = []
    base_index = 0
    for size in sizes:
        loss.append(criterion(outputs[base_index: base_index+size], targets[base_index: base_index+size]))
        base_index += size

    return sum(loss)

### test the actual network on an entire dataset and returns MAP and MRR
def test_on_dataset(ds):
    ds.reset()

    res_outputs = []
    res_targets = []

    while True:
        batch = ds.next()
        if batch is None:
            break

        sizes, questions, answers, targets = batch
        outputs = net(questions, answers)

        base_index = 0
        for size in sizes:
            res_outputs.append(outputs[base_index: base_index+size])
            res_targets.append(targets[base_index: base_index+size])
            base_index += size

    return (metrics.MRR(res_outputs, res_targets), metrics.MAP(res_outputs, res_targets))


################################################################################
### TRAINING
################################################################################

## Starting training time
starting_time = time()

for epoch in range(training_epochs):

    adjust_learning_rate(epoch+1)
    training_dataset.reset()

    while True:
        optimizer.zero_grad()   # zero the gradient buffers

        batch = training_dataset.next()
        if batch is None:    ## round on the training set complete, go to next epoch
            break

        loss = train_batch(batch)
        loss.backward(retain_graph=True)
        optimizer.step()    # Does the update
        sprint.p("Batch trained, AVG loss: %2.8f" % (loss.item()/batch_size), 3)

    sprint.p('Epoch %d done, MRR: %.2f, MAP: %.2f' % (epoch+1, *test_on_dataset(validation_dataset)), 2)
sprint.p('Training done, it took %.2f seconds' % (time()-starting_time), 2)



################################################################################
### TESTING THE NETWORK
################################################################################

sprint.p("Testing NN", 1)
starting_time = time()
sprint.p('Starting', 2)

sprint.p('MRR: %2.2f, MAP: %2.2f' % test_on_dataset(test_dataset), 2)

sprint.p('Testing done, it took %.2f seconds' % (time()-starting_time), 2)
