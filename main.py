#!/usr/bin/env python3

### IMPORT DATA
import gensim
import torch
import torch.nn as nn
from torch import optim
from time import time
import argparse
from utilities import networks, metrics, sprint, datasets
import numpy


################################################################################
### PARSE COMMAND LINE ARGUMENTS
################################################################################

parser = argparse.ArgumentParser(description='Create an AP network for Question Answering')

parser.add_argument("-n", help="type of the network, either CNN or biLSTM", type=str, default='CNN', dest='network_type', choices=['CNN','biLSTM'])
parser.add_argument("-d", help="dataset to use, either TrecQA or WikiQA", type=str, default='TrecQA', dest='dataset_name', choices=['TrecQA','WikiQA'])
parser.add_argument("-r", help="Use reduced GoogleNews WE model", action="store_true", dest="user_red_model")
args = parser.parse_args()
network_type = args.network_type
dataset_name = args.dataset_name
user_red_model = args.user_red_model

################################################################################
### LOADING WORD EMBEDDINGS GOOGLE-NEWS MODEL
################################################################################

sprint.p('Loading the Google News Word Embedding model', 1)

starting_time = time()
if user_red_model:
    we_model = gensim.models.KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300-SLIM.bin', binary=True)
else:
    we_model = gensim.models.KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
sprint.p('Loading took %d seconds' % (time()-starting_time), 2)

sprint.p('Done', 2)


################################################################################
### HYPERPARAMETERS
################################################################################

sprint.p('Initializing Hyperparameters', 1)
k = 3 # 3, 5, 7
word_embedding_size = we_model.syn0.shape[1]
convolutional_filters = 400
batch_size = 20
learning_rate = 1.1
loss_margin = 0.5
training_epochs = 25
test_rounds = 40

def get_device():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            if torch.cuda.get_device_capability(i) >= (3,5):
                return torch.device('cuda:%d' % i)
    return torch.device('cpu:0')

device = get_device()

sprint.p('Will train on %s' % (torch.cuda.get_device_name(device.index) if device.type.startswith('cuda') else device.type), 2)


################################################################################
### LOADING DATASET
################################################################################

sprint.p('Loading datasets', 1)
dataset = datasets.DatasetManager(dataset_name, batch_size, device, we_model)


################################################################################
### STATISTICS ON THE DATASET
################################################################################

sprint.p("Statistics on the 3 datasets", 1)

results = dataset.get_statistics()

sprint.p("Average number of positive answers", 2)
sprint.p("TRAIN: %2.2f - VALIDATION: %2.2f - TEST %2.2f" % (results['train']['average_number_pos_answers'], results['valid']['average_number_pos_answers'], results['test']['average_number_pos_answers']), 3)
sprint.p("Average number of negative answers", 2)
sprint.p("TRAIN: %2.2f - VALIDATION: %2.2f - TEST %2.2f" % (results['train']['average_number_neg_answers'], results['valid']['average_number_neg_answers'], results['test']['average_number_neg_answers']), 3)
sprint.p("Average questions length", 2)
sprint.p("TRAIN: %2.2f - VALIDATION: %2.2f - TEST %2.2f" % (results['train']['average_question_len'], results['valid']['average_question_len'], results['test']['average_question_len']), 3)
sprint.p("Average answers length", 2)
sprint.p("TRAIN: %2.2f - VALIDATION: %2.2f - TEST %2.2f" % (results['train']['average_answer_len'], results['valid']['average_answer_len'], results['test']['average_answer_len']), 3)


################################################################################
### NEURAL NETWORK
################################################################################

sprint.p("Neural network creation",1)
net = networks.AttentivePoolingNetwork((dataset.max_question_len, dataset.max_answer_len), word_embedding_size, device, type_of_nn=network_type, convolutional_filters=convolutional_filters, context_len=k).to(device)
#print(net)
sprint.p("NN Instantiated", 2)

################################################################################
### TRAINING NETWORK
################################################################################

sprint.p("Training NN",1)

optimizer = optim.SGD(net.parameters(), lr=learning_rate)
net.train()
criterion = nn.MSELoss()

#print([x for x in net.parameters()])

def adjust_learning_rate(epo):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate / epo
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(questions, answers):
    optimizer.zero_grad()   # zero the gradient buffers
    output = net(questions, answers)

    loss = torch.max(torch.tensor(0.).to(device), (loss_margin - output[0] + output[1:].max()) )

    loss.backward()
    optimizer.step()    # Does the update

    return loss.item()

def test(questions, answers):
    return net(questions, answers)


starting_time = time()
sprint.p('Batch size: %d' % batch_size, 2)
sprint.p("Starting",2)
dataset.train_mode()

for epoch in range(training_epochs):

    # adjust learning rate
    adjust_learning_rate(epoch+1)

    sprint.p("Epoch %d, loss: %2.3f" % (epoch+1, train(*dataset.next())), 3)
    #sprint("Epoch %d, loss: %2.3f" % (epoch+1, train(*train_ds.test_batch(balanced=True, size=20))), 3)
    # validation
    #sprint('Accuracy: %2.2f - Precision: %2.2f - Recall: %2.2f' % test(*valid_ds.next()), 4)

sprint.p('Training took %.2f seconds' % (time()-starting_time), 2)


################################################################################
### TESTING THE NETWORK
################################################################################

sprint.p("Testing NN", 1)

starting_time = time()
results = []
round = 1

sprint.p('Starting', 2)
dataset.test_mode()

while round < test_rounds:
    results.append(test(*dataset.next()))

    sprint.p('Round %d' % round, 3)
    round += 1

sprint.p('MRR: %2.2f, MAP: %2.2f' % (metrics.MRR(results), metrics.MAP(results)), 2)
sprint.p('Testing took %.2f seconds' % (time()-starting_time), 2)
