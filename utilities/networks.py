import math
import torch
import torch.nn as nn
from torch.nn import Module, Parameter, init
from utilities import sprint as sp

sprint = sp.SPrint()


class WordEmbeddingModule(Module):

    def __init__(self, vocab_size, embedding_size, word_embedding_model=None):
        super(WordEmbeddingModule, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        if word_embedding_model:
            weights = torch.tensor(word_embedding_model.wv.syn0)
            weights = torch.cat((torch.zeros(1,self.embedding_size), weights), dim=0)
            self.embedding_layer = nn.Embedding(self.vocab_size + 1, self.embedding_size, padding_idx=0).from_pretrained(weights, freeze=True)
        else:
            self.embedding_layer = nn.Embedding(self.vocab_size + 1, self.embedding_size, padding_idx=0)

    def forward(self, sentences):
        return self.embedding_layer(sentences)



class CNN(Module):

    def __init__(self, embedding_size, convolutional_filters, context_len):
        super(CNN, self).__init__()

        self.embedding_size = embedding_size
        self.convolutional_filters = convolutional_filters
        self.context_len = context_len

        self.conv = nn.Conv1d(self.embedding_size, self.convolutional_filters, self.context_len, padding=1)

    def forward(self, sentence):
        ## bs * M/L * d

        sentence = sentence.transpose(1, 2)
        ## bs * d * M/L

        sentence = self.conv(sentence)
        ## bs * c * M/L

        return sentence



class biLSTM(Module):

    def __init__(self, embedding_size, hidden_dim, bidirectional=True):
        super(biLSTM, self).__init__()

        self.hidden_dim = int(hidden_dim/2)
        self.embedding_size = embedding_size

        self.lstm = nn.LSTM(self.embedding_size, self.hidden_dim, bias=True, bidirectional=True, batch_first=True)

    def init_hidden(self, bs):
        # print('size of hidden_dim', self.hidden_dim)
        '''
        return (torch.zeros(1 * 2, bs, self.hidden_dim),
                torch.zeros(1 * 2, bs, self.hidden_dim))
        '''
        return (torch.zeros(1 * 2, bs, self.hidden_dim).cuda(),
                torch.zeros(1 * 2, bs, self.hidden_dim).cuda())

    def forward(self, x):
        ## bs * M/L * d -> (batch, seq_len, input_size)
        #sprint.p('heilaaaa', 2)
        bs = x.size()[0]
        self.hidden = self.init_hidden(bs)

        x, self.hidden = self.lstm(x, self.hidden)

        ## bs * M * c
        x = x.transpose(1,2)
        #print('must be bs*c*M',x.size())
        ## bs * c * M
        return x



class Bilinear2D(nn.Module):

    """ Apply Bilinear transformation with 2D input matrices: A * U * B"""

    def __init__(self, in1_features, in2_features):
        super(Bilinear2D, self).__init__()

        self.in1_features = in1_features    # size (*, n, m)
        self.in2_features = in2_features    # size (*, c, d)
        ## self.weight matrix will have size (m * c)
        self.weight = Parameter(torch.Tensor(self.in1_features, self.in2_features), requires_grad=True)
        torch.nn.init.normal_(self.weight, mean=0, std=1)

    def forward(self, input1, input2):
        if input1.dim() == 2 and input2.dim() == 2:
            # fused op is marginally faster
            return input1.mm(self.weight).mm(input2)
        print(self.weight.sum())
        return input1.matmul(self.weight).matmul(input2)

    def extra_repr(self):
        return 'in1_features={}, in2_features={}'.format(self.in1_features, self.in2_features)



class AttentivePoolingNetwork(Module):

    def __init__(self, vocab_size, embedding_size , word_embedding_model=None, type_of_nn='CNN', convolutional_filters=400, context_len=3):
        super(AttentivePoolingNetwork, self).__init__()

        self.convolutional_filters = convolutional_filters
        self.context_len = context_len
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.batch_size = 2

        self.embedding_layer = WordEmbeddingModule(vocab_size, embedding_size, word_embedding_model)

        ## CNN or biLSTM
        if type_of_nn == 'CNN':
            self.cnn_or_bilstm = CNN(self.embedding_size, self.convolutional_filters, self.context_len)
        elif type_of_nn == 'biLSTM':
            self.cnn_or_bilstm = biLSTM(self.embedding_size, self.convolutional_filters)
        else:
            raise ValueError('Mode must be CNN or biLSTM')

        #self.bilinear = Bilinear2D(self.convolutional_filters, self.convolutional_filters)
        self.weight = Parameter(torch.Tensor(self.convolutional_filters, self.convolutional_filters), requires_grad=True)
        torch.nn.init.normal_(self.weight, mean=0, std=1)



    def forward(self, questions, answers):
        ## questions: bs * M
        ## answers: bs * L

        questions = self.embedding_layer(questions)
        ## bs * M * d

        answers = self.embedding_layer(answers)
        ## bs * L * d

        Q = self.cnn_or_bilstm(questions)
        ## bs * c * M
        #print('q size bs*c*M', Q.size())
        A = self.cnn_or_bilstm(answers)
        ## bs * c * L
        #print('a size bs*c*L', A.size())

        G = (Q.transpose(1,2).matmul(self.weight).matmul(A)).tanh()
        #G = self.bilinear(Q.transpose(1,2), A).tanh()
        ## bs * M * L
        #print('G size bs * M * L', G.size())
        roQ = torch.max(G, dim=2)[0].softmax(dim=1)
        ## bs * M
        roA = torch.max(G, dim=1)[0].softmax(dim=1)
        ## bs * L

        rQ = Q.matmul(roQ.unsqueeze(2)).squeeze()
        ## bs * c
        rA = A.matmul(roA.unsqueeze(2)).squeeze()
        ## bs * c

        return torch.nn.functional.cosine_similarity(rQ, rA, dim=1, eps=1e-08)
        ## bs



class ClassicQANetwork(nn.Module):
    #def __init__(self, max_len, vocab_size, embedding_size, word_embedding_model=None, network_type='CNN', convolutional_filters=400, context_len=3):
    def __init__(self, vocab_size, embedding_size, word_embedding_model=None, network_type='CNN', convolutional_filters=400, context_len=3):
        super(ClassicQANetwork, self).__init__()

        self.convolutional_filters = convolutional_filters
        self.context_len = context_len
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.network_type = network_type
        #self.M, self.L = max_len

        self.embedding_layer = WordEmbeddingModule(vocab_size, embedding_size, word_embedding_model)

        #if network_type == 'CNN':
        self.cnn = CNN(self.embedding_size, self.convolutional_filters, self.context_len)
        #else:
        #    self.bilstm_q = biLSTM(self.M, self.embedding_size,  self.convolutional_filters)
        #    self.bilstm_a = biLSTM(self.L, self.embedding_size,  self.convolutional_filters)

    def forward(self, questions, answers):
        ## questions: bs * M
        ## answers: bs * L

        questions = self.embedding_layer(questions)
        ## bs * M * d
        answers = self.embedding_layer(answers)
        ## bs * L * d

        Q = self.cnn(questions)
        ## bs * c * M

        A = self.cnn(answers)
        ## bs * c * L

        rQ = torch.max(Q, dim=2)[0].tanh()
        ## bs * c
        rA = torch.max(A, dim=2)[0].tanh()
        ## bs * c

        return torch.nn.functional.cosine_similarity(rQ, rA, dim=1, eps=1e-08)
        ## bs
