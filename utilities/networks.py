import math
import torch
import torch.nn as nn
from torch.nn import Module, Parameter, init

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
        ## sentences: bs * M/L

        sentences = self.embedding_layer(sentences)
        ## bs * M/L * d

        return sentences


class CNN(Module):

    def __init__(self, embedding_size, convolutional_filters, context_len):
        super(CNN, self).__init__()

        self.embedding_size = embedding_size
        self.convolutional_filters = convolutional_filters
        self.context_len = context_len

        self.conv = nn.Conv1d(self.embedding_size, self.convolutional_filters, self.context_len, padding=1)

    def forward(self, sentence):
        ## question: bs * M * d
        ## answer: bs * L * d

        sentence = sentence.transpose(1, 2)
        ## bs * d * M/L

        sentence = self.conv(sentence)
        ## bs * c * M/L

        return sentence


class biLSTM(Module):

    def __init__(self, max_len, embedding_size, hidden_dim, device):
        super(biLSTM, self).__init__()

        hidden_dim = int(hidden_dim/2)
        self.device = device
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.embedding_size = embedding_size

        self.lstm = nn.LSTM(embedding_size, hidden_dim, bidirectional=True)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(2, self.max_len, self.hidden_dim),
                torch.zeros(2, self.max_len, self.hidden_dim))

    def forward(self, x):
        out, self.hidden = self.lstm(x, self.hidden)
        ## bs*M*c
        out = out.transpose(1,2)
        ## bs*c*M
        return out


class AttentivePoolingNetwork(Module):

    def __init__(self, vocab_size, embedding_size , word_embedding_model=None, type_of_nn='CNN', convolutional_filters=400, context_len=3):
        super(AttentivePoolingNetwork, self).__init__()

        self.convolutional_filters = convolutional_filters
        self.context_len = context_len
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.embedding_layer = WordEmbeddingModule(vocab_size, embedding_size, word_embedding_model)

        ## CNN or biLSTM
        if type_of_nn == 'CNN':
            self.cnn_bilstm = CNN(self.embedding_size, self.convolutional_filters, self.context_len)
        elif type_of_nn == 'biLSTM':
            raise ValueError('Not implemented yet')
            #self.question_cnn_bilstm = biLSTM(self.M, self.embedding_size, self.convolutional_filters, self.device)
            #self.answer_cnn_bilstm = biLSTM(self.L, self.embedding_size, self.convolutional_filters, self.device)
        else:
            raise ValueError('Mode must be CNN or biLSTM')

        #self.U = torch.nn.Parameter(torch.Tensor(self.convolutional_filters, self.convolutional_filters))
        #torch.nn.init.normal_(self.U, mean=0, std=0.1)
        #self.U = nn.Linear(self.convolutional_filters, self.convolutional_filters, bias=False)
        self.bilinear = Bilinear2D(self.convolutional_filters, self.convolutional_filters)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, question, answer):
        ## question: bs * M
        ## answer: bs * L

        question = self.embedding_layer(question)
        ## bs * M * d
        answer = self.embedding_layer(answer)
        ## bs * M * d

        Q = self.cnn_bilstm(question)
        ## bs * c * M
        A = self.cnn_bilstm(answer)
        ## bs * c * L

        G = self.bilinear(Q.transpose(1,2), A).tanh()
        #G = torch.tanh(self.U(Q.transpose(1,2)).matmul(A))
        ## bs * M * L

        roQ = torch.max(G, dim=2)[0]
        ## bs * M
        roA = torch.max(G, dim=1)[0]
        ## bs * L

        roQ = self.softmax(roQ)
        ## bs * M
        roA = self.softmax(roA)
        ## bs * L

        rQ = Q.matmul(roQ.unsqueeze(2)).squeeze()
        ## bs * c
        rA = A.matmul(roA.unsqueeze(2)).squeeze()
        ## bs * c

        return torch.nn.functional.cosine_similarity(rQ, rA, dim=1, eps=1e-08)
        ## bs


class ClassicQANetwork(nn.Module):
    def __init__(self, vocab_size, embedding_size, word_embedding_model=None, convolutional_filters=400, context_len=3):
        super(ClassicQANetwork, self).__init__()

        self.convolutional_filters = convolutional_filters
        self.context_len = context_len
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.embedding_layer = WordEmbeddingModule(vocab_size, embedding_size, word_embedding_model)

        self.cnn = CNN(self.embedding_size, self.convolutional_filters, self.context_len)


    def forward(self, question, answer):
        ## question: bs * M
        ## answer: bs * L

        question = self.embedding_layer(question)
        ## bs * M * d
        answer = self.embedding_layer(answer)
        ## bs * M * d

        Q = self.cnn(question)
        ## bs * c * M
        A = self.cnn(answer)
        ## bs * c * L

        rQ = torch.max(Q, dim=2)[0]
        ## bs * c
        rA = torch.max(A, dim=2)[0]
        ## bs * c

        rQ = torch.tanh(rQ)
        ## bs * c
        rA = torch.tanh(rA)
        ## bs * c

        return torch.nn.functional.cosine_similarity(rQ, rA, dim=1, eps=1e-08)
        ## bs



class Bilinear2D(nn.Module):

    """ Apply Bilinear transformation with 2D input matrices: A * U * B + b"""

    def __init__(self, in1_features, in2_features):
        super(Bilinear2D, self).__init__()
        assert(in1_features == in2_features)

        self.in1_features = in1_features    # expected (*,b)
        self.in2_features = in2_features    # expected (b,*)
        ## out will be a tensor of size (N,a,c)

        self.weight = Parameter(torch.Tensor(self.in1_features, self.in2_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input1, input2):

        if input1.dim() == 2 and input2.dim() == 2:
            # fused op is marginally faster
            return input1.mm(self.weight).mm(input2)

        return input1.matmul(self.weight).matmul(input2)

    def extra_repr(self):
        return 'in1_features={}, in2_features={}'.format(
            self.in1_features, self.in2_features
        )
