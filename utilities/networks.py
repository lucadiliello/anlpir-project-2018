import torch
import torch.nn as nn

class QA_CNN(nn.Module):

    def __init__(self, max_len, embedding_size, convolutional_filters, context_len, device):
        super(QA_CNN, self).__init__()

        self.device = device
        self.max_len = max_len
        self.embedding_size = embedding_size
        self.convolutional_filters = convolutional_filters
        self.context_len = context_len

        self.conv2 = nn.Conv2d(1, self.convolutional_filters, (self.embedding_size * self.context_len, 1))
        #self.lin = nn.Linear(self.context_len * self.embedding_size, self.convolutional_filters, bias=True)

    def forward(self, x):
        ## x: bs * M * d

        x = self.sentence_to_Z_vector(x)
        ## bs * M * dk

        '''
        x = self.lin(x)
        ## bs * M * c
        x.transpose_(1,2)
        print(x.size())
        # bs * c * M
        return x
        '''

        x = x.view(-1, 1, self.max_len, self.embedding_size * self.context_len)
        ## bs * 1 * M * dk

        x = x.transpose(2,3)
        ## bs * 1 * dk * M

        x = self.conv2(x)
        ## bs * c * 1 * M

        x = x.view(-1, self.convolutional_filters, self.max_len)
        ## bs * c * M

        return x

    def sentence_to_Z_vector(self, sentences):
        tot = []
        for sentence in sentences:
            res = []
            for index in range(self.max_len):
                tmp = [None] * self.context_len
                for disc in range(-int(self.context_len/2), int(self.context_len/2) + 1):
                    tmp[disc + int(self.context_len/2)] = sentence[(index + disc)] if (index + disc) >= 0 and (index + disc) < len(sentence) else self.pad_tensor()
                res.append(torch.cat(tmp))
            tot.append(torch.stack(res))
        return torch.stack(tot)

    def pad_tensor(self):
        return torch.FloatTensor([.0] * self.embedding_size).to(self.device)


class QA_biLSTM(nn.Module):

    def __init__(self, max_len, embedding_size, hidden_dim, device):
        super(QA_biLSTM, self).__init__()

        hidden_dim = int(hidden_dim/2)
        self.device = device
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.embedding_size = embedding_size

        self.lstm = nn.LSTM(embedding_size, hidden_dim, bidirectional = True)

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

class AttentivePoolingNetwork(nn.Module):

    def __init__(self, max_len, embedding_size, device, type_of_nn='CNN', convolutional_filters=400, context_len=3):
        super(AttentivePoolingNetwork, self).__init__()

        self.device = device
        self.M, self.L = max_len

        ## params of the CNN
        self.convolutional_filters = convolutional_filters
        self.context_len = context_len
        self.embedding_size = embedding_size

        ## CNN or biLSTM
        if type_of_nn == 'CNN':
            self.question_cnn_bilstm = QA_CNN(self.M, self.embedding_size, self.convolutional_filters, self.context_len, self.device)
            self.answer_cnn_bilstm = QA_CNN(self.L, self.embedding_size, self.convolutional_filters, self.context_len, self.device)
        elif type_of_nn == 'biLSTM':
            self.question_cnn_bilstm = QA_biLSTM(self.M, self.embedding_size, self.convolutional_filters, self.device)
            self.answer_cnn_bilstm = QA_biLSTM(self.L, self.embedding_size, self.convolutional_filters, self.device)
        else:
            raise ValueError('Mode must be CNN or biLSTM')

        self.U = torch.nn.Parameter(torch.Tensor(self.convolutional_filters, self.convolutional_filters))

        self.tanh = nn.Tanh()
        self.extract_q_feats = nn.MaxPool1d(self.L)
        self.extract_a_feats = nn.MaxPool1d(self.M)

        self.softmax = nn.Softmax(dim=0)

        self.sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, question, answer):
        ## get batch_size
        batch_size = question.size()[0]

        ## question: bs * M * d
        ## answer: bs * L * d

        Q = self.question_cnn_bilstm(question)
        ## bs * c * M

        A = self.answer_cnn_bilstm(answer)
        ## bs * c * L

        ## transpose does not make a copy of the tensor, it only swaps the access indexes.
        ## To make a complete new copy use .contiguous()
        Q = Q.transpose(1,2).contiguous()
        ## bs * M * c

        res = torch.mm(Q.view(batch_size * self.M, self.convolutional_filters), self.U).view(batch_size, self.M, self.convolutional_filters).bmm(A)
        ## bs * M * L

        G = self.tanh(res)
        ## bs * M * L

        max_pool_Q = self.extract_q_feats(G).view(-1, self.M, 1)
        ## bs * M * 1

        max_pool_A = self.extract_a_feats(G.transpose(1,2)).view(-1, self.L, 1)
        ## bs * L * 1

        Q = Q.transpose(1,2)
        ## bs * c * M

        roQ = self.softmax(max_pool_Q)
        ## bs * M * 1
        roA = self.softmax(max_pool_A)
        ## bs * L * 1

        rQ = Q.bmm(roQ).view(-1, self.convolutional_filters)
        rA = A.bmm(roA).view(-1, self.convolutional_filters)
        # rA=rQ : bs * c

        return self.sim(rQ, rA)
        ## bs
