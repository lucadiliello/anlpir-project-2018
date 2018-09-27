import torch
import torch.nn as nn

class QA_CNN(nn.Module):

    def __init__(self, max_len_Q, max_len_A, embedding_size, convolutional_filters, context_len, device):
        super(QA_CNN, self).__init__()

        self.device = device
        self.max_len_Q = max_len_Q
        self.max_len_A = max_len_A
        self.embedding_size = embedding_size
        self.convolutional_filters = convolutional_filters
        self.context_len = context_len

        self.conv = nn.Conv2d(1, self.convolutional_filters, (self.embedding_size * self.context_len, 1))
        #self.conv = nn.Conv1d(1, self.convolutional_filters, self.embedding_size * self.context_len)
        #self.W = nn.Parameter(torch.Tensor(self.embedding_size * self.context_len, self.convolutional_filters))
        #self.b = nn.Parameter(torch.Tensor(self.convolutional_filters))


    def forward(self, question, answer):
        ## question: bs * M * d
        ## answer: bs * L * d
        '''
        assert(question.size()[0] == answer.size()[0])
        batch_size = question.size()[0]

        question = self.sentence_to_Z_vector(question, self.max_len_Q)
        ## bs * M * dk
        answer = self.sentence_to_Z_vector(answer, self.max_len_A)
        ## bs * L * dk

        Q = torch.Tensor(batch_size, self.max_len_Q, self.convolutional_filters).to(self.device)
        A = torch.Tensor(batch_size, self.max_len_A, self.convolutional_filters).to(self.device)

        for i in range(question.size()[0]):
            Q[i] = torch.addmm(self.b, question[i], self.W)
        ## bs * M * c
        for i in range(answer.size()[0]):
            A[i] = torch.addmm(self.b, answer[i], self.W)
        ## bs * L * c

        return Q.transpose(1,2), A.transpose(1,2)

        '''

        question = self.sentence_to_Z_vector(question, self.max_len_Q)
        ## bs * M * dk
        answer = self.sentence_to_Z_vector(answer, self.max_len_A)
        ## bs * L * dk

        question = question.unsqueeze(1).transpose(2,3)
        ## bs * 1 * dk * M
        answer = answer.unsqueeze(1).transpose(2,3)
        ## bs * 1 * dk * M

        question = self.conv(question).squeeze()
        ## bs * c * M
        answer = self.conv(answer).squeeze()
        ## bs * c * M

        return question, answer


    def sentence_to_Z_vector(self, sentences, length):
        tot = []
        for sentence in sentences:
            res = []
            for index in range(length):
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


class AttentivePoolingNetwork(nn.Module):

    def __init__(self, max_len, embedding_size, device, type_of_nn='CNN', convolutional_filters=400, context_len=3):
        super(AttentivePoolingNetwork, self).__init__()

        self.device = device
        self.M, self.L = max_len
        self.convolutional_filters = convolutional_filters
        self.context_len = context_len
        self.embedding_size = embedding_size

        ## CNN or biLSTM
        if type_of_nn == 'CNN':
            self.cnn_bilstm = QA_CNN(self.M, self.L, self.embedding_size, self.convolutional_filters, self.context_len, self.device)
        elif type_of_nn == 'biLSTM':
            self.question_cnn_bilstm = QA_biLSTM(self.M, self.embedding_size, self.convolutional_filters, self.device)
            self.answer_cnn_bilstm = QA_biLSTM(self.L, self.embedding_size, self.convolutional_filters, self.device)
        else:
            raise ValueError('Mode must be CNN or biLSTM')

        self.U = torch.nn.Parameter(torch.Tensor(self.convolutional_filters, self.convolutional_filters))
        torch.nn.init.normal_(self.U, mean=0, std=1)
        self.sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, question, answer):
        ## question: bs * M * d
        ## answer: bs * L * d

        Q, A = self.cnn_bilstm(question, answer)
        ## Q: bs * c * M - A: bs * c * L

        G = torch.tanh(Q.transpose(1,2).matmul(self.U).matmul(A))
        ## bs * M * L

        roQ = G.max(2)[0].softmax(dim=1)
        ## bs * M
        roA = G.max(1)[0].softmax(dim=1)
        ## bs * L

        rQ = Q.matmul(roQ.unsqueeze(2)).squeeze()
        ## bs * c
        rA = A.matmul(roA.unsqueeze(2)).squeeze()
        ## bs * c

        return self.sim(rQ, rA)
        ## bs
