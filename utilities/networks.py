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

        #self.conv = nn.Conv2d(1, self.convolutional_filters, (self.context_len, self.embedding_size), padding=(1,0))
        #self.conv = nn.Conv1d(1, self.convolutional_filters, self.embedding_size * self.context_len)
        #self.W = nn.Parameter(torch.Tensor(self.embedding_size * self.context_len, self.convolutional_filters))
        #self.b = nn.Parameter(torch.Tensor(self.convolutional_filters))


    def forward(self, question, answer):
        ## question: bs * M * d
        ## answer: bs * L * d
        '''
        assert(question.size()[0] == answer.size()[0])
        batch_size = question.size()[0]

        question = self.sentences2Z(question, self.max_len_Q)
        ## bs * M * dk
        answer = self.sentences2Z(answer, self.max_len_A)
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

        question = self.sentences2Z(question, self.max_len_Q)
        ## bs * M * dk
        answer = self.sentences2Z(answer, self.max_len_A)
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

    def sentences2Z(self, sentences, length):
        res = []
        for sentence in sentences:
            res.append(self._sentece2Z(sentence, length))
        return torch.stack(res, dim=0).to(self.device)

    def _sentece2Z(self, sentence, length):
        assert(len(sentence) == length)
        res = torch.zeros(length, self.embedding_size * self.context_len)

        for index in range(len(sentence)):
            tmp = []
            for j, i in self.context2indexes(index):
                if i >= 0 and i < len(sentence):
                    tmp.append(sentence[i])
                else:
                    tmp.append(torch.zeros(self.embedding_size))
            res[index] = torch.cat(tmp)
        return res

    def context2indexes(self, i):
        return zip(range(self.context_len), range(i - int(self.context_len/2), i + int(self.context_len/2) + 1))


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

    def __init__(self, max_len, word_embedding_dims, device, word_embedding_model=None, type_of_nn='CNN', convolutional_filters=400, context_len=3):
        super(AttentivePoolingNetwork, self).__init__()

        self.device = device
        self.M, self.L = max_len
        self.convolutional_filters = convolutional_filters
        self.context_len = context_len
        self.vocab_size, self.embedding_size = word_embedding_dims

        if word_embedding_model:
            weights = torch.tensor(word_embedding_model.wv.syn0)
            weights = torch.cat((torch.zeros(1,self.embedding_size), weights), dim=0)
            self.embedding_layer = nn.Embedding(self.vocab_size + 1, self.embedding_size, padding_idx=0).from_pretrained(weights, freeze=True)
        else:
            self.embedding_layer = nn.Embedding(self.vocab_size + 1, self.embedding_size, padding_idx=0)

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

    def forward(self, question, answer):
        ## question: bs * M
        ## answer: bs * L

        question = self.embedding_layer(question)
        ## bs * M * d
        answer = self.embedding_layer(answer)
        ## bs * M * d

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

        return torch.nn.functional.cosine_similarity(rQ, rA, dim=1, eps=1e-08)
        ## bs
