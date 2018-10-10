import torch
import torch.nn as nn

class QA_CNN(nn.Module):

    def __init__(self, embedding_size, convolutional_filters, context_len):
        super(QA_CNN, self).__init__()

        self.embedding_size = embedding_size
        self.convolutional_filters = convolutional_filters
        self.context_len = context_len

        self.conv = nn.Conv1d(self.embedding_size, self.convolutional_filters, self.context_len, padding=1)

    def forward(self, question, answer):
        ## question: bs * M * d
        ## answer: bs * L * d

        question = question.transpose(1, 2)
        ## bs * d * M
        answer = answer.transpose(1, 2)
        ## bs * d * L

        question = self.conv(question)
        ## bs * c * M
        answer = self.conv(answer)
        ## bs * c * L

        return (question, answer)


class AttentivePoolingNetwork(nn.Module):

    def __init__(self, word_embedding_dims, word_embedding_model=None, convolutional_filters=400, context_len=3):
        super(AttentivePoolingNetwork, self).__init__()

        self.convolutional_filters = convolutional_filters
        self.context_len = context_len
        self.vocab_size, self.embedding_size = word_embedding_dims

        weights = torch.tensor(word_embedding_model.wv.syn0)
        weights = torch.cat((torch.zeros(1,self.embedding_size), weights), dim=0)
        self.embedding_layer = nn.Embedding(self.vocab_size + 1, self.embedding_size, padding_idx=0).from_pretrained(weights, freeze=True)

        self.cnn_bilstm = QA_CNN(self.embedding_size, self.convolutional_filters, self.context_len)


    def forward(self, question, answer):
        ## question: bs * M
        ## answer: bs * L

        question = self.embedding_layer(question)
        ## bs * M * d
        answer = self.embedding_layer(answer)
        ## bs * M * d

        Q, A = self.cnn_bilstm(question, answer)
        ## Q: bs * c * M - A: bs * c * L


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
