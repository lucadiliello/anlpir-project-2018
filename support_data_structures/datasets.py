import random
import torch
from random import shuffle

class BatchCreator(object):

    def __init__(self, data, batch_size, device):
        super(BatchCreator, self).__init__()
        self.data = self.elaborate_data(data)
        self.batch_size = batch_size
        self.device = device
        shuffle(data)

    def next(self):
        tmp = random.sample(self.data, self.batch_size)
        questions = []
        pos_array = []
        neg_array = []

        for entry in tmp:
            index = random.randint(0, len(entry['candidates_pos']) - 1)
            pos = entry['candidates_pos'][index]
            index = random.randint(0, len(entry['candidates_neg']) - 1)
            neg = entry['candidates_neg'][index]

            questions.append(entry['question'])
            pos_array.append(pos)
            neg_array.append(neg)

        return torch.tensor(questions, requires_grad=False).to(self.device), torch.tensor(pos_array, requires_grad=False).to(self.device), torch.tensor(neg_array, requires_grad=False).to(self.device)

    def elaborate_data(self, ds):
        ##organize candidates in pos and neg for simple extraction....
        res = []
        for entry in ds:
            tmp = {}
            tmp['question'] = entry['question']
            tmp['candidates_pos'] = [cand['sentence'] for cand in entry['candidates'] if cand['label'] == 1]
            tmp['candidates_neg'] = [cand['sentence'] for cand in entry['candidates'] if cand['label'] == 0]

            res.append(tmp)
        return res

    def get_all_data(self):
        pos_ques = []
        neg_ques = []
        pos_array = []
        neg_array = []

        for entry in self.data:
            for ans in entry['candidates_pos']:
                pos_ques.append(entry['question'])
                pos_array.append(ans)
            for ans in entry['candidates_neg']:
                neg_ques.append(entry['question'])
                neg_array.append(ans)

        return torch.tensor(pos_ques, requires_grad=False).to(self.device), torch.tensor(neg_ques, requires_grad=False).to(self.device), torch.tensor(pos_array, requires_grad=False).to(self.device), torch.tensor(neg_array, requires_grad=False).to(self.device)
