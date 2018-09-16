import random
import torch
from random import shuffle

class BatchCreator(object):

    def __init__(self, data, batch_size, device):
        super(BatchCreator, self).__init__()
        shuffle(data)
        self.data = self.elaborate_data(data)
        self.batch_size = batch_size
        self.device = device
        self.last_index = 0

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

    def reset_index(self):
        self.last_index = 0

    def question_to_batch(self, index):
        entry = self.data[index]
        risposte = [random.choice(entry['candidates_pos'])] + (random.sample(entry['candidates_neg'], self.batch_size) if len(entry['candidates_neg']) > self.batch_size else entry['candidates_neg'])
        domande = [entry['question']] * len(risposte)
        return torch.tensor(domande, requires_grad=False).to(self.device), torch.tensor(risposte, requires_grad=False).to(self.device), torch.tensor([1] + [0] * (len(risposte)-1)).to(self.device)

    def next(self):
        index = random.randint(0, len(self.data) - 1)
        return self.question_to_batch(index)

    def ordered_next(self):
        if self.last_index >= len(self.data):
            self.reset_index()
            return None
        res = self.question_to_batch(self.last_index)
        self.last_index += 1
        return res
