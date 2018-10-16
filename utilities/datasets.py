from random import shuffle
import random
import torch
import numpy
from utilities import sprint
from functools import reduce

class DatasetManager(object):

    def __init__(self, data, word2index, device, hard_negative_training=False, batch_size=20, max_negative_answer_count=20):
        super(DatasetManager, self).__init__()
        self.hard_negative_training = hard_negative_training
        self.data = data
        self.device = device
        self.batch_size = batch_size
        ### used only if hard_negative_training is True
        self.max_negative_answer_count = max_negative_answer_count

        self.word2index = word2index
        self.max_question_len = 0
        self.max_answer_len = 0

        ### REMOVE USELESS ENTRIES
        sprint.p('Removing entries without both positive and negative answers', 3)
        self.remove_entries_without_a_labels()

        ### CLEANING AND REMOVING WORDS NOT IN THE GOOGLE MODEL
        sprint.p('Cleaning', 3)
        self.cleaning()

        ### FIND MAX QUESTIONS AND ANSWER LENGTH
        sprint.p('Finding max lengths', 3)
        self.set_max_len()

        ### WORD-2-INDEX AND PADDING
        sprint.p('Word2index and padding', 3)
        self.WI_and_padding()

        ### SHUFFLE DATA FOR BETTER TRAINING
        self.reset()

    def reset(self):
        self.index = 0
        shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def set_max_len(self):
        self.max_question_len = max(list(map(lambda e: len(e['question']), self.data)))
        self.max_answer_len = max( list( map(lambda e: max( max(list(map(lambda x: len(x), e['candidates_pos']))),  max(list(map(lambda x: len(x), e['candidates_neg']))) ), self.data) ) )

    def next(self):
        if self.index >= self.__len__():
            return None

        batch = self.data[self.index : self.index+self.batch_size]
        self.index += self.batch_size

        res_sizes = []
        res_domande = []
        res_risposte = []
        res_target = []

        for entry in batch:
            if self.hard_negative_training:
                risposte = [random.choice(entry['candidates_pos'])] + (random.sample(entry['candidates_neg'], self.max_negative_answer_count) if len(entry['candidates_neg']) > self.max_negative_answer_count else entry['candidates_neg'])
                target = [1.] + [0.] * (len(risposte) - 1)
            else:
                risposte = entry['candidates_pos'] + entry['candidates_neg']
                target = [1.] * len(entry['candidates_pos']) + [0.] * len(entry['candidates_neg'])

            domande = [entry['question']] * len(risposte)

            res_domande += domande
            res_risposte += risposte
            res_target += target
            res_sizes.append(len(risposte))
        # sizes, question, answers, targets
        return (res_sizes, torch.tensor(res_domande, requires_grad=False, device=self.device), torch.tensor(res_risposte, requires_grad=False, device=self.device), torch.tensor(res_target, requires_grad=False, device=self.device))

    def remove_entries_without_a_labels(self):
        res = []
        for entry in self.data:
            if (len(entry['candidates_pos']) > 0) and (len(entry['candidates_neg']) > 0):
                res.append(entry)
        self.data = res

    def cleaning(self):
        clean = lambda a: [word for word in a if word in self.word2index]
        for i in range(len(self.data)):
            self.data[i]['question'] = clean(self.data[i]['question'])
            self.data[i]['candidates_pos'] = [clean(sentence) for sentence in self.data[i]['candidates_pos']]
            self.data[i]['candidates_neg'] = [clean(sentence) for sentence in self.data[i]['candidates_neg']]


    def WI_and_padding(self):
        for i in range(len(self.data)):
            self.data[i]['question'] = [self.word2index[x] for x in self.data[i]['question']] + [0] * (self.max_question_len - len(self.data[i]['question']))
            self.data[i]['candidates_pos'] = [([self.word2index[x] for x in sentence] + [0] * (self.max_answer_len - len(sentence))) for sentence in self.data[i]['candidates_pos']]
            self.data[i]['candidates_neg'] = [([self.word2index[x] for x in sentence] + [0] * (self.max_answer_len - len(sentence))) for sentence in self.data[i]['candidates_neg']]

    ## statistics
    def get_statistics(self):
        return ( numpy.mean(list(map(lambda a: len(a['candidates_pos']), self.data))), ## average_number_pos_answers
            numpy.mean(list(map(lambda a: len(a['candidates_neg']), self.data))), ## average_number_neg_answers
            numpy.mean(list(map(lambda a: len(a['question']), self.data))), ## average_question_len
            numpy.mean(reduce(lambda a,b: a+b, map(lambda e: list(map(lambda x: len(x), e['candidates_pos'] + e['candidates_neg'])), self.data ))) ) ## average_answer_len
