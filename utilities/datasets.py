import random
import torch
import numpy
import gensim
from random import shuffle
from functools import reduce
from utilities import sprint

class DatasetManager(object):

    def __init__(self, datasets, batch_size, device, vocabulary):
        super(DatasetManager, self).__init__()
        self.last_index = 0
        self.batch_size = batch_size
        self.device = device

        self.vocabulary = vocabulary

        self.max_question_len = 0
        self.max_answer_len = 0

        self.train, self.valid, self.test = datasets

        ### init to train()
        self.data = self.train

        ### SHUFFLING DATASET
        #shuffle(self.train)
        #shuffle(self.valid)
        #shuffle(self.test)

        ### ORGANIZING ANSWERS
        sprint.p('Organizing answers per label', 1)
        self.train = self.organise_answers(self.train)
        self.valid = self.organise_answers(self.valid)
        self.test = self.organise_answers(self.test)
        sprint.p('Done', 2)

        ### REMOVE USELESS ENTRIES
        sprint.p('Removing entries without both labels', 1)
        self.train = self.remove_entries_without_both_labels(self.train)
        sprint.p('Train done', 2)
        self.valid = self.remove_entries_without_both_labels(self.valid)
        sprint.p('Validation done', 2)
        self.test = self.remove_entries_without_both_labels(self.test)
        sprint.p('Test done', 2)
        sprint.p("After label filtering: %d training elements, %d validation elements and %d test elements" % (len(self.train),len(self.valid),len(self.test)), 2)

        ### CLEANING AND REMOVING WORDS NOT IN THE GOOGLE MODEL
        sprint.p('Cleaning datasets and removing words not in the google model', 1)
        self.train = self.cleaning(self.train)
        sprint.p('Train done', 2)
        self.valid = self.cleaning(self.valid)
        sprint.p('Validation done', 2)
        self.test = self.cleaning(self.test)
        sprint.p('Test done', 2)

        ### SAVING ORIGINAL DATASETS
        self.original_train = self.train
        self.original_valid = self.valid
        self.original_test = self.test

        ### FIND MAX LENGTH OF QUESTIONS AND ANSWERS
        sprint.p('Finding max Q/A length', 1)
        self.find_max_len()
        sprint.p("Max question length: %d, max answer length: %d" % (self.max_question_len, self.max_answer_len), 2)

        ### WORD-2-INDEX AND PADDING
        self.train = self.WI_and_padding(self.train)
        sprint.p('Train done', 2)
        self.valid = self.WI_and_padding(self.valid)
        sprint.p('Validation done', 2)
        self.test = self.WI_and_padding(self.test)
        sprint.p('Test done', 2)


    def organise_answers(self, dataset):
        res = []
        for entry in dataset:
            tmp = {}
            tmp['question'] = entry['question']
            tmp['candidates_pos'] = [cand['sentence'] for cand in entry['candidates'] if cand['label']]
            tmp['candidates_neg'] = [cand['sentence'] for cand in entry['candidates'] if not cand['label']]
            res.append(tmp)
        return res


    def remove_entries_without_both_labels(self, dataset):
        res = []
        for entry in dataset:
            if len(entry['candidates_pos']) > 0 and len(entry['candidates_neg']) > 0:
                res.append(entry)
        return res


    def cleaning(self, dataset):
        clean = lambda a: [word for word in gensim.utils.simple_preprocess(a) if word in self.vocabulary]
        res = []
        for entry in dataset:
            tmp = {}
            tmp['question'] = clean(entry['question'])
            tmp['candidates_pos'] = [clean(sentence) for sentence in entry['candidates_pos']]
            tmp['candidates_neg'] = [clean(sentence) for sentence in entry['candidates_neg']]
            res.append(tmp)
        return res


    def find_max_len(self):
        find_max_len_Q = lambda ds: max(list(map(lambda e: len(e['question']), ds)))
        find_max_len_A = lambda ds: max( list( map(lambda e: max( max(list(map(lambda x: len(x), e['candidates_pos']))),  max(list(map(lambda x: len(x), e['candidates_neg']))) ), ds) ) )
        self.max_question_len = max(find_max_len_Q(self.train), find_max_len_Q(self.valid), find_max_len_Q(self.test))
        self.max_answer_len = max(find_max_len_A(self.train), find_max_len_A(self.valid), find_max_len_A(self.test))


    def WI_and_padding(self, dataset):
        WI_padding = lambda sentence, max_len: [self.vocabulary[x] for x in sentence] + [0] * (max_len - len(sentence))
        res = []
        for entry in dataset:
            tmp = {}
            tmp['question'] = WI_padding(entry['question'], self.max_question_len)
            tmp['candidates_pos'] = [WI_padding(sentence, self.max_answer_len) for sentence in entry['candidates_pos']]
            tmp['candidates_neg'] = [WI_padding(sentence, self.max_answer_len) for sentence in entry['candidates_neg']]
            res.append(tmp)
        return res


    def get_statistics(self):
        results = {}

        res = {}
        res['average_number_pos_answers'] = numpy.mean(list(map(lambda a: len(a['candidates_pos']), self.original_train)))
        res['average_number_neg_answers'] = numpy.mean(list(map(lambda a: len(a['candidates_neg']), self.original_train)))
        res['average_question_len'] = numpy.mean(list(map(lambda a: len(a['question']), self.original_train)))
        res['average_answer_len'] = numpy.mean(reduce(lambda a,b: a+b, map(lambda e: list(map(lambda x: len(x), e['candidates_pos'] + e['candidates_neg'])), self.original_train )))
        results['train'] = res

        res = {}
        res['average_number_pos_answers'] = numpy.mean(list(map(lambda a: len(a['candidates_pos']), self.original_valid)))
        res['average_number_neg_answers'] = numpy.mean(list(map(lambda a: len(a['candidates_neg']), self.original_valid)))
        res['average_question_len'] = numpy.mean(list(map(lambda a: len(a['question']), self.original_valid)))
        res['average_answer_len'] = numpy.mean(reduce(lambda a,b: a+b, map(lambda e: list(map(lambda x: len(x), e['candidates_pos'] + e['candidates_neg'])), self.original_valid )))
        results['valid'] = res

        res = {}
        res['average_number_pos_answers'] = numpy.mean(list(map(lambda a: len(a['candidates_pos']), self.original_test)))
        res['average_number_neg_answers'] = numpy.mean(list(map(lambda a: len(a['candidates_neg']), self.original_test)))
        res['average_question_len'] = numpy.mean(list(map(lambda a: len(a['question']), self.original_test)))
        res['average_answer_len'] = numpy.mean(reduce(lambda a,b: a+b, map(lambda e: list(map(lambda x: len(x), e['candidates_pos'] + e['candidates_neg'])), self.original_test )))
        results['test'] = res

        return results


    def train_mode(self):
        self.reset_index()
        self.data = self.train

    def validation_mode(self):
        self.reset_index()
        self.data = self.valid

    def test_mode(self):
        self.reset_index()
        self.data = self.test


    def reset_index(self):
        self.last_index = 0

    def question_to_batch(self, index, bs):
        dimension = bs
        ## First tuple is (q,a+), other tuples are (q,a-)
        entry = self.data[index]
        risposte = [random.choice(entry['candidates_pos'])] + (random.sample(entry['candidates_neg'], dimension) if len(entry['candidates_neg']) > dimension else entry['candidates_neg'])
        domande = [entry['question']] * len(risposte)
        return torch.tensor(domande, requires_grad=False).to(self.device), torch.tensor(risposte, requires_grad=False).to(self.device)

    def next(self, batch_size=None):
        index = random.randint(0, len(self.data) - 1)
        return self.question_to_batch(index, batch_size or self.batch_size)

    def ordered_next(self):
        if self.last_index >= len(self.data):
            self.reset_index()
            return None
        res = self.question_to_batch(self.last_index)
        self.last_index += 1
        return res
