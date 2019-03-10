import gensim

class Loader:

    def __init__(self, pkg_name):
        if pkg_name in ['InsuranceQA', 'TrecQA', 'WikiQA']:
            self.pkg_name = pkg_name
        else:
            raise ValueError('argument should be one between InsuranceQA, TrecQA or WikiQA')

    def __str__(self):
        return 'Loader object of type %s' % self.pkg_name

    def get_vocabulary(self):
        docs = self.get_documents()
        vocab = dict()
        index = 1
        for sent in docs:
            for token in sent:
                if token not in vocab:
                    vocab[token] = index
                    index += 1
        return vocab

    def get_sentences(self, ds):
        res = []
        for row in ds:
            res.append(row['question'])
            for answer in row['candidates_pos'] + row['candidates_neg']:
                res.append(answer)
        return res

    def get_documents(self):
        documents = []
        for ds in self.load():
            documents += self.get_sentences(ds)
        return documents

    def organise_answers(self, data):
        res = []
        for entry in data:
            tmp = {}
            tmp['question'] = entry['question']
            tmp['candidates_pos'] = [cand['sentence'] for cand in entry['candidates'] if cand['label']]
            tmp['candidates_neg'] = [cand['sentence'] for cand in entry['candidates'] if not cand['label']]
            res.append(tmp)
        return res

    def clean(self, data):
        ## call always after self.organise_answers()
        res = []
        for entry in data:
            tmp = {}
            tmp['question'] = gensim.utils.simple_preprocess(entry['question'])
            tmp['candidates_pos'] = [gensim.utils.simple_preprocess(sentence) for sentence in entry['candidates_pos']]
            tmp['candidates_neg'] = [gensim.utils.simple_preprocess(sentence) for sentence in entry['candidates_neg']]
            res.append(tmp)
        return res

    def load(self):
        train, valid, test = [], [], []

        if self.pkg_name == 'InsuranceQA':
            raise NotImplementedError('InsuranceQA dataset not yet available')

        elif self.pkg_name == 'TrecQA':

            import jsonlines
            with jsonlines.open('data/Old-Trec-QA/valid.jsonl') as reader:
                for obj in reader:
                    obj['question'] = str(obj['question'])
                    obj['candidates'] = [{'sentence': str(cand['sentence']), 'label': cand['label']} for cand in obj['candidates']]
                    valid.append(obj)

            with jsonlines.open('data/Old-Trec-QA/train.jsonl') as reader:
                for obj in reader:
                    obj['question'] = str(obj['question'])
                    obj['candidates'] = [{'sentence': str(cand['sentence']), 'label': cand['label']} for cand in obj['candidates']]
                    train.append(obj)

            with jsonlines.open('data/Old-Trec-QA/test.jsonl') as reader:
                for obj in reader:
                    obj['question'] = str(obj['question'])
                    obj['candidates'] = [{'sentence': str(cand['sentence']), 'label': cand['label']} for cand in obj['candidates']]
                    test.append(obj)


        elif self.pkg_name == 'WikiQA':

            import csv
            valid = {}
            with open('data/WikiQA/WikiQA-dev.tsv', 'r') as csvfile:
                spamreader = csv.reader(csvfile, delimiter='\t', quotechar='"')
                next(spamreader, None)  # skip the headers
                for row in spamreader:

                    res = {}
                    res['sentence'] = str(row[5])
                    res['label'] = int(row[6])
                    row[1] = str(row[1])

                    if row[1] in valid:
                        valid[row[1]].append(res)
                    else:
                        valid[row[1]] = [res]

            valid = [ {'question': key, 'candidates': value} for key, value in valid.items()]

            train = {}
            with open('data/WikiQA/WikiQA-train.tsv', 'r') as csvfile:
                spamreader = csv.reader(csvfile, delimiter='\t', quotechar='"')
                next(spamreader, None)  # skip the headers
                for row in spamreader:

                    res = {}
                    res['sentence'] = str(row[5])
                    res['label'] = int(row[6])

                    row[1] = str(row[1])

                    if row[1] in train:
                        train[row[1]].append(res)
                    else:
                        train[row[1]] = [res]

            train = [ {'question': key, 'candidates': value} for key, value in train.items()]

            test = {}
            with open('data/WikiQA/WikiQA-test.tsv', 'r') as csvfile:
                spamreader = csv.reader(csvfile, delimiter='\t', quotechar='"')
                next(spamreader, None)  # skip the headers
                for row in spamreader:

                    res = {}
                    res['sentence'] = str(row[5])
                    res['label'] = int(row[6])

                    row[1] = str(row[1])

                    if row[1] in test:
                        test[row[1]].append(res)
                    else:
                        test[row[1]] = [res]

            test = [ {'question': key, 'candidates': value} for key, value in test.items()]

        else:
            raise ValueError('dataset type should be one between InsuranceQA, TrecQA or WikiQA')

        train = self.organise_answers(train)
        valid = self.organise_answers(valid)
        test = self.organise_answers(test)

        train = self.clean(train)
        valid = self.clean(valid)
        test = self.clean(test)

        return train, valid, test
