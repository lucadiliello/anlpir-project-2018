class Loader:

    def __init__(self, pkg_name):
        if pkg_name in ['InsuranceQA', 'TrecQA', 'WikiQA']:
            self.pkg_name = pkg_name
        else:
            raise ValueError('argument should be one between InsuranceQA, TrecQA or WikiQA')

    def __str__(self):
        return 'Loader object of type %s' % self.pkg_name

    def load(self):
        if self.pkg_name == 'InsuranceQA':
            raise NotImplementedError('InsuranceQA dataset not yet available')

        elif self.pkg_name == 'TrecQA':

            import jsonlines
            valid = []
            with jsonlines.open('data/Trec-QA/valid.jsonl') as reader:
                for obj in reader:
                    obj['question'] = str(obj['question'])
                    obj['candidates'] = [{'sentence': str(cand['sentence']), 'label': cand['label']} for cand in obj['candidates']]
                    valid.append(obj)

            train = []
            with jsonlines.open('data/Trec-QA/train.jsonl') as reader:
                for obj in reader:
                    obj['question'] = str(obj['question'])
                    obj['candidates'] = [{'sentence': str(cand['sentence']), 'label': cand['label']} for cand in obj['candidates']]
                    train.append(obj)

            test = []
            with jsonlines.open('data/Trec-QA/test.jsonl') as reader:
                for obj in reader:
                    obj['question'] = str(obj['question'])
                    obj['candidates'] = [{'sentence': str(cand['sentence']), 'label': cand['label']} for cand in obj['candidates']]
                    test.append(obj)

            return train, valid, test

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

            return train, valid, test

        else:
            raise ValueError('dataset type should be one between InsuranceQA, TrecQA or WikiQA')
