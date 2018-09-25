import numpy
import torch
import torch.nn as nn

'''
def accuracy(output, labels):
    assert(len(output) == len(labels))
    true = (labels == output).sum().item()
    return true / len(output) if len(output) else 0

def recall(output, labels):
    true_pos = (labels.__and__(output)).sum().item()
    false_neg = (labels.__and__(1 - output)).sum().item()
    return true_pos / (true_pos+false_neg) if (true_pos+false_neg) else 0
'''

def precision(output):
    value = max(output)
    output = (output == value).long()
    labels = torch.tensor([1] + [0] * (len(output) - 1))
    true_pos = (labels.__and__(output)).sum().item()
    false_pos = ((1 - labels).__and__(output)).sum().item()
    return true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0

def MAP(output_array):
    res = []
    size = len(output_array)
    for i in range(size):
        res.append(precision(output_array[i]))
    return numpy.mean(res)


def RR(output):
    output = output.tolist()
    first = output[0]
    output.sort(reverse=True)
    index = output.index(max(output))
    return 1.0/(index+1)

def MRR(output_array):
    res = []
    size = len(output_array)
    for i in range(size):
        res.append(RR(output_array[i]))
    return numpy.mean(res)
