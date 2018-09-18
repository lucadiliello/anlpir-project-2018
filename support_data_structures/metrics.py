import numpy
import torch
import torch.nn as nn

def accuracy(output, labels):
    assert(len(output) == len(labels))
    true = (labels == output).sum().item()
    return true / len(output) if len(output) else 0

def precision(output, labels):
    true_pos = (labels.__and__(output)).sum().item()
    false_pos = ((1 - labels).__and__(output)).sum().item()
    return true_pos / (true_pos+false_pos) if (true_pos+false_pos) else 0

def recall(output, labels):
    true_pos = (labels.__and__(output)).sum().item()
    false_neg = (labels.__and__(1 - output)).sum().item()
    return true_pos / (true_pos+false_neg) if (true_pos+false_neg) else 0

def F1score(output, labels):
    p = precision(output, labels)
    r = recall(output, labels)
    return (2 * p * r) / (p + r)

def MAP(output, labels):
    pass

def MRR(output, labels):
    pass
