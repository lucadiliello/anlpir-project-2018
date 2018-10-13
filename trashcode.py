################################################################################
### TESTING PART - TO BE COMMENTED/DELETED IN FINAL RELEASE
################################################################################
'''
vocabulary = {value:key for (key, value) in vocabulary.items()}
vocabulary[0] = None
dataset.train_mode()

def get_original(voc, sentence):
    return ([voc[x] for x in sentence.data.tolist() if x])

bs = dataset.next(3)
question = bs[0][0]

original = get_original(vocabulary,question)
print(original)
print(question)

net = networks.AttentivePoolingNetwork((dataset.max_question_len, dataset.max_answer_len), (len(vocabulary), word_embedding_size), device, word_embedding_model=we_model, type_of_nn=network_type, convolutional_filters=convolutional_filters, context_len=k).to(device)
print(net.embedding_layer(question))
for word in original:
    print(we_model.wv[word])
exit()
'''
