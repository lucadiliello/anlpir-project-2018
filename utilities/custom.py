from utilities import sprint as sp
import gensim

sprint = sp.SPrint()

def word_embedding_model(documents, word_embedding_size, word_embedding_window, n_threads):
    sprint.p('WE custom model creation', 1)

    sprint.p('Training the word2vec encoder', 2)
    ## sudo pip3 install cython - to speed up embedding

    model = gensim.models.Word2Vec(documents, size=word_embedding_size, window=word_embedding_window, min_count=0, workers=n_threads)
    model.train(documents, total_examples=len(documents), epochs=100)

    return model
