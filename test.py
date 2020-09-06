import numpy as np
import pickle, pdb
from model import word2vec_tf
from utils_modified import nearest_word
from config import EMBEDDING_DIM, MODEL_DIR, DATA_DICT_PATH
import tensorflow as tf

def q():
    sys.exit()

def print_nearest_words_tf(model, test_words, word_to_ix, ix_to_word, top = 5):
    
    emb_matrix = model.embeddings_input(np.array(range(len(word_to_ix))))
    print('type(emb_matrix): ', type(emb_matrix))
    
    nearest_words_dict = {}

    print('==============================================')
    for t_w in test_words:
        
        inp_emb = emb_matrix[word_to_ix[t_w], :]  

        emb_ranking_top, _ = nearest_word(inp_emb, emb_matrix, top = top+1)
        print(t_w.ljust(10), ' | ', ', '.join([ix_to_word[i] for i in emb_ranking_top[1:]]))

    return nearest_words_dict

if __name__ == '__main__':

    # load data_dict pickle
    print('loading data_dict pickle...')
    infile = open(DATA_DICT_PATH,'rb')
    data_dict = pickle.load(infile)
    infile.close()
    print('pickle loaded')

    vocab = data_dict['vocab']
    word_to_ix = data_dict['word_to_ix']
    ix_to_word = data_dict['ix_to_word']

    model = word2vec_tf(EMBEDDING_DIM, len(vocab))#, noise_dist, NEGATIVE_SAMPLES)
    model.load_weights('{}/model{}'.format(MODEL_DIR, 1))
    
    EMBEDDINGS = model.embeddings_input(np.array(range(len(word_to_ix))))

    def vec( word):
        return EMBEDDINGS[word_to_ix[word], :]

    # inp = vec('king') - vec('man') + vec('woman')                                       
    # inp = vec('word1')
    inp = vec('desert')

    # print('inp.shape: ', inp.shape)

    emb_ranking_top, euclidean_dis_top = nearest_word(inp, EMBEDDINGS, top = 20)
    print('emb_ranking_top: ', emb_ranking_top, type(emb_ranking_top))

    for idx, t in enumerate(emb_ranking_top):
        print(ix_to_word[t], euclidean_dis_top[idx])