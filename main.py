from __future__ import print_function
from tqdm import tqdm
# from tqdm import tqdm_gui
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys, pdb, os, shutil, pickle
from pprint import pprint 

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print('gpus: ', gpus)
# tf.config.experimental.set_memory_growth(gpus[0], True)

from model import word2vec_tf
from datasets import word2vec_dataset
from config import *
from test import print_nearest_words_tf
from utils_modified import q

# remove MODEL_DIR if it exists
if os.path.exists(MODEL_DIR):
    shutil.rmtree(MODEL_DIR)
# create fresh MODEL_DIR    
os.makedirs(MODEL_DIR)

write_summary = True
if write_summary:
    # SUMMARY_DIR is the path of the directory where the tensorboard SummaryWriter files are written
    # the directory is removed, if it already exists
    if os.path.exists(SUMMARY_DIR):
        shutil.rmtree(SUMMARY_DIR)

    # os.makedirs(SUMMARY_DIR)
    train_summary_writer = tf.summary.create_file_writer(SUMMARY_DIR)
    
    summary_counter = 0

# make training data and dataloader
if not os.path.exists(PREPROCESSED_DATA_PATH):
    train_dataset = word2vec_dataset(DATA_SOURCE, BATCH_SIZE, CONTEXT_SIZE, FRACTION_DATA, SUBSAMPLING, SAMPLING_RATE)

    if not os.path.exists(PREPROCESSED_DATA_DIR):
        os.makedirs(PREPROCESSED_DATA_DIR)

    # pickle dump
    print('\ndumping pickle...')
    outfile = open(PREPROCESSED_DATA_PATH,'wb')
    pickle.dump(train_dataset, outfile)
    outfile.close()
    print('pickle dumped\n')

else:
    # pickle load
    print('\nloading pickle...')
    infile = open(PREPROCESSED_DATA_PATH,'rb')
    train_dataset = pickle.load(infile)
    train_dataset.batch_size = BATCH_SIZE
    infile.close()
    print('pickle loaded\n')

print('len(train_dataset): ', len(train_dataset))

vocab = train_dataset.vocab
word_to_ix = train_dataset.word_to_ix
ix_to_word = train_dataset.ix_to_word

# saving some data information in a pickle file, to be used later while inference/testing 

# dump data_dict pickle
# OVERRITING THE OLDER PICKLE FILE
print('\ndumping data_dict pickle...')
outfile = open(DATA_DICT_PATH,'wb')

data_dict = {
    'vocab': vocab,
    'word_to_ix': word_to_ix,
    'ix_to_word': ix_to_word
    }

pickle.dump(data_dict, outfile)
outfile.close()
print('pickle dumped\n')

print('len(vocab): ', len(vocab), '\n')

# make noise distribution to sample negative examples from
word_freqs = np.array(list(vocab.values()))
unigram_dist = word_freqs/sum(word_freqs)
noise_dist = unigram_dist**(0.75)/np.sum(unigram_dist**(0.75))

model = word2vec_tf(EMBEDDING_DIM, len(vocab), noise_dist, NEGATIVE_SAMPLES)
optimizer = tf.keras.optimizers.Adam(LR)

print('TRAINING...')
for epoch in range(NUM_EPOCHS):
    print('\n===== EPOCH {}/{} ====='.format(epoch + 1, NUM_EPOCHS))    
    
    for batch_idx, (x, y) in enumerate(train_dataset):

        with tf.GradientTape() as tape:
            loss_value = model(x, y) # model is supposed to output the loss
    
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print('batch# ' + str(batch_idx+1).zfill(len(str(len(train_dataset)))) + '/' + str(len(train_dataset)) + ' | Loss: ' + str(round(loss_value.numpy(), 5)), end = '\r')
        
        if batch_idx%DISPLAY_EVERY_N_BATCH == 0 and DISPLAY_BATCH_LOSS:
            print(f'Batch: {batch_idx+1}/{len(train_dataset)}, Loss: {loss_value}')    
            # show 5 closest words to some test words
            print_nearest_words_tf(model, TEST_WORDS, word_to_ix, ix_to_word, top = 5)   
    
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss_value.numpy(), step=summary_counter)            
            summary_counter += 1

    # write embeddings every SAVE_EVERY_N_EPOCH epoch
    if epoch%SAVE_EVERY_N_EPOCH == 0:      
        model.save_weights('{}/model{}'.format(MODEL_DIR, epoch))