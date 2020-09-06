########################## MODEL ARCHITECTURE ##########################

import tensorflow as tf

class word2vec_tf(tf.keras.Model):

    def __init__(self, embedding_size, vocab_size, noise_dist = None, negative_samples = 10):
        super(word2vec_tf, self).__init__()
        
        # self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        # self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

        self.embeddings_input   = tf.keras.layers.Embedding(vocab_size, embedding_size, embeddings_initializer='uniform', mask_zero=False)
        self.embeddings_context = tf.keras.layers.Embedding(vocab_size, embedding_size, embeddings_initializer='uniform', mask_zero=False)
        
        self.vocab_size = vocab_size
        self.negative_samples = negative_samples
        self.noise_dist = noise_dist

    def call(self, input_word, context_word):
        debug = not True
        
        if debug:print('input_word  : ', input_word.shape)
        if debug:print('context_word: ', context_word.shape)

        ##### computing out loss #####
        emb_input = self.embeddings_input(input_word)     # bs, emb_dim
        if debug:print('emb_input: ', emb_input.shape)

        emb_context = self.embeddings_context(context_word)  # bs, emb_dim
        if debug:print('emb_context: ', emb_context.shape)

        # POSITIVE SAMPLES
        emb_product = tf.keras.layers.dot([emb_input, emb_context], axes = (1, 1))  # bs
        if debug:print('emb_product.shape: ', emb_product.shape)

        out_loss = tf.squeeze(tf.math.log_sigmoid(emb_product), axis = 1)
        if debug:print('out_loss.shape: ', out_loss.shape)


        # NEGATIVE SAMPLES
        if self.negative_samples > 0:
            # print('\nNEG SAMPLES')
            # computing negative loss
            if self.noise_dist is None:
                # noise_dist = torch.ones(self.vocab_size) 
                noise_dist = tf.ones(self.vocab_size) 
            else:
                noise_dist = self.noise_dist

            noise_dist = tf.reshape(noise_dist, (1, self.vocab_size))
            if debug:print('noise_dist: ', noise_dist.shape)
        
            num_neg_samples_for_this_batch = context_word.shape[0]*self.negative_samples
            # negative_example = tf.compat.v1.multinomial(noise_dist, num_neg_samples_for_this_batch)#, replacement = True) # coz bs*num_neg_samples > vocab_size
            negative_example = tf.random.categorical(noise_dist, num_neg_samples_for_this_batch)#, replacement = True) # coz bs*num_neg_samples > vocab_size
            negative_example = tf.reshape(negative_example, (context_word.shape[0], self.negative_samples))
            if debug:print('negative_example: ', negative_example.shape)

            emb_negative = self.embeddings_context(negative_example) # bs, neg_samples, emb_dim
            if debug:print('emb_negative: ', emb_negative.shape)

            emb_product_neg_samples = tf.matmul(tf.math.negative(emb_negative), tf.expand_dims(emb_input, axis = 2)) # bs, neg_samples, 1
            if debug:print('emb_product_neg_samples: ', emb_product_neg_samples.shape)
        
            noise_loss = tf.math.reduce_sum(tf.squeeze(tf.math.log_sigmoid(emb_product_neg_samples), axis = 2), axis = 1) # bs
            if debug:print('noise_loss.shape: ', noise_loss.shape)

            total_loss = tf.reduce_mean(tf.math.negative(tf.math.add(out_loss, noise_loss)))    
            if debug:print('total_loss.shape: ', total_loss.shape)

            return total_loss

        return tf.reduce_mean(tf.math.negative(out_loss)) 

    def this(self):
        print('testing this!')