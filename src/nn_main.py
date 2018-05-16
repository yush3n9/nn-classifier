#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from gensim.models import Word2Vec
from keras.layers import Embedding, Dense, Flatten, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from src.data.newsgroup_dataprovider import TwentyNewsgroup

# ----------------------
# Fetch train dataset:
# ----------------------
dp = TwentyNewsgroup()
VEC_DIM = 100
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
NUM_CATEGORIES = dp.categories_size()

# ----------------------
# build w2v model:
# ----------------------
w2v_model = Word2Vec(dp, iter=5, size=VEC_DIM, min_count=1)
vocab_size = len(w2v_model.wv.vocab)
weights = np.array(w2v_model.wv.syn0)
assert vocab_size == weights.shape[0], "vocab_size must be same as weights.shape[0]"
assert VEC_DIM == weights.shape[1], "vec_dim must be same as weights.shape[1]"

# ----------------------
# prepare dataset:
# ----------------------
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(dp.fetch_dataset_train().data)
word_index = tokenizer.word_index
X_train = pad_sequences(tokenizer.texts_to_sequences(dp.fetch_dataset_train().data), maxlen=MAX_SEQUENCE_LENGTH)
X_test = pad_sequences(tokenizer.texts_to_sequences(dp.fetch_dataset_test().data), maxlen=MAX_SEQUENCE_LENGTH)
y_train = dp.fetch_dataset_train().target
y_test = dp.fetch_dataset_test().target
assert vocab_size == len(word_index), "vocab_size must be same as len(word_index)"

# ----------------------
# build embedding matrix:
# ----------------------
embedding_matrix = np.zeros((len(word_index) + 1, VEC_DIM))
for word, i in word_index.items():
    embedding_vector = w2v_model[word]
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# -------------------------------------
# build embedding layer:
# -------------------------------------
embedding_layer = Embedding(input_dim=vocab_size + 1, output_dim=VEC_DIM, weights=[embedding_matrix],
                            trainable=False, input_length=MAX_SEQUENCE_LENGTH)

keras_model = Sequential()
keras_model.add(embedding_layer)
keras_model.add(Dense(20))
keras_model.add(Dropout(0.5))
keras_model.add(Activation('relu'))
keras_model.add(Flatten())
keras_model.add(Dense(1, activation='sigmoid'))
keras_model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['acc'])

# keras_model = Sequential()
# e = Embedding(vocab_size + 1, VEC_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)
# keras_model.add(e)
# keras_model.add(Flatten())
# keras_model.add(Dense(1, activation='sigmoid'))
# keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(keras_model.summary())

keras_model.fit(X_train, y_train, epochs=50, verbose=0)
loss, accuracy = keras_model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %f' % (accuracy * 100))
