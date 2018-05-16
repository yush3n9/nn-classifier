#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import keras.preprocessing.text as keras_prep
from tqdm import tqdm


class IterableDataProvider(ABC):
    @abstractmethod
    def inspect_dataset(self):
        '''
        log head of dataset out to check if the dataset is empty or not
        :return:
        '''
        pass

    @abstractmethod
    def fetch_dataset_train(self):
        pass

    @abstractmethod
    def fetch_dataset_test(self):
        pass

    @abstractmethod
    def categories_size(self):
        pass

    def __iter__(self):
        '''
        Tokenize the dataset
        :return:
        '''
        data = self.fetch_dataset_train().data
        for post in tqdm(data, desc="Tokenize"):
            yield keras_prep.text_to_word_sequence(post)
