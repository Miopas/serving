#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''分类器，负责模型训练流程和预测流程.

    fit_with_file:从文件中加载训练集和验证集
    fit:意图分类模型的训练过程
    predict_with_file:意图分类模型 batch test 
    predict:意图分类模型预测接口
    save:将意图分类模型导出到模型文件中
    restore:从文件中加载模型
'''

import os
import sys
import time
import traceback
import logging

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.sparse import hstack

from util.preprocess.tfidf_processor import TFIDFProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(threadName)-10s %(message)s')
LOGGER = logging.getLogger('DataTransformer')
LOGGER.setLevel(logging.DEBUG)

class DataTransformer(object):

    def __init__(self, corpus_file, term_file=None, normalizer='basic_with_num'):
        '''
        initialize model
        Args:
            corpus_file: Fit it to TFIDFProcessor for learning idf vector. 
                         Usually it is the training data.
            term_file: for normalization
        Returns:
            None
        '''
        assert term_file is None or os.path.exists(term_file), "term_file not found in %s" % term_file
        assert os.path.exists(corpus_file), "corpus_file not found in %s" % corpus_file 
        
        # init TFIDFProcessor
        self.char_unigram_preprocessor = TFIDFProcessor(normalizer=normalizer, term_file=term_file, 
                                                   token_level='char', ngram_size=1)
        #self.char_bigram_preprocessor = TFIDFProcessor(normalizer=normalizer, term_file=term_file, 
        #                                           token_level='char', ngram_size=2)
        self.word_unigram_preprocessor = TFIDFProcessor(normalizer=normalizer, term_file=term_file, 
                                                   token_level='word', ngram_size=1)
        #self.word_bigram_preprocessor = TFIDFProcessor(normalizer=normalizer, term_file=term_file, 
        #                                           token_level='word', ngram_size=2)

        # learn idf vector
        corpus_data, labels = self.read_data(corpus_file)
        LOGGER.debug('corpus data size {}'.format(len(corpus_data))) 
        self.char_unigram_preprocessor.fit(corpus_data)
        #self.char_bigram_preprocessor.fit(corpus_data)
        self.word_unigram_preprocessor.fit(corpus_data)
        #self.word_bigram_preprocessor.fit(corpus_data)
        
        #对分类标签进行编码
        self.postprocessor = preprocessing.LabelEncoder()
        self.postprocessor.fit(labels)

        self.label_list = list(set(labels))
        #for i, label in enumerate(self.label_list):
        #    print('label map:{} -> {}'.format(label, self.postprocessor.transform([label])))

    def read_data(self, filepath):
        df = pd.read_csv(filepath, dtype=object)
        df = shuffle(df)
        return df['text'], df['class']

    def fit_with_file(self, train_file, n_class=None):
        '''
        从文件中加载训练集和验证集, 并转换成向量
        Args:
            train_file:训练集数据文件
        Returns:
            文本向量数组
            标签向量数组
        '''
        assert os.path.exists(train_file), "train_file invalid: {}".format(train_file)
        x_train, y_train = self.read_data(train_file)
        x_train, y_train = self.fit(x_train, y_train)

        # Generate labels
        y_train = self.generate_labels(y_train, n_class)
        return x_train, y_train

    def generate_labels(self, y_train, n_class):
        label_vec = np.zeros((len(y_train), n_class))
        for i, label in enumerate(y_train):
            label_vec[i][label] = 1
        return label_vec.astype(np.float32)

    def fit(self, x_train, y_train):
        '''
        Args:
            x_train, y_train:训练集的文本及对应的分类标签
        Returns:
            None
        '''
        #对文本进行字符串预处理、分词、提取特征，得到文本的特征向量
        LOGGER.debug('run extract_features')
        x_train = self.__extract_features(x_train)

        ##对分类标签进行编码
        #self.postprocessor = preprocessing.LabelEncoder()
        #self.postprocessor.fit(y_train)

        start_time = time.time()
        y_train = self.postprocessor.transform(y_train)
        end_time = time.time()
        return x_train, y_train

    def __extract_features(self, input_data):
        '''
        transform sentences into feature maxtrix
        Args:
            input_data: a list of sentences to transform
        Returns:
            a feature matrix of input_data of shape (n, m) in which 'n' is the number of sentences and 'm' is the size of vector
        '''
        x_feature_maxtrix = hstack([
                          self.char_unigram_preprocessor.transform(input_data), 
                          #self.char_bigram_preprocessor.transform(input_data),
                          self.word_unigram_preprocessor.transform(input_data),
                          #self.word_bigram_preprocessor.transform(input_data)
                         ], format='csr').toarray().astype(np.float32)
        LOGGER.debug('data size is %d' % (len(input_data))) 
        LOGGER.debug('data matrix shape is %s' % str(x_feature_maxtrix.shape)) 
        return x_feature_maxtrix


if __name__ == '__main__':
    training_data = './data/mojie/sample.csv'
    data_transformaer = DataTransformer(corpus_file=training_data)
    x_train, y_train = data_transformaer.fit_with_file(training_data)

    #print x_train, y_train
    
