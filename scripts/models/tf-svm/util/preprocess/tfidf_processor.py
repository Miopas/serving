#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''tfidf处理器，对分词后的文本进行统计词频和计算tfidf值，输出文本特征.

    init:初始化tfidf流程
    fit:初始化模型内部参数
    transform:将数据代入模型进行转换
'''

import logging

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import FunctionTransformer
import numpy as np

from jieba_processor import JiebaProcessor 


logging.basicConfig()
LOGGER = logging.getLogger('TFIDFProcessor')
LOGGER.setLevel(level=logging.DEBUG)

class TFIDFProcessor(object):

    def __init__(self, normalizer='basic', term_file=None, token_level='word', ngram_size=1):
        '''
        初始化tfidf流程，包括统计词频和计算tfidf值

        Args:
            term_file:同义词词典文件
            token_level: char-level or word-level
            ngram_size: the window size of ngram
        Returns:
            None
        '''
        self.tokenizer = JiebaProcessor(normalizer=normalizer, term_file=term_file)
        self.token_level = token_level
        self.ngram_size = ngram_size
        self.clf = Pipeline([
                ('vect', CountVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(ngram_size, ngram_size))),
                ('tfidf', TfidfTransformer())
            ])

    def fit(self, x):
        '''
        fit training data to TFIDFProcessor to learn idf vector
        Args:
            x: training data/corpus
        Returns:
            None
        '''
        tokenized_x = self.__tokenize(x)
        self.clf.fit(tokenized_x)

    def transform(self, x):
        '''
        Transform data into a tfidf feature matrix
        Args:
            x: input data, a list of sentences
        Returns:
            a sparse matrix of shape (n, m) in which 'n' is the size of data and 'm' is the size of vocabulary 
        '''
        #LOGGER.debug('input data: ["%s"]' % ('\",\"'.join(x)))

        tokenized_x = self.__tokenize(x)
        #LOGGER.debug('tokenize for param [%s] output: ["%s"]' % (self.get_params(), '\",\"'.join(tokenized_x)))

        #LOGGER.debug('ngram sequence for param [%s] output: ["%s"]' % 
        #                (self.get_params(), '\",\"'.join(self.clf.named_steps['vect'].get_feature_names())))
        #LOGGER.debug('vect for param [%s] output: \n%s' % 
        #                (self.get_params(), str(self.clf.named_steps['vect'].transform(tokenized_x))))
        #LOGGER.debug('tfidf for param [%s] output: \n%s' % 
        #                (self.get_params(), str(self.clf.named_steps['tfidf'].transform(self.clf.named_steps['vect'].transform(tokenized_x)))))

        return self.clf.transform(tokenized_x)

    def __process_ngram(self, tokenized_data):
        '''
        add start_pos_tag&end_pos_tag tokens for ngram
        Args:
            tokenized_data: a list of tokenized sentences
        Returns:
            a list of tokenized sentences with start_pos_tag&end_pos_tag tokens 
        '''
        return ['start_pos_tag ' + tokenized_s + ' end_pos_tag' for tokenized_s in tokenized_data] 
            
    def __tokenize(self, data):
        '''
        tokenize data with jieba and process for n-gram(n > 1)
        Args:
            data: a list of sentences
        Returns:
            a list of tokenized sentences
        '''
        tokenized_x = self.tokenizer.transform(data, self.token_level)
        #LOGGER.debug('jieba tokenizer for param [%s] output is: ["%s"]' % (self.get_params(), '\",\"'.join(tokenized_x)))

        # if ngram mode is enable, add start_pos_tag&end_pos_tag token
        if self.ngram_size > 1:
            tokenized_x = self.__process_ngram(tokenized_x)
            #LOGGER.debug('ngram processer for param [%s] output is: ["%s"]' % (self.get_params(), '\",\"'.join(tokenized_x)))
        return tokenized_x

    def get_params(self):
        return 'token_level = %s, ngram_size = %s' % (self.token_level, self.ngram_size)

if __name__ == '__main__':
    x_train = [u'@kg.MutualFund 的收益如何', u'一年@kg.MutualFund 基金@托管费 收多少']
    #x_train = [u'@kg.MutualFund 的收益如何']
    term_file = '/Users/gyt/work/text_classification/train_data/term_data_20181106.csv'

    preprocessor = TFIDFProcessor(normalizer='basic_with_num', term_file=term_file, 
                            token_level='char', ngram_size=2)
    preprocessor.fit(x_train)
    LOGGER.debug('CountVectorizer output is: [\"%s\"]' % '\",\"'.join(preprocessor.clf.named_steps['vect'].get_feature_names()))

    x_train_encode = preprocessor.transform(x_train)
    LOGGER.debug('transform output type is: %s' % str(type(x_train_encode)))
    print x_train_encode
    print type(x_train_encode)
