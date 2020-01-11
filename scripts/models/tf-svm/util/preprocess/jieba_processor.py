#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''结巴分词器，对预处理后的文本进行分词.

    init:初始化分词器，初始化字符串处理器，从term_file中加载同义词词典
    transform:调用字符串处理器对文本进行预处理，然后切词
'''

import logging

import jieba
import pandas as pd
from processor import Processor
from string_utils import normalize
from string_utils import normalize_with_num

logging.basicConfig()
LOGGER = logging.getLogger('JiebaProcessor')
LOGGER.setLevel(level=logging.DEBUG)

class JiebaProcessor(Processor):

    def __init__(self, normalizer='basic_with_num', term_file=None):
        '''
        初始化分词器，初始化字符串处理器，从term_file中加载同义词词典
        Args:
            term_file:自定义词典
        Returns:
            None
        '''
        #加载字符串预处理器和同义词典
        self.normalizer = normalizer
        self.tokenizer = jieba.Tokenizer()
        self.synonym = {}
        if term_file is not None:
            df = pd.read_csv(term_file)
            for i, row in df.iterrows():
                word = unicode(str(row['word']), 'utf8')
                if self.normalizer == 'basic':
                    word = normalize(word)
                elif self.normalizer == 'basic_with_num':
                    word = normalize_with_num(word)
                else:
                    pass
                if len(word) == 0:
                    continue
                self.tokenizer.add_word(word)

                #替换同义词
                if row['synonym'] is not None:
                    synonym = unicode(str(row['synonym']), 'utf8')

                    if self.normalizer == 'basic':
                        synonym = normalize(synonym)
                    elif self.normalizer == 'basic_with_num':
                        synonym = normalize_with_num(synonym)
                    else:
                        pass

                    if len(synonym) == 0:
                        continue
                    self.tokenizer.add_word(synonym)

                    if word != synonym:
                        self.synonym[synonym] = word
        LOGGER.debug('init JiebaProcessor success')

    def fit(self, x):
        pass

    def transform(self, x, level='word'):
        '''
        将数据代入模型进行处理

        Args:
            x:待分词文本
            level: char or word
        Returns:
            分词结果
        '''
        new_x = []
        for s in x:
            tmp = None
            if isinstance(s, str):
                tmp = unicode(s, 'utf8')
            elif isinstance(s, unicode):
                tmp = s
            else:
                assert False, "input data format invalid"

            if self.normalizer == 'basic':
                tmp = normalize(tmp)
            elif self.normalizer == 'basic_with_num':
                tmp = normalize_with_num(tmp)
            elif self.normalizer == 'none':
                pass
            else:
                assert False, "invalid"

            if level == 'word':
                for k in self.synonym.keys():
                    tmp = tmp.replace(k, self.synonym[k])
                new_x.append(u' '.join(self.tokenizer.lcut(tmp)))
            elif level == 'char':
                new_x.append(u' '.join(self.char_tokenize(self.tokenizer.lcut(tmp))))

        return new_x

    @classmethod
    def has_hanzi(self, text):
        if isinstance(text, unicode):
            return any(u'\u4e00' <= char_txt <= u'\u9fff' for char_txt in text)
        else:
            return any(u'\u4e00' <= char_txt <= u'\u9fff' for char_txt in text.decode('utf8'))

    @classmethod
    def split_char(self, text):
        if isinstance(text, unicode):
            return [char_txt for char_txt in text]
        else:
            return [char_txt for char_txt in text.decode('utf8')]

    @classmethod
    def char_tokenize(self, word_tokens):
        char_tokens = []
        for word in word_tokens:
            if self.has_hanzi(word):
                char_tokens += self.split_char(word)
            else:
                char_tokens.append(word)
        return char_tokens


if __name__ == '__main__':
    x_train = [u'@kg.MutualFund 的收益如何', u'一年@kg.MutualFund 基金@托管费 收多少']
    tokenizer = JiebaProcessor(normalizer='basic_with_num')

    print tokenizer.transform(x_train, level='word')
    #print '\n'.join(tokenizer.transform(x_train, level='word'))
    #print '\n'.join(tokenizer.transform(x_train, level='char'))

    ## TODO: when input is a string, some bug happens
    #print('|||'.join(tokenizer.transform(u'@kg.MutualFund 的收益如何')))
    #print('|||'.join(tokenizer.transform(u'一年@kg.MutualFund 基金@托管费 收多少')))
