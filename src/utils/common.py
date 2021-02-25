# -*- coding: utf-8 -*-
# @Time    : 05/12/2020 10:21 AM
# @File    : common.py

# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import os
import numpy as np
import _pickle as pickle
from pyltp import Segmentor, Postagger
import nltk
import importlib

importlib.reload(sys)
# sys.setdefaultencoding('utf8')


def _readFile(path):
    if not os.path.exists(path):
        print("No such file or directory:", path)
        return False

    content = []
    with open(path, "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            content.append(line.strip())
    return content


def _writeFile(path, contents):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(dir_path + ' create successfully!')

    with open(path, "w", encoding='utf-8') as f:
        for line in contents:
            f.write(line + "\n")
    return True


def _readBunch(path):
    if not os.path.exists(path):
        print("No such file or directory:", path)
        return False

    with open(path, "rb") as f:
        bunch = pickle.load(f)
    return bunch


def _writeBunch(path, bunch):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(dir_path + ' create successfully!')

    with open(path, "wb") as f:
        pickle.dump(bunch, f)
    return True


def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(dir_path + ' create successfully!')


class docHandlerC:
    # Preprocessing of Chinese corpus
    def __init__(self, ltp_dir):
        self.ltp_dir = ltp_dir
        self.cws_model_path = os.path.join(self.ltp_dir, "cws.model")
        self.pos_model_path = os.path.join(self.ltp_dir, "pos.model")
        self.userdict_path = os.path.join(self.ltp_dir, "userdict.txt")

        self.segmentor = Segmentor()
        self.segmentor.load_with_lexicon(self.cws_model_path, self.userdict_path)

        self.postagger = Postagger()
        self.postagger.load(self.pos_model_path)
        self.judge_word = judgeWordC(self.ltp_dir)

    def hander(self, raw_doc, filter_pos=False):
        """
        :param raw_doc: raw corpus
        :return: raw_words: word list after word segmentation
                 doc_words: word list after filtration by the pos tags
                 doc_pos: POS list after filtration by the pos tags
        """

        raw_words = list(self.segmentor.segment(raw_doc))
        raw_postags = list(self.postagger.postag(raw_words))
        raw_words_u = [w.decode('utf8') for w in raw_words]
        doc_words = []
        doc_pos = []
        for word, pos in zip(raw_words_u, raw_postags):
            if self.judge_word.useful(word, pos, filter_pos):
                doc_words.append(word)
                doc_pos.append(pos)
        return raw_words_u, doc_words, doc_pos

    def cut(self, raw_doc):
        """
        word segmentation
        :param raw_doc:
        :return:
        """
        raw_words = list(self.segmentor.segment(raw_doc))
        raw_postags = list(self.postagger.postag(raw_words))
        # raw_words_u = [w.decode('utf8') for w in raw_words]
        return raw_words, raw_postags

    def close(self):
        self.segmentor.release()
        self.postagger.release()


class judgeWordC:
    def __init__(self, ltp_dir):
        self.ltp_dir = ltp_dir

        self.stopword_path = os.path.join(self.ltp_dir, "stopwords.txt")
        # self.stopwords = set([w.decode('utf8') for w in _readFile(self.stopword_path)])
        self.stopwords = set(_readFile(self.stopword_path))
        self.key_pos = set(['ns', 'ni', 'nh', 'j', 'n', 'nz', 'v', 'a', 'i'])

    def is_chinese(self, word):
        if len(word) <= 1:
            return False

        for uchar in word:
            if uchar < u'\u4e00' or uchar > u'\u9fa5':
                return False
        return True

    def useful(self, word, pos, filter_pos=False):
        """
     
        """
        if word not in self.stopwords and self.is_chinese(word):
            if not filter_pos:
                return True
            else:
                return pos in self.key_pos
        else:
            return False

    def pos2num(self, pos):
        """
       
        """
        q = -1
        if pos == 'ns':  # geographical name
            q = 0
        elif pos in ['ni', 'nh', 'j']:  # ni:organization name; nh:person name; j:abbreviation
            q = 1
        elif pos in ['n', 'nz', 'v', 'a', 'i']:  # noun; other proper noun; verb; adjective; idiom
            s = np.random.random() < 0.9  
            q = 2 if s == 1 else 3
        return q


class docHandlerE:
    # Preprocessing of English corpus
    def __init__(self):
        self.judge_word_e = judgeWordE()

    def hander(self, raw_doc, filter_pos=False):
        """
        
        """
        raw_words = nltk.word_tokenize(raw_doc.strip())  # word segmentation
        raw_words = [word for word in raw_words if word.replace('-', '').isalpha()]  # 
        raw_postags = nltk.pos_tag(raw_words)
        raw_ners = nltk.ne_chunk(raw_postags)

        doc_words = []
        doc_pos = []
        for tagged_tree in raw_ners:
            # extract only chunks having NE labels
            if hasattr(tagged_tree, 'label'):  # Named Entities
                word = ' '.join(c[0] for c in tagged_tree.leaves())  # get NE name
                pos = tagged_tree.label()  # get NE category
            else:
                word, pos = tagged_tree

            word = word.lower()
            if self.judge_word_e.useful(word, pos, filter_pos):
                doc_words.append(word)
                doc_pos.append(pos)
        return raw_words, doc_words, doc_pos

    def cut(self, raw_doc):
        """
        word segmentation
        :param raw_doc:
        :return:
        """
        raw_words = nltk.word_tokenize(raw_doc.strip())  # word segmentation
        raw_words = [word for word in raw_words if word.replace('-', '').isalpha()]  # character
        raw_postags = nltk.pos_tag(raw_words)
        raw_ners = nltk.ne_chunk(raw_postags)

        for i, tagged_tree in enumerate(raw_ners):
            # extract only chunks having NE labels
            if hasattr(tagged_tree, 'label'):  # NE
                word = ' '.join(c[0] for c in tagged_tree.leaves())  # get NE name
                pos = tagged_tree.label()  # get NE category
            else:
                word, pos = tagged_tree

            raw_words[i] = word.lower()
            raw_postags[i] = pos
        return raw_words, raw_postags


class judgeWordE:
    def __init__(self):
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.key_pos = set(['PERSON', 'ORGANIZATION', 'FACILITY', 'GPE', 'LOCATION',
                            'JJ', 'JJR', 'JJS',  # adjective, Comparison of Adjectives, Adjectives the highest level
                            'NN', 'NNS', 'NNP', 'NNPS',  # noun, noun plurals, proper noun, Plural proper nouns
                            'VB', 'VBD', 'VBG', 'VBN', 'VBP',
                            'VBZ'])  # verb, past tense of verb, Present participle of the verb, Past participle of the verb, The present tense of the verb that is not the third person tense, The third person tense of verb present tense

    def useful(self, word, pos, filter_pos=False):
        """
        Preprocessing
        """
        if word not in self.stopwords and len(word) > 2:
            if not filter_pos:
                return True
            else:
                return pos in self.key_pos
        else:
            return False

    def pos2num(self, pos):
        """
       
        """
        q = -1
        if pos in set(['GPE', 'LOCATION']):
            q = 0
        elif pos in set(['PERSON', 'ORGANIZATION']):
            q = 1
        elif pos in set(['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']):
            s = np.random.random() < 0.9  # threshold parameter
            q = 2 if s == 1 else 3
        return q


class judgeWordEBert:
    def __init__(self):
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.key_pos = set(['PER', 'ORG', 'LOC', 'MISC',
                            'JJ', 'JJR', 'JJS',  
                            'NN', 'NNS', 'NNP', 'NNPS',  
                            'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
        

    def useful(self, word, pos, filter_pos=False):
        """
        
        """
        if word not in self.stopwords and len(word) > 2:
            if not filter_pos:
                return True
            else:
                return pos in self.key_pos
        else:
            return False

    def pos2num(self, pos):
        """
        
        """
        q = -1
        if pos == 'LOC':
            q = 0
        elif pos in set(['PER', 'ORG']):
            q = 1
        elif pos in set(
                ['MISC', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']):
            s = np.random.random() < 0.9  # threshold parameter
            q = 2 if s == 1 else 3
        return q
