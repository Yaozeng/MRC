# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
:author: lxm
:description: 字典操作类
:ctime: 2018.07.10 17:44
:mtime: 2018.07.10 17:44
"""

import numpy as np
from gensim.models.keyedvectors import KeyedVectors


class Vocab(object):
    def __init__(self, filename=None, initial_tokens=None, lower=False):
        # word
        self.id2word = {}
        self.word2id = {}
        self.word_cnt = {}

        self.lower = lower  # lower fn
        self.word_embed_dim = None
        self.word_embeddings = None

        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.split_token = '<splitter>'

        self.initial_tokens = initial_tokens if initial_tokens is not None else []
        self.initial_tokens.extend([self.pad_token, self.unk_token,self.split_token])
        for token in self.initial_tokens:
            self.add_word(token)
        if filename is not None:
            self.load_from_file(filename)

    def load_from_file(self, file_path):
        for line in open(file_path, 'r'):
            token = line.rstrip('\n')
            self.add_word(token)

    def word_size(self):
        return len(self.id2word)

    def get_word_id(self, token):
        token = token.lower() if self.lower else token
        return self.word2id[token] if token in self.word2id else self.word2id[self.unk_token]

    def get_word_token(self, idx):
        return self.id2word[idx] if idx in self.id2word else self.unk_token

    def add_word(self, token, cnt=1):
        token = token.lower() if self.lower else token
        if token in self.word2id:
            idx = self.word2id[token]
        else:
            idx = len(self.id2word)
            self.id2word[idx] = token
            self.word2id[token] = idx
        if cnt > 0:
            if token in self.word_cnt:
                self.word_cnt[token] += cnt
            else:
                self.word_cnt[token] = cnt
        return idx

    def filter_words_by_cnt(self, min_cnt):
        filtered_tokens = [token for token in self.word2id if self.word_cnt[token] >= min_cnt]
        # rebuild the token x id map
        self.word2id = {}
        self.id2word = {}
        for token in self.initial_tokens:
            self.add_word(token, cnt=0)

        for token in filtered_tokens:
            self.add_word(token, cnt=0)

    def randomly_init_word_embeddings(self, embed_dim):
        self.word_embed_dim = embed_dim
        self.word_embeddings = np.random.rand(self.word_size(), embed_dim)
        for token in [self.pad_token, self.unk_token]:
            self.word_embeddings[self.get_word_id(token)] = np.zeros([self.word_embed_dim])


    def load_pretrained_word_embeddings(self, embedding_path):
        trained_embeddings = {}
        w2v = KeyedVectors.load_word2vec_format(embedding_path, binary=False)
        embed_mean = 0.007213036829414903
        embed_std = 0.32103247469013046
        for token in w2v.vocab:
            # print(token)
            if token not in self.word2id:
                continue
            trained_embeddings[token] = w2v[token]
            if self.word_embed_dim is None:
                self.word_embed_dim = len(w2v[token])
        filtered_tokens = trained_embeddings.keys()
        # rebuild the token x id map
        self.word2id = {}
        self.id2word = {}
        for token in self.initial_tokens:
            self.add_word(token, cnt=0)
        for token in filtered_tokens:
            self.add_word(token, cnt=0)
        # load embeddings
        self.word_embeddings = np.random.normal(embed_mean, embed_std, (self.word_size(), self.word_embed_dim))
        for token in self.word2id.keys():
            if token in trained_embeddings:
                self.word_embeddings[self.get_word_id(token)] = trained_embeddings[token]

    def convert_word_to_ids(self, tokens):
        vec = [self.get_word_id(label) for label in tokens]
        return vec

    def recover_from_word_ids(self, ids, stop_id=None):
        tokens = []
        for i in ids:
            tokens += [self.get_word_token(i)]
            if stop_id is not None and i == stop_id:
                break
        return tokens
