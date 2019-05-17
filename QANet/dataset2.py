# -*- coding:utf8 -*-
'''Writen by Yanxu, FangYueran and ZhangTianyang'''
import json
import logging
import numpy as np
from collections import Counter
import os
import pickle as pkl


class DataLoader(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """
    def __init__(self, max_p_len, max_q_len,prepared_dir,
                 train_files=[], dev_files=[], test_files=[],prepare=False):
        self.logger = logging.getLogger("QAnet")
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len

        self.train_set, self.dev_set, self.test_set = [], [], []
        if train_files:
            if prepare:
                for train_file in train_files:
                    self.train_set += self._load_dataset(train_file, train=True)
                with open(os.path.join(prepared_dir, 'train_set.pkl'), 'wb') as f_train_out:
                    pkl.dump(self.train_set, f_train_out)
                f_train_out.close()
                self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))
                #del self.train_set
            else:
                with open(os.path.join(prepared_dir, 'train_set.pkl'), 'rb') as f_train_in:
                    self.train_set = pkl.load(f_train_in)
                f_train_in.close()

        if dev_files:
            if prepare:
                for dev_file in dev_files:
                    self.dev_set += self._load_dataset(dev_file)
                with open(os.path.join(prepared_dir, 'dev_set.pkl'), 'wb') as f_dev_out:
                    pkl.dump(self.dev_set, f_dev_out)
                f_dev_out.close()
                self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))
                #del self.dev_set
            else:
                with open(os.path.join(prepared_dir, 'dev_set.pkl'), 'rb') as f_dev_in:
                    self.dev_set = pkl.load(f_dev_in)
                f_dev_in.close()

        if test_files:
            if prepare:
                for test_file in test_files:
                    self.test_set += self._load_dataset(test_file)
                with open(os.path.join(prepared_dir, 'test_set.pkl'), 'wb') as f_test_out:
                    pkl.dump(self.test_set, f_test_out)
                f_test_out.close()
                self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))
                #del self.test_set
            else:
                with open(os.path.join(prepared_dir, 'test_set.pkl'), 'rb') as f_test_in:
                    self.test_set = pkl.load(f_test_in)
                f_test_in.close()

    def _load_dataset(self, data_path, train=False):
        with open(data_path,encoding='utf-8') as fin:
            data_set = []
            for lidx, line in enumerate(fin):
                sample = json.loads(line.strip())
                if train:
                    if len(sample['answer_spans']) == 0:
                        continue
                    if sample['answer_spans'][0][1] >= self.max_p_len:
                        continue
                    if 'answer_docs' in sample:
                        sample['answer_passages'] = sample['answer_docs']
                        if len(sample['documents']) <= sample['answer_passages'][0]:
                            continue

                if 'answer_docs' in sample:
                    sample['answer_passages'] = sample['answer_docs']
                    del sample['answer_docs']
                    del sample['fake_answers']
                    del sample['segmented_answers']

                sample['question_tokens'] = sample['segmented_question']
                sample['passage_tokens'] = []
                for d_idx, doc in enumerate(sample['documents']):
                    del doc['title']
                    del doc['segmented_title']
                    del doc['paragraphs']
                    if train:
                        answer_para = sample['documents'][sample['answer_passages'][0]] ['segmented_paragraphs'] [sample['documents'][sample['answer_passages'][0]]['most_related_para']]
                        most_related_para = doc['most_related_para']
                        passage_tokens = doc['segmented_paragraphs'][most_related_para]
                        if d_idx == sample['answer_passages'][0] and most_related_para + 1 < len(doc['segmented_paragraphs']):
                            passage_tokens.extend(doc['segmented_paragraphs'][most_related_para + 1])
                        if d_idx < sample['answer_passages'][0] and (len(passage_tokens) >=self.max_p_len or len(sample['passage_tokens']) >= self.max_p_len-len(passage_tokens)-len(answer_para)):
                            continue
                        else:
                            sample['passage_tokens'].extend(passage_tokens)
                            if d_idx < sample['answer_passages'][0]:
                                sample['answer_spans'][0][0] += len(passage_tokens)
                                sample['answer_spans'][0][1] += len(passage_tokens)
                    else:
                        para_infos = []
                        for para_tokens in doc['segmented_paragraphs']:
                            para_tokens = para_tokens
                            question_tokens = sample['segmented_question']

                            common_with_question = Counter(para_tokens) & Counter(question_tokens)
                            correct_preds = sum(common_with_question.values())
                            if correct_preds == 0:
                                recall_wrt_question = 0
                            else:
                                recall_wrt_question = float(correct_preds) / len(question_tokens)
                            para_infos.append((para_tokens, recall_wrt_question, len(para_tokens)))
                        para_infos.sort(key=lambda x: (-x[1], x[2]))
                        fake_passage_tokens = []
                        for para_info in para_infos[:1]:
                            fake_passage_tokens += para_info[0]
                        sample['passage_tokens'].extend(fake_passage_tokens)
                data_set.append(sample)
        return data_set

    def _one_mini_batch(self, data, indices, pad_id,train=False):
        batch_data = {'raw_data': [data[i] for i in indices],
                      'question_token_ids': [],
                      'passage_token_ids': [],
                      'start_id': [],
                      'end_id': []}
        for sidx, sample in enumerate(batch_data['raw_data']):
            batch_data['question_token_ids'].append(sample['question_token_ids'])
            batch_data['passage_token_ids'].append(sample['passage_token_ids'])
            if train:
                batch_data['start_id'].append(min(sample['answer_spans'][0][0], self.max_p_len))
                batch_data['end_id'].append(min(sample['answer_spans'][0][1], self.max_p_len))
            else:
                batch_data['start_id'].append(0)
                batch_data['end_id'].append(0)
        #print(batch_data['start_id'])
        #print(batch_data['end_id'])
        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data, pad_id)
        #print(batch_data['start_id'])
        #print(batch_data['end_id'])
        return batch_data

    def _dynamic_padding(self, batch_data, pad_id):
        pad_p_len = self.max_p_len
        pad_q_len = self.max_q_len
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_token_ids']]

        #print(np.array(batch_data['passage_char_ids']).shape, "==========")

        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids']]
        #print(np.array(batch_data['question_char_ids']).shape, "==========")

        return batch_data, pad_p_len, pad_q_len

    def word_iter(self, set_name=None):
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['question_tokens']:
                    yield token
                for token in sample['passage_tokens']:
                    yield token
    def convert_to_ids(self, vocab):
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['question_token_ids'] = vocab.convert_word_to_ids(sample['question_tokens'])
                sample['passage_token_ids'] = vocab.convert_word_to_ids(sample['passage_tokens'])

    def next_batch(self, set_name, batch_size, pad_id, shuffle=True,train=False):
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices, pad_id,train)

