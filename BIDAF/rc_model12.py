# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements the reading comprehension models based on:
1. the BiDAF algorithm described in https://arxiv.org/abs/1611.01603
2. the Match-LSTM algorithm described in https://openreview.net/pdf?id=B1-q5Pqxl
Note that we use Pointer Network for the decoding stage of both models.
"""

import os
import time
import logging
import json
from utils import compute_bleu_rouge
from utils import normalize
from layers.match_layer import AttentionFlowMatchLayer8
from layers.pointer_net import PointerNetDecoder
from layers.ops import *
from optimizer import *
from layers.basic_rnn import rnn,bilstm_layer


class RCModel(object):
    """
    Implements the main reading comprehension model.
    """

    def __init__(self, vocab, num_train_steps,num_warm_up,args):

        # logging
        self.logger = logging.getLogger("QAPointNet")

        # basic config
        self.algo = args.algo
        self.num_train_steps = num_train_steps
        self.num_warm_up = num_warm_up
        self.clip_weight=args.clip_weight
        self.max_norm_grad=args.max_norm_grad
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.decay=args.decay
        self.use_dropout = args.dropout < 1

        # length limit
        self.max_p_num = args.max_p_num
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        self.max_a_len = args.max_a_len

        # the vocab
        self.vocab = vocab

        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self._build_graph()

        # save info
        self.saver = tf.train.Saver(max_to_keep=10)
        self.train_writer = tf.summary.FileWriter(args.summary_dir, self.sess.graph)

        # initialize the model
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        self._setup_placeholders()
        self._embed()
        self._encode()
        self._match()
        self._fuse()
        self._decode()
        self._compute_loss()
        self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        param_num = total_params(tf.trainable_variables())
        self.logger.info('There are {} parameters in the model'.format(param_num))

    def _setup_placeholders(self):
        """
        Placeholders
        """
        self.p = tf.placeholder(tf.int32, [None, None], "context")
        self.q = tf.placeholder(tf.int32, [None, None], "question")
        self.p_length = tf.placeholder(tf.int32, [None])
        self.q_length = tf.placeholder(tf.int32, [None])
        self.start_label = tf.placeholder(tf.int32, [None])
        self.end_label = tf.placeholder(tf.int32, [None])
        self.dropout = tf.placeholder(tf.float32, name="dropout")
        self.global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)

    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        """
        with tf.variable_scope('word_embedding'):
            self.pretrained_word_mat = tf.get_variable("word_emb_mat",
                                                       [self.vocab.word_size() - 2, self.vocab.word_embed_dim],
                                                       dtype=tf.float32,
                                                       initializer=tf.constant_initializer(
                                                           self.vocab.word_embeddings[2:],
                                                           dtype=tf.float32),
                                                       trainable=False)
            self.word_pad_unk_mat = tf.get_variable("word_unk_pad",
                                                    [2, self.pretrained_word_mat.get_shape()[1]],
                                                    dtype=tf.float32,
                                                    initializer=tf.constant_initializer(
                                                        self.vocab.word_embeddings[:2],
                                                        dtype=tf.float32),
                                                    trainable=True)

            self.word_mat = tf.concat([self.word_pad_unk_mat, self.pretrained_word_mat], axis=0)

            self.p_emb = tf.nn.embedding_lookup(self.word_mat, self.p)
            self.q_emb = tf.nn.embedding_lookup(self.word_mat, self.q)
            self.c_mask = tf.cast(self.p, tf.bool)
            self.q_mask = tf.cast(self.q, tf.bool)

    def _encode(self):
        """
        Employs two Bi-LSTMs to encode passage and question separately
        """
        with tf.variable_scope('encoding'):
            self.sep_p_encodes, _ =  bilstm_layer(self.p_emb, self.p_length, self.hidden_size)
            tf.get_variable_scope().reuse_variables()
            self.sep_q_encodes, _ =  bilstm_layer(self.q_emb, self.q_length, self.hidden_size)
        if self.use_dropout:
            self.sep_p_encodes = tf.nn.dropout(self.sep_p_encodes, 1-self.dropout)
            self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes, 1-self.dropout)

    def _match(self):

        match_layer = AttentionFlowMatchLayer8(self.hidden_size)
        self.match_p_encodes, _ = match_layer.match(self.sep_p_encodes, self.sep_q_encodes,self.c_mask,self.q_mask,self.p_length, self.q_length)
        self.match_p_encodes = tf.layers.dense(self.match_p_encodes, self.hidden_size * 2,activation=tf.nn.relu)

        if self.use_dropout:
            self.match_p_encodes = tf.nn.dropout(self.match_p_encodes, 1 - self.dropout)


    def _fuse(self):
        with tf.variable_scope('self-attention'):
            self.fuse_p_encodes,_= bilstm_layer(self.match_p_encodes, self.p_length, self.hidden_size)
            if self.use_dropout:
                self.fuse_p_encodes = tf.nn.dropout(self.fuse_p_encodes, 1 - self.dropout)
            JX = tf.shape(self.fuse_p_encodes)[1]
            sim_matrix = tf.matmul(self.fuse_p_encodes, self.fuse_p_encodes, transpose_b=True)
            sim_matrix /= self.hidden_size ** 0.5
            mask_c = tf.tile(tf.expand_dims(self.c_mask, axis=1), [1, JX, 1])
            context2context_attn = tf.matmul(tf.nn.softmax(softmax_mask(sim_matrix, mask_c), -1), self.fuse_p_encodes)
            self.residual_match = self.match_p_encodes + tf.nn.dropout(tf.layers.dense(tf.concat([self.fuse_p_encodes, context2context_attn], -1), self.hidden_size * 2, activation=tf.nn.relu), 1 - self.dropout)
            self.fuse_p_encodes2, _ = bilstm_layer(self.residual_match, self.p_length, self.hidden_size)
            if self.use_dropout:
                self.fuse_p_encodes2 = tf.nn.dropout(self.fuse_p_encodes2, 1 - self.dropout)
    def _decode(self):
        with tf.variable_scope('same_question_concat'):
            batch_size = tf.shape(self.start_label)[0]
            concat_passage_encodes = tf.reshape(
                self.fuse_p_encodes2,
                [batch_size, -1, 2 * self.hidden_size]
            )
            no_dup_question_encodes = tf.reshape(
                self.sep_q_encodes,
                [batch_size, -1, tf.shape(self.sep_q_encodes)[1], 2 * self.hidden_size]
            )[0:, 0, 0:, 0:]
        decoder = PointerNetDecoder(self.hidden_size)
        self.start_probs, self.end_probs = decoder.decode(concat_passage_encodes,
                                                          no_dup_question_encodes)

    def _compute_loss(self):

        def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
            """
            negative log likelyhood loss
            """
            with tf.name_scope(scope, "log_loss"):
                labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
            return losses

        self.start_loss = sparse_nll_loss(probs=self.start_probs, labels=self.start_label)
        self.end_loss = sparse_nll_loss(probs=self.end_probs, labels=self.end_label)

        self.loss = tf.reduce_mean(tf.add(self.start_loss, self.end_loss))

        self.all_params = tf.trainable_variables()
        if self.weight_decay > 0:
            self.logger.info("applying l2 loss")
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss += self.weight_decay * l2_loss
        if self.decay is not None:
            self.var_ema = tf.train.ExponentialMovingAverage(self.decay)
            ema_op = self.var_ema.apply(tf.trainable_variables())
            with tf.control_dependencies([ema_op]):
                self.loss = tf.identity(self.loss)

                self.shadow_vars = []
                self.global_vars = []
                for var in tf.global_variables():
                    v = self.var_ema.average(var)
                    if v is not None:
                        self.shadow_vars.append(v)
                        self.global_vars.append(var)
                self.assign_vars = []
                for g, v in zip(self.global_vars, self.shadow_vars):
                    self.assign_vars.append(tf.assign(g, v))

    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        self.lr = self.learning_rate
        # global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.constant(value=self.learning_rate, shape=[], dtype=tf.float32)
        learning_rate =tf.train.exponential_decay(learning_rate,self.global_step,2*self.num_warm_up,0.96,staircase=True,name="exponential_decay")

        # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
        # learning rate will be `global_step/num_warmup_steps * init_lr`.
        if self.num_warm_up:
            global_steps_int = tf.cast(self.global_step, tf.int32)
            warmup_steps_int = tf.constant(self.num_warm_up, dtype=tf.int32)

            global_steps_float = tf.cast(global_steps_int, tf.float32)
            warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

            warmup_percent_done = global_steps_float / warmup_steps_float
            warmup_learning_rate = self.learning_rate * warmup_percent_done

            is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
            learning_rate = (
                    (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
        self.current_learning_rate = learning_rate
        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.lr)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        elif self.optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.lr)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif self.optim_type == "bert":
            self.optimizer = AdamWeightDecayOptimizer(learning_rate=learning_rate, weight_decay_rate=0.01, beta_1=0.9,
                                                      beta_2=0.999, epsilon=1e-6,
                                                      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))

        self.logger.info("applying optimize %s" % self.optim_type)
        if self.clip_weight:
            # clip_weight
            tvars = tf.trainable_variables()
            grads = tf.gradients(self.loss, tvars)
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.max_norm_grad)
            grad_var_pairs = zip(grads, tvars)
            train_op = self.optimizer.apply_gradients(grad_var_pairs, name='apply_grad', global_step=self.global_step)
            new_global_step = self.global_step + 1
            train_op = tf.group(train_op, [self.global_step.assign(new_global_step)])
            self.train_op = train_op
        else:
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def train(self, data, epochs, batch_size, save_dir, save_prefix,
              dropout=0.0, evaluate=True):

        def _train_epoch(train_batches, dropout, merged, train_writer, sess, evaluate):
            total_num, total_loss = 0, 0
            log_every_n_batch, n_batch_loss = 1000, 0
            for bitx, batch in enumerate(train_batches, 1):
                feed_dict = {self.p: batch['passage_token_ids'],
                             self.q: batch['question_token_ids'],
                             self.p_length: batch['passage_length'],
                             self.q_length: batch['question_length'],
                             self.start_label: batch['start_id'],
                             self.end_label: batch['end_id'],
                             self.dropout: dropout}
                # print(self.sess.run(self.global_step))
                #total_step = sess.run(self.global_step) + 1
                _, loss,total_step,summary= sess.run([self.train_op, self.loss,self.global_step,merged], feed_dict)
                total_loss += loss * len(batch['raw_data'])
                total_num += len(batch['raw_data'])
                n_batch_loss += loss
                #summary = self.sess.run(merged, feed_dict)
                train_writer.add_summary(summary, total_step)

                if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                    self.logger.info('Average loss from batch {} to {} is {}'.format(
                        bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
                    n_batch_loss = 0

                if evaluate and total_step % 2000 == 0:
                    self.logger.info('Evaluating the model after iter {}'.format(total_step))
                    if data.dev_set is not None:
                        eval_batches = data.gen_mini_batches('dev', batch_size, pad_id, shuffle=False)
                        eval_loss,bleu_rouge = self.evaluate(eval_batches)
                        self.logger.info('Dev eval loss {}'.format(eval_loss))
                        self.logger.info('Dev eval result: {}'.format(bleu_rouge))
                        self.save(save_dir, save_prefix + '_' + str(total_step))
                    else:
                        self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            return 1.0 * total_loss / total_num

        pad_id = self.vocab.get_word_id(self.vocab.pad_token)
        tf.summary.scalar('learning_rate', self.current_learning_rate)
        tf.summary.scalar('loss', self.loss)
        merged = tf.summary.merge_all()

        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            train_batches = data.gen_mini_batches('train', batch_size, pad_id, shuffle=True)
            train_loss = _train_epoch(train_batches, dropout, merged, self.train_writer, self.sess, evaluate)
            self.logger.info('Average train loss for epoch {} is {}'.format(epoch, train_loss))

    def evaluate(self, eval_batches, result_dir=None, result_prefix=None, save_full_info=False):
        pred_answers, ref_answers = [], []
        total_loss, total_num = 0, 0
        for b_itx, batch in enumerate(eval_batches):

            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.dropout: 0.0}

            start_probs, end_probs,loss= self.sess.run([self.start_probs,self.end_probs,self.loss], feed_dict)
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])
            padded_p_len = len(batch['passage_token_ids'][0])
            for sample, start_prob, end_prob in zip(batch['raw_data'], start_probs, end_probs):
                best_answer = self.find_best_answer(sample, start_prob, end_prob, padded_p_len)
                if save_full_info:
                    sample['pred_answers'] = [best_answer]
                    pred_answers.append(sample)
                else:
                    pred_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': [best_answer],
                                         'entity_answers': [[]],
                                         'yesno_answers': []})
                if 'answers' in sample:
                    ref_answers.append({'question_id': sample['question_id'],
                                        'question_type': sample['question_type'],
                                        'answers': sample['answers'],
                                        'entity_answers': [[]],
                                        'yesno_answers': []})

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.json')
            with open(result_file, 'w',encoding='utf-8') as fout:
                for pred_answer in pred_answers:
                    fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')

            self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))

        # this average loss is invalid on test set, since we don't have true start_id and end_id
        ave_loss = 1.0 * total_loss / total_num
        # compute the bleu and rouge scores if reference answers is provided
        if len(ref_answers) > 0:
            pred_dict, ref_dict = {}, {}
            for pred, ref in zip(pred_answers, ref_answers):
                question_id = ref['question_id']
                if len(ref['answers']) > 0:
                    pred_dict[question_id] = normalize(pred['answers'])
                    ref_dict[question_id] = normalize(ref['answers'])
            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
        else:
            bleu_rouge = None
        return ave_loss,bleu_rouge

    def find_best_answer(self, sample, start_prob, end_prob, padded_p_len,para_prior_scores=(0.44, 0.23, 0.15, 0.09, 0.07)):
        """
        para_prior_scores=(0.44, 0.23, 0.15, 0.09, 0.07)
        Finds the best answer for a sample given start_prob and end_prob for each position.
        This will call find_best_answer_for_passage because there are multiple passages in a sample
        """
        best_p_idx, best_span, best_score = None, None, 0
        for p_idx, passage in enumerate(sample['passages']):
            if p_idx >= self.max_p_num:
                continue
            passage_len = min(self.max_p_len, len(passage['passage_tokens']))
            answer_span, score = self.find_best_answer_for_passage(
                start_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                end_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                passage_len)
            if para_prior_scores is not None:
                # the Nth prior score = the Number of training samples whose gold answer comes
                #  from the Nth paragraph / the number of the training samples
                score *= para_prior_scores[p_idx]
            if score > best_score:
                best_score = score
                best_p_idx = p_idx
                best_span = answer_span
        if best_p_idx is None or best_span is None:
            best_answer = ''
        else:
            best_answer = ''.join(
                sample['passages'][best_p_idx]['passage_tokens'][best_span[0]: best_span[1] + 1])
        return best_answer

    def find_best_answer_for_passage(self, start_probs, end_probs, passage_len=None):
        """
        Finds the best answer with the maximum start_prob * end_prob from a single passage
        """
        if passage_len is None:
            passage_len = len(start_probs)
        else:
            passage_len = min(len(start_probs), passage_len)
        best_start, best_end, max_prob = -1, -1, 0
        for start_idx in range(passage_len):
            for ans_len in range(self.max_a_len):
                end_idx = start_idx + ans_len
                if end_idx >= passage_len:
                    continue
                prob = start_probs[start_idx] * end_probs[end_idx]
                if prob > max_prob:
                    best_start = start_idx
                    best_end = end_idx
                    max_prob = prob
        return (best_start, best_end), max_prob

    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))
