# -*- coding: utf-8 -*-


import os
import time
import logging
import json
import numpy as np

import tensorflow as tf
from layers import initializer, regularizer, residual_block, highway, conv, mask_logits, trilinear, total_params
from optimizer import AdamWOptimizer, AdamWeightDecayOptimizer
from tensorflow.python.ops import array_ops

from utils.dureader_eval import compute_bleu_rouge
from utils.dureader_eval import normalize


class Model(object):
    def __init__(self, vocab, num_train_steps, num_warm_up, config):

        # logging
        self.logger = logging.getLogger("QANet")
        self.num_train_steps = num_train_steps
        self.num_warm_up = num_warm_up
        self.config = config

        # basic config
        self.optim_type = config.optim
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.use_dropout = config.dropout < 1

        self.max_p_len = config.max_p_len
        self.max_q_len = config.max_q_len
        self.max_a_len = config.max_a_len

        # the vocab
        self.vocab = vocab

        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self._build_graph()

        # save info
        self.saver = tf.train.Saver(max_to_keep=10)
        self.train_writer = tf.summary.FileWriter(self.config.summary_dir, self.sess.graph)

        # initialize the model
        self.sess.run(tf.global_variables_initializer())
        # self.sess.run(self.global_step)
        # self.global_step=max(self.sess.run(self.global_step),1)

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        self._setup_placeholders()
        self._embed()
        self._encode()
        self._fuse()
        self._decode()
        self._compute_loss()
        self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        param_num = total_params(tf.trainable_variables())
        self.logger.info('There are {} parameters in the model'.format(param_num))

    """
    :description: Placeholders
    """

    def _setup_placeholders(self):

        self.c = tf.placeholder(tf.int32, [None, self.config.max_p_len], "context")
        self.q = tf.placeholder(tf.int32, [None, self.config.max_q_len], "question")
        self.start_label = tf.placeholder(tf.int32, [None], "answer_label1")
        self.end_label = tf.placeholder(tf.int32, [None], "answer_label2")

        self.c_mask = tf.cast(self.c, tf.bool)  # index 0 is padding symbol  N x self.max_p_num, max_p_len
        self.q_mask = tf.cast(self.q, tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)
        self.dropout = tf.placeholder(tf.float32, name="dropout")
        self.global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)

    """
    :descrition: The embedding layer, question and passage share embeddings
    """

    def _embed(self):
        with tf.variable_scope('word_char_embedding'):

            if self.config.fix_pretrained_vector:
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

            else:
                self.word_mat = tf.get_variable(
                    'word_embeddings',
                    shape=[self.vocab.word_size(), self.vocab.word_embed_dim],
                    initializer=tf.constant_initializer(self.vocab.word_embeddings),
                    trainable=True
                )

        PL, QL, CL, d, dc, nh = self._params()
        with tf.variable_scope("Input_Embedding_Layer"):

            c_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.c), 1.0 - self.dropout)
            q_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.q), 1.0 - self.dropout)

            self.c_emb = highway(c_emb, size=d, scope="highway", dropout=self.dropout, reuse=None)
            self.q_emb = highway(q_emb, size=d, scope="highway", dropout=self.dropout, reuse=True)

    def _encode(self):
        PL, QL, CL, d, dc, nh = self._params()
        with tf.variable_scope("Embedding_Encoder_Layer"):
            self.c_embed_encoding = residual_block(self.c_emb,
                                                   num_blocks=1,
                                                   num_conv_layers=4,
                                                   kernel_size=5,
                                                   mask=self.c_mask,
                                                   num_filters=d,
                                                   num_heads=nh,
                                                   seq_len=self.c_len,
                                                   scope="Encoder_Residual_Block",
                                                   bias=False,
                                                   dropout=self.dropout)
            self.q_embed_encoding = residual_block(self.q_emb,
                                                   num_blocks=1,
                                                   num_conv_layers=4,
                                                   kernel_size=5,
                                                   mask=self.q_mask,
                                                   num_filters=d,
                                                   num_heads=nh,
                                                   seq_len=self.q_len,
                                                   scope="Encoder_Residual_Block",
                                                   reuse=True,  # Share the weights between passage and question
                                                   bias=False,
                                                   dropout=self.dropout)

    def _fuse(self):

        with tf.variable_scope("Context_to_Query_Attention_Layer"):
            C = tf.tile(tf.expand_dims(self.c_embed_encoding, 2), [1, 1, self.max_q_len, 1])
            Q = tf.tile(tf.expand_dims(self.q_embed_encoding, 1), [1, self.max_p_len, 1, 1])
            S = trilinear([C, Q, C * Q], input_keep_prob=1.0 - self.dropout)
            mask_q = tf.expand_dims(self.q_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask=mask_q))
            mask_c = tf.expand_dims(self.c_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask=mask_c), dim=1), (0, 2, 1))
            self.c2q = tf.matmul(S_, self.q_embed_encoding)
            self.q2c = tf.matmul(tf.matmul(S_, S_T), self.c_embed_encoding)
            self.attention_outputs = [self.c_embed_encoding, self.c2q, self.c_embed_encoding * self.c2q,
                                      self.c_embed_encoding * self.q2c]

        PL, QL, CL, d, dc, nh = self._params()
        with tf.variable_scope("Model_Encoder_Layer"):
            inputs = tf.concat(self.attention_outputs, axis=-1)
            self.enc = [conv(inputs, d, name="input_projection")]
            for i in range(3):
                if i % 2 == 0:
                    self.enc[i] = tf.nn.dropout(self.enc[i], 1.0 - self.dropout)
                self.enc.append(
                    residual_block(self.enc[i],
                                   num_blocks=7,
                                   num_conv_layers=2,
                                   kernel_size=3,
                                   mask=self.c_mask,
                                   num_filters=d,
                                   num_heads=nh,
                                   seq_len=self.c_len,
                                   scope="Model_Encoder",
                                   bias=False,
                                   reuse=True if i > 0 else None,
                                   dropout=self.dropout)
                )
    def _decode(self):

        start_logits = tf.squeeze(
            conv(tf.concat([self.enc[1], self.enc[2]], axis=-1), 1, bias=False, name="start_pointer"), -1)
        end_logits = tf.squeeze(
            conv(tf.concat([self.enc[1], self.enc[3]], axis=-1), 1, bias=False, name="end_pointer"), -1)

        self.logits = [mask_logits(start_logits, mask=self.c_mask),
                       mask_logits(end_logits, mask=self.c_mask)]

        self.logits1, self.logits2 = [l for l in self.logits]

        outer = tf.matmul(tf.expand_dims(tf.nn.softmax(self.logits1), axis=2),
                          tf.expand_dims(tf.nn.softmax(self.logits2), axis=1))

        outer = tf.matrix_band_part(outer, 0, self.max_a_len)
        self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
        self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)

    def _compute_loss(self):
        def focal_loss(logits, labels, weights=None, alpha=0.25, gamma=2):
            logits = tf.nn.sigmoid(logits)
            zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
            pos_p_sub = array_ops.where(labels > zeros, labels - logits, zeros)
            neg_p_sub = array_ops.where(labels > zeros, zeros, logits)
            cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(logits, 1e-8, 1.0)) \
                        - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - logits, 1e-8, 1.0))
            return tf.reduce_sum(cross_ent, 1)

        def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
            """
            negative log likelyhood loss
            """
            with tf.name_scope(scope, "log_loss"):
                # labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
            return losses

        start_label = tf.one_hot(self.start_label, tf.shape(self.logits1)[1], axis=1)
        end_label = tf.one_hot(self.end_label, tf.shape(self.logits2)[1], axis=1)

        if self.config.loss_type == 'cross_entropy':
            start_loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits1, labels=start_label)
            end_loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits2, labels=end_label)
            self.loss = tf.reduce_mean(start_loss + end_loss)
        elif self.config.loss_type == 'log':
            start_loss = sparse_nll_loss(probs=self.logits1, labels=start_label)
            end_loss = sparse_nll_loss(probs=self.logits2, labels=end_label)
            self.loss = tf.reduce_mean(tf.add(start_loss, end_loss))
        else:
            start_loss = focal_loss(tf.nn.softmax(self.logits1, -1), start_label)
            end_loss = focal_loss(tf.nn.softmax(self.logits2, -1), end_label)
            self.loss = tf.reduce_mean(start_loss + end_loss)
        self.logger.info("loss type %s" % self.config.loss_type)

        self.all_params = tf.trainable_variables()

        if self.config.l2_norm is not None:
            self.logger.info("applying l2 loss")
            variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            l2_loss = tf.contrib.layers.apply_regularization(regularizer, variables)
            self.loss += l2_loss

        if self.config.decay is not None:
            self.var_ema = tf.train.ExponentialMovingAverage(self.config.decay)
            ema_op = self.var_ema.apply(tf.trainable_variables())
            with tf.control_dependencies([ema_op]):
                self.loss = tf.identity(self.loss)

                self.shadow_vars = []
                self.global_vars = []
                for var in tf.global_variables():
                    v = self.var_ema.average(var)
                    if v:
                        self.shadow_vars.append(v)
                        self.global_vars.append(var)
                self.assign_vars = []
                for g, v in zip(self.global_vars, self.shadow_vars):
                    self.assign_vars.append(tf.assign(g, v))

    def _create_train_op(self):
        # self.lr = tf.minimum(self.learning_rate, self.learning_rate / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))
        self.lr = self.learning_rate
        # global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.constant(value=self.learning_rate, shape=[], dtype=tf.float32)
        learning_rate = tf.train.polynomial_decay(
            learning_rate,
            self.global_step,
            self.num_train_steps,
            end_learning_rate=0.0,
            power=1.0,
            cycle=False)

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
        elif self.optim_type == 'adamW':
            self.optimizer = AdamWOptimizer(self.config.weight_decay,
                                            learning_rate=self.lr)
        elif self.optim_type == "bert":
            self.optimizer = AdamWeightDecayOptimizer(learning_rate=learning_rate, weight_decay_rate=0.01, beta_1=0.9,
                                                      beta_2=0.999, epsilon=1e-6,
                                                      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))

        self.logger.info("applying optimize %s" % self.optim_type)
        trainable_vars = tf.trainable_variables()
        if self.config.clip_weight:
            # clip_weight
            tvars = tf.trainable_variables()
            grads = tf.gradients(self.loss, tvars)
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.config.max_norm_grad)
            grad_var_pairs = zip(grads, tvars)
            train_op = self.optimizer.apply_gradients(grad_var_pairs, name='apply_grad', global_step=self.global_step)
            new_global_step = self.global_step + 1
            train_op = tf.group(train_op, [self.global_step.assign(new_global_step)])
            self.train_op = train_op
        else:
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def _params(self):
        return (
        self.max_p_len, self.max_q_len, self.config.max_ch_len, self.config.hidden_size, self.config.char_embed_size,
        self.config.head_size)

    def train(self, data, epochs, batch_size, save_dir, save_prefix,
              dropout=0.0, evaluate=True):

        def _train_epoch(train_batches, dropout, merged, train_writer, sess, evaluate):
            total_num, total_loss = 0, 0
            log_every_n_batch, n_batch_loss = 1000, 0
            for bitx, batch in enumerate(train_batches, 1):
                feed_dict = {self.c: batch['passage_token_ids'],
                             self.q: batch['question_token_ids'],
                             self.start_label: batch['start_id'],
                             self.end_label: batch['end_id'],
                             self.dropout: dropout}
                #total_step = self.sess.run(self.global_step) + 1
                _, loss ,summary,total_step= self.sess.run([self.train_op, self.loss,merged,self.global_step], feed_dict)
                total_loss += loss * len(batch['raw_data'])
                total_num += len(batch['raw_data'])
                n_batch_loss += loss
                train_writer.add_summary(summary, total_step)

                if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                    self.logger.info('Average loss from batch {} to {} is {}'.format(
                        bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
                    n_batch_loss = 0

                if evaluate and total_step % 2000 == 0:
                    self.logger.info('Evaluating the model after iter {}'.format(total_step))
                    if data.dev_set is not None:
                        eval_batches = data.next_batch('dev', batch_size, pad_id,shuffle=False)
                        eval_loss, bleu_rouge = self.evaluate(eval_batches)
                        self.logger.info('Dev eval loss {}'.format(eval_loss))
                        self.logger.info('Dev eval result: {}'.format(bleu_rouge))
                        self.save(save_dir, save_prefix + '_' + str(total_step))
                    else:
                        self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            # print("total_num", total_num)
            return 1.0 * total_loss / total_num

        pad_id = self.vocab.get_word_id(self.vocab.pad_token)
        tf.summary.scalar('learning_rate', self.current_learning_rate)
        tf.summary.scalar('loss', self.loss)
        merged = tf.summary.merge_all()

        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            train_batches = data.next_batch('train', batch_size, pad_id,shuffle=True,train=True)
            train_loss = _train_epoch(train_batches, dropout, merged, self.train_writer, self.sess, evaluate)
            self.logger.info('Average train loss for epoch {} is {}'.format(epoch, train_loss))

    def evaluate(self, eval_batches, result_dir=None, result_prefix=None, save_full_info=False):
        pred_answers, ref_answers = [], []
        total_loss, total_num = 0, 0
        for b_itx, batch in enumerate(eval_batches):

            feed_dict = {self.c: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.dropout: 0.0}
            start_probs, end_probs, loss = self.sess.run([self.yp1, self.yp2, self.loss], feed_dict)
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

            for sample, start_prob, end_prob in zip(batch['raw_data'], start_probs.tolist(), end_probs.tolist()):

                best_answer = "".join(sample['passage_tokens'][start_prob:end_prob + 1])
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
        return ave_loss, bleu_rouge

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

