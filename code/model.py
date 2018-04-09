# Copyright 2018 Christopher Chute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file defines the top-level model."""

import time
import logging
import os
import sys
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from saver import BestModelSaver
from tensorflow.python.ops import embedding_ops
from tqdm import tqdm

from evaluate import get_pred_text, unofficial_eval
from modules import BiDAFAttn, CharLevelEncoder, EncoderBlock, HighwayEncoder, SimpleSoftmaxLayer

logging.basicConfig(level=logging.INFO)


class SQuADTransformer(object):
    """Top-level module for the SQuADTransformer model.
    Exports build_graph to build the compute graph."""

    def __init__(self, flags, input_iterator, input_handle, word_emb_matrix, char_emb_matrix):
        """
        Initialize the QA model.

        Inputs:
          flags: the flags passed in from main.py
          input_iterator: Iterator over input batches.
          input_handle: String handle that can be assigned to as a key in a feed_dict.
          word_emb_matrix: Numpy array of word embeddings (must be trimmed to be < 2 GB).
          char_emb_matrix: Numpy array of char embeddings. Values fine-tuned in training.
        """
        print("Initializing the SQuADTransformer model...")
        self.flags = flags
        self.input_handle = input_handle

        # Add all parts of the graph
        with tf.variable_scope("SQuADTransformer"):
            self.add_placeholders()
            self.add_embedding_layer(input_iterator, word_emb_matrix, char_emb_matrix)
            self.build_graph()
            self.add_ema_ops()
            self.add_loss()

        # Define optimizer and updates (fetch updates from session to apply gradients).
        self.global_step = tf.get_variable("global_step", shape=(), dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)

        # Learning rate is linear from step 0 to self.FLAGS.lr_warmup. Then it decays as 1/sqrt(timestep).
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                     beta1=self.flags.adam_beta_1,
                                     beta2=self.flags.adam_beta_2,
                                     epsilon=self.flags.adam_epsilon)
        params = tf.trainable_variables()
        grads_and_vars = opt.compute_gradients(self.loss, params)
        gradients, variables = zip(*grads_and_vars)
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.flags.max_gradient_norm)
        self.param_norm = tf.global_norm(params)

        # Include batch norm mean and variance in gradient descent updates
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Fetch self.updates to apply gradients to all trainable parameters.
            self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        # Define savers (for checkpoints) and summaries (for TensorBoard).
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=flags.keep_last)
        best_checkpoints_dir = os.path.join(self.flags.train_dir, 'best_checkpoints')
        self.best_model_saver = BestModelSaver(best_checkpoints_dir, num_to_keep=self.flags.keep_best)
        self.summaries = tf.summary.merge_all()

    def add_placeholders(self):
        """
        Add placeholders to the graph. Placeholders are used to feed in hyperparameters.
        """
        # Add placeholder for adaptive learning rate.
        self.learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
        tf.summary.scalar('learning_rate', self.learning_rate)  # log to tensorboard

        # Add a placeholder to feed in the keep probability (for dropout). Default to 1.0 for test time.
        self.keep_prob = tf.placeholder_with_default(1.0, shape=(), name="keep_prob")

    def add_embedding_layer(self, input_iterator, word_embeddings, char_embeddings):
        """
        Add word embedding layer to the graph.

        Inputs:
          emb_matrix: shape (400002, word_emb_size).
            The GloVe vectors, plus vectors for PAD and UNK.
        """
        self.example_id, c_ids, c_char_ids, q_ids, q_char_ids, ans_start, ans_end = input_iterator.get_next()

        with tf.variable_scope("EmbeddingLayer"):
            # Compute the masks and lengths of each context and question example (assumes PAD_ID == 0).
            self.c_mask = tf.cast(c_ids, tf.bool)  # shape: (batch_size, max_c_len)
            self.q_mask = tf.cast(q_ids, tf.bool)  # shape: (batch_size, max_q_len)
            self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)  # (batch_size,)
            self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)  # (batch_size,)

            # Compute the maximum length of each example, and truncate tensors to that length.
            self.c_longest = tf.reduce_max(self.c_len)
            self.q_longest = tf.reduce_max(self.q_len)
            self.c_ids_short = c_ids[:, :self.c_longest]
            self.q_ids_short = q_ids[:, :self.q_longest]
            self.c_mask = self.c_mask[:, :self.c_longest]
            self.q_mask = self.q_mask[:, :self.q_longest]
            self.c_char_ids_short = c_char_ids[:, :self.c_longest, :]
            self.q_char_ids_short = q_char_ids[:, :self.q_longest, :]
            self.ans_start = ans_start[:, :self.c_longest]
            self.ans_end = ans_end[:, :self.c_longest]

            # Get the char-level word vectors for the context and question.
            # We first map character IDs to their character embeddings.
            # Shape: (batch_size, context_len, word_len, char_emb_size)
            char_emb_matrix = tf.get_variable("char_emb_matrix",
                                              dtype=tf.float32,
                                              initializer=tf.constant(char_embeddings, dtype=tf.float32),
                                              trainable=True)
            c_char_embs = embedding_ops.embedding_lookup(char_emb_matrix, self.c_char_ids_short)
            # Shape: (batch_size, question_len, word_len, char_emb_size)
            q_char_embs = embedding_ops.embedding_lookup(char_emb_matrix, self.q_char_ids_short)

            # Take the max over each char embedding feature to get char-level word embeddings.
            char_encoder = CharLevelEncoder(self.flags.char_emb_size, self.flags.kernel_size_emb,
                                            keep_prob=1. - 0.5 * (1. - self.keep_prob))
            # Shape: (batch_size, context_len, char_emb_size)
            c_char_embs = char_encoder.build_graph(c_char_embs, self.c_longest, self.flags.max_w_len)
            # Shape: (batch_size, question_len, char_emb_size)
            q_char_embs = char_encoder.build_graph(q_char_embs, self.q_longest, self.flags.max_w_len, reuse=True)

            # Get the GloVe word embeddings for the context and question.
            self.word_emb_matrix = tf.get_variable("word_emb_matrix",
                                                   dtype=tf.float32,
                                                   initializer=tf.constant(word_embeddings, dtype=tf.float32),
                                                   trainable=False)
            c_word_embs = embedding_ops.embedding_lookup(self.word_emb_matrix, self.c_ids_short)
            c_word_embs = tf.nn.dropout(c_word_embs, self.keep_prob)
            # Shape: (batch_size, question_len, word_emb_size)
            q_word_embs = embedding_ops.embedding_lookup(self.word_emb_matrix, self.q_ids_short)
            q_word_embs = tf.nn.dropout(q_word_embs, self.keep_prob)

            c_embs = tf.concat([c_word_embs, c_char_embs], 2)
            q_embs = tf.concat([q_word_embs, q_char_embs], 2)

        # Apply a highway network to refine the embeddings
        with tf.variable_scope("HighwayEncoder"):
            # Pass the word embeddings through a multi-layer highway network
            highway = HighwayEncoder(self.flags.num_highway_layers, self.keep_prob, l2_lambda=self.flags.l2_lambda)
            # Shape: (batch_size, context_len, word_emb_size+char_emb_size)
            self.c_embs = highway.build_graph(c_embs, reuse=None)
            # Shape: (batch_size, question_len, word_emb_size+char_emb_size)
            self.q_embs = highway.build_graph(q_embs, reuse=True)

    def build_graph(self):
        """Builds the main part of the graph for the model, starting from the input embeddings
        to the final distributions for the answer span.

        Defines:
          self.logits_start, self.logits_end: Both tensors shape (batch_size, context_len).
            These are the logits (i.e. values fed into the softmax function) for the start and end distribution.
            Important: these are -large in the pad locations. Necessary for when we feed into cross entropy function.
          self.pdist_start, self.pdist_end: Both shape (batch_size, context_len). Each row sums to 1.
            These are the result of taking (masked) softmax of logits_start and logits_end.
        """

        # Apply EncoderBlock for the stacked embedding encoder layer
        with tf.variable_scope("StackedEmbeddingEncoder"):
            emb_encoder = EncoderBlock(self.flags.num_blocks_enc, self.keep_prob, self.flags.kernel_size_enc,
                                       self.flags.d_model, self.flags.num_conv_enc, self.flags.num_heads,
                                       self.flags.d_ff, l2_lambda=self.flags.l2_lambda)
            c_enc = emb_encoder.build_graph(self.c_embs, self.c_longest, self.c_mask, reduce_input_dim=True, reuse=None)
            q_enc = emb_encoder.build_graph(self.q_embs, self.q_longest, self.q_mask, reduce_input_dim=True, reuse=True)

        # Apply bidirectional attention for the context-query attention layer
        with tf.variable_scope("ContextQueryAttention"):
            bidaf = BiDAFAttn(self.keep_prob, l2_lambda=self.flags.l2_lambda)
            # Shape: [batch_size, context_len, vec_size*8].
            attn_outputs = bidaf.build_graph(c_enc, self.c_mask, self.c_longest, q_enc, self.q_mask, self.q_longest)

        # Apply EncoderBlock x3 for the modeling layer
        with tf.variable_scope("ModelEncoder"):
            model_encoder = EncoderBlock(self.flags.num_blocks_mod, self.keep_prob, self.flags.kernel_size_mod,
                                         self.flags.d_model, self.flags.num_conv_mod, self.flags.num_heads,
                                         self.flags.d_ff, l2_lambda=self.flags.l2_lambda)
            model_1 = model_encoder.build_graph(attn_outputs, self.c_longest, self.c_mask, reduce_input_dim=True)
            model_2 = model_encoder.build_graph(model_1, self.c_longest, self.c_mask, reuse=True)
            model_3 = model_encoder.build_graph(model_2, self.c_longest, self.c_mask, reuse=True)

        # Use a simple softmax output layer to compute start and end probability distributions
        with tf.variable_scope("Output"):
            with tf.variable_scope("StartDistribution"):
                start_inputs = tf.concat([model_1, model_2], axis=-1)
                softmax_layer_start = SimpleSoftmaxLayer(l2_lambda=self.flags.l2_lambda)
                self.logits_start, self.pdist_start = softmax_layer_start.build_graph(start_inputs, self.c_mask)

            with tf.variable_scope("EndDistribution"):
                end_inputs = tf.concat([model_1, model_3], axis=-1)
                softmax_layer_end = SimpleSoftmaxLayer(l2_lambda=self.flags.l2_lambda)
                self.logits_end, self.pdist_end = softmax_layer_end.build_graph(end_inputs, self.c_mask)

    def add_ema_ops(self):
        """
        Add ops to keep an exponential moving average of all trainable variables.

        Adapted from https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
        """
        self.ema = tf.train.ExponentialMovingAverage(self.flags.ema_decay_rate)
        self.shadows = []
        self.globals = []
        for g in tf.global_variables():
            s = self.ema.average(g)
            if s:
                self.shadows.append(s)
                self.globals.append(g)
        self.shadow_assign_ops = [tf.assign(g, s) for g, s in zip(self.globals, self.shadows)]

    def add_loss(self):
        """
        Add loss computation to the graph.

        Uses:
          self.logits_start: shape (batch_size, context_len)
            IMPORTANT: Assumes that self.logits_start is masked (i.e. has -large in masked locations).
            That's because the tf.nn.softmax_cross_entropy_with_logits
            function applies softmax and then computes cross-entropy loss.
            So you need to apply masking to the logits (by subtracting large
            number in the padding location) BEFORE you pass to the
            softmax_cross_entropy_with_logits function.

          self.ans_start: shape (batch_size, context_len). One-hot with true answer start.
          self.ans_end: shape (batch_size, context_len). One-hot with true answer end.

        Defines:
          self.loss_start, self.loss_end, self.loss: all scalar tensors
        """
        with tf.variable_scope("loss"):

            # Calculate loss for prediction of start position
            loss_start = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits_start,
                                                                    labels=self.ans_start)
            self.loss_start = tf.reduce_mean(loss_start)      # scalar. avg across batch
            tf.summary.scalar('loss_start', self.loss_start)  # log to tensorboard

            # Calculate loss for prediction of end position
            loss_end = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits_end,
                                                                  labels=self.ans_end)
            self.loss_end = tf.reduce_mean(loss_end)
            tf.summary.scalar('loss_end', self.loss_end)

            # Calculate the L2 regularization loss
            regularization_loss_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            regularizer = tf_layers.l2_regularizer(scale=self.flags.l2_lambda)
            self.l2_loss = tf_layers.apply_regularization(regularizer, regularization_loss_vars)

            # Add the loss components
            self.loss = self.loss_start + self.loss_end + self.l2_loss
            tf.summary.scalar('loss', self.loss)

            # Apply EMA decay (https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage)
            ema_op = self.ema.apply(tf.trainable_variables())

            with tf.control_dependencies([ema_op]):
                self.loss = tf.identity(self.loss)

    def run_train_iter(self, session, train_handle, summary_writer, global_step):
        """
        This performs a single training iteration (forward pass, loss computation, backprop, parameter update)

        Inputs:
          session: TensorFlow session
          train_handle: Handle for training set data generator.
          summary_writer: for Tensorboard
          global_step: The current number of training iterations we've done

        Returns:
          loss: The loss (averaged across the batch) for this batch.
          global_step: The current number of training iterations we've done
          param_norm: Global norm of the parameters
          gradient_norm: Global norm of the gradients
        """
        def get_learning_rate(step_num):
            """Get the learning rate for the current training iteration.
            Based on the learning rate formula in the paper "Attention Is All You Need" by Vaswani et al., 2017
            (https://arxiv.org/pdf/1706.03762.pdf).

            Sets the learning rate using a linear scale from (step, lr) = (0, 0) -> (warmup_steps, 1/sqrt(d_model)).
            Thereafter, decreases the learning rate proportionally to 1/sqrt(global_step).
            """

            lr = self.flags.learning_rate
            if step_num < self.flags.lr_warmup:
                lr *= math.log(step_num + 1.) / math.log(self.flags.lr_warmup - 1.)

            return lr

        # Match up our input data with the placeholders
        input_feed = {self.input_handle: train_handle,
                      self.learning_rate: get_learning_rate(global_step),
                      self.keep_prob: 1.0 - self.flags.dropout}

        # output_feed contains the things we want to fetch.
        output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]

        # Run the model
        [_, summaries, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, input_feed)

        # All summaries in the graph are added to Tensorboard
        summary_writer.add_summary(summaries, global_step)

        return loss, global_step, param_norm, gradient_norm

    def get_loss(self, session, handle):
        """
        Run forward-pass only; get loss.

        Inputs:
          session: TensorFlow session
          batch: a Batch object

        Returns:
          loss: The loss (averaged across the batch) for this batch
        """

        input_feed = {self.input_handle: handle}
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.loss]

        [loss] = session.run(output_feed, input_feed)

        return loss

    def get_prob_dists(self, session, input_handle):
        """
        Run forward-pass only; get probability distributions for start and end positions.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          example_id: ID of example
          pdist_start and pdist_end: both shape (batch_size, context_len)
        """
        input_feed = {self.input_handle: input_handle}
        output_feed = [self.example_id, self.pdist_start, self.pdist_end]
        [example_id, pdist_start, pdist_end] = session.run(output_feed, input_feed)
        return example_id, pdist_start, pdist_end

    def pdist_to_pred(self, start_dist, end_dist):
        """Convert probability distributions to predictions.
        We take the pair that maximizes the joint probability subject to start <= end <= start + max_answer_len.
        """
        # Compute pairwise probabilities in parallel
        start_dist, end_dist = np.expand_dims(start_dist, axis=2), np.expand_dims(end_dist, axis=1)
        joint_pdist = np.matmul(start_dist, end_dist)  # shape (batch_size, context_len, context_len)

        # Take the maximal pair subject to p_start <= p_end <= p_start + answer_len
        batch_size, c_len, _ = joint_pdist.shape
        lower_tri = np.tril(np.ones((c_len, c_len)), -1)
        upper_tri_plus_len = np.triu(np.ones((c_len, c_len)), self.flags.max_answer_len + 1)
        illegal_idxs = np.array(lower_tri + upper_tri_plus_len, dtype=bool)
        illegal_idxs = np.tile(illegal_idxs, [batch_size, 1, 1])
        np.putmask(joint_pdist, illegal_idxs, 0.)  # Zeros out everything below the diagonal and above the legal ans len

        max_over_cols = np.max(joint_pdist, axis=2)
        max_over_rows = np.max(joint_pdist, axis=1)
        start_pos = np.argmax(max_over_cols, axis=-1)
        end_pos = np.argmax(max_over_rows, axis=-1)
        probs = np.max(max_over_rows, axis=-1)

        return start_pos, end_pos, probs  # Each is shape (batch_size)

    def get_answers(self, session, input_iterator, eval_dict, num_batches):
        """Get answers to num_batches worth of examples read from input_iterator.
        Returns:
            answers_dict: A mapping from official example UUID to tuple (predicted answer text, associated prob).
        """
        print("Generating answers for {} batches...".format(num_batches))
        input_handle = session.run(input_iterator.string_handle())
        answers_dict = {}  # Dictionary mapping from UUID to (answer text, probability) pairs.
        for _ in tqdm(range(num_batches)):
            example_id, pdist_start, pdist_end = self.get_prob_dists(session, input_handle)
            pred_start, pred_end, probs = self.pdist_to_pred(pdist_start, pdist_end)
            pred_text = get_pred_text(eval_dict, example_id.tolist(), pred_start.tolist(), pred_end.tolist(), probs.tolist(), use_official_ids=True)
            answers_dict.update(pred_text)

        return answers_dict

    def eval(self, session, input_handle, num_batches, eval_dict, dataset_name="train"):
        """Evaluate the model on a given dataset. Performs "official" evaluation, where we
        take the max evaluation metric over all available ground truths.
        """
        predictions = {}  # Maps question IDs to predicted answer text.
        losses = []
        if logging is not None:
            logging.info("Evaluating on {} batches from the {} dataset...".format(num_batches, dataset_name))
        for _ in tqdm(range(num_batches)):
            input_feed = {self.input_handle: input_handle}
            output_feed = [self.loss, self.example_id, self.pdist_start, self.pdist_end]
            loss, example_id, pdist_start, pdist_end = session.run(output_feed, input_feed)
            pred_start, pred_end, probs = self.pdist_to_pred(pdist_start, pdist_end)
            losses.append(loss)
            pred_text = get_pred_text(eval_dict, example_id.tolist(), pred_start.tolist(), pred_end.tolist(), probs.tolist())
            for k, v in pred_text.items():  # Contains (answer text, probability) pairs, but we just want answer text
                predictions[k] = v[0]

        loss = np.mean(losses)
        statistics = unofficial_eval(eval_dict, predictions)
        statistics["loss"] = loss

        return statistics

    def train(self, session, train_generator, train_answers, train_info, dev_generator, dev_answers, dev_info):
        """
        Main training loop.

        Inputs:
          session: TensorFlow session
          train_generator: Dataset iterator over training batches.
          train_answers: Dictionary with all answer information for train dataset.
          train_info: Dictionary with information about the train dataset.
          dev_generator: Dataset iterator over dev batches.
          dev_answers: Dictionary with all answer information for dev dataset.
          dev_info: Dictionary with information about the dev dataset.
        """

        # Print number of model parameters
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        # We will keep track of exponentially-smoothed loss
        exp_loss = None

        # Checkpoint management.
        # We keep one latest checkpoint, and flags.keep_best best checkpoints (ranked by dev F1)
        checkpoint_path = os.path.join(self.flags.train_dir, "qa.ckpt")

        # for TensorBoard
        summary_writer = tf.summary.FileWriter(self.flags.train_dir, session.graph)

        # Get handles to datasets
        train_handle = session.run(train_generator.string_handle())
        dev_handle = session.run(dev_generator.string_handle())

        # Calculate number of batches in each training set
        dev_num_batches = dev_info['num_examples'] // self.flags.batch_size + 1

        global_step = 0
        logging.info("Beginning training loop...")
        while True:

            # Run training iteration
            iter_tic = time.time()
            loss, global_step, param_norm, grad_norm = self.run_train_iter(session, train_handle, summary_writer, global_step)
            iter_toc = time.time()
            iter_time = iter_toc - iter_tic
            # Update exponentially-smoothed loss
            if not exp_loss:  # first iter
                exp_loss = loss
            else:
                exp_loss = 0.99 * exp_loss + 0.01 * loss

            # Sometimes print info to screen
            if global_step % self.flags.print_every == 0:
                logging.info(
                    'iter %d, loss %.5f, smoothed loss %.5f, grad norm %.5f, param norm %.5f, batch time %.3f' %
                    (global_step, loss, exp_loss, grad_norm, param_norm, iter_time))

            # Sometimes save model
            if global_step % self.flags.save_every == 0:
                logging.info("Saving to %s..." % checkpoint_path)
                self.saver.save(session, checkpoint_path, global_step=global_step)

            # Sometimes evaluate model on dev loss, train F1/EM and dev F1/EM
            if global_step % self.flags.eval_every == 0:
                # Get loss for dev set and log to tensorboard
                stats = self.eval(session, dev_handle, dev_num_batches, dev_answers, dataset_name="dev")
                logging.info("Dev Results (iter %d): loss: %f, F1: %f, EM: %f" % (global_step, stats['loss'], stats['f1'], stats['exact_match']))
                write_summary(stats['loss'], "dev/loss", summary_writer, global_step)
                write_summary(stats['f1'], "dev/F1", summary_writer, global_step)
                write_summary(stats['exact_match'], "dev/EM", summary_writer, global_step)
                dev_f1 = stats['f1']  # Save dev F1 score for keeping track of best

                # Get loss for training set and log to tensorboard
                stats = self.eval(session, train_handle, dev_num_batches, train_answers, dataset_name="train")
                logging.info("Train Results (iter %d): loss: %f, F1: %f, EM: %f" % (
                global_step, stats['loss'], stats['f1'], stats['exact_match']))
                write_summary(stats['loss'], "train/loss", summary_writer, global_step)
                write_summary(stats['f1'], "train/F1", summary_writer, global_step)
                write_summary(stats['exact_match'], "train/EM", summary_writer, global_step)

                # Save this checkpoint if the dev_f1 is among the N best scores (N = self.flags.keep_best).
                self.best_model_saver.handle(dev_f1, session, self.global_step)

            sys.stdout.flush()


def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)
