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

"""This file contains the entry point to the rest of the code"""

import io
import json
import logging
import os
import numpy as np
import tensorflow as tf

from model import SQuADTransformer
from data_batcher import get_data_loader, load_dataset
from preprocessing.squad_preprocess import data_from_json, preprocess, get_formatted_examples
from collections import defaultdict
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # relative path of main dir
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data")
LOGS_DIR = os.path.join(MAIN_DIR, "logs")

# Modes
TRAIN_MODE = "train"
EVAL_MODE = "eval"

# High-level options
tf.app.flags.DEFINE_integer("gpu", 0, "Which GPU to use, if you have multiple.")
tf.app.flags.DEFINE_string("mode", TRAIN_MODE, "Available modes: {} / {}".format(TRAIN_MODE, EVAL_MODE))
tf.app.flags.DEFINE_string("name", "", "Unique name for your experiment.")
tf.app.flags.DEFINE_integer("num_epochs", 0, "Number of epochs to train. 0 means train indefinitely.")
tf.app.flags.DEFINE_integer("num_loader_threads", 4, "Number of threads to use when parsing examples.")
tf.app.flags.DEFINE_integer("shuffle_buffer", 12000, "Size of dataset shuffle buffer.")
tf.app.flags.DEFINE_boolean("is_training", True, "True if and only if --mode={}.".format(TRAIN_MODE))

# Hyperparameters
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate after warmup period.")
tf.app.flags.DEFINE_float("lr_warmup", 1000, "Number of learning rate warmup steps.")
tf.app.flags.DEFINE_float("adam_beta_1", 0.8, "Adam optimizer beta_1 parameter.")
tf.app.flags.DEFINE_float("adam_beta_2", 0.999, "Adam optimizer beta_2 parameter.")
tf.app.flags.DEFINE_float("adam_epsilon", 1e-7, "Adam optimizer epsilon parameter.")
tf.app.flags.DEFINE_float("l2_lambda", 3e-7, "L2 regularization parameter.")
tf.app.flags.DEFINE_float("ema_decay_rate", 0.9999, "Exponential moving average decay rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.1, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use")
tf.app.flags.DEFINE_integer("num_highway_layers", 2, "Number of highway network layers to use for encoding.")
tf.app.flags.DEFINE_integer("d_model", 128, "Model dimension. I.e., number of filters for conv in an EncoderBlock.")
tf.app.flags.DEFINE_integer("d_ff", 0, "Feed-forward layer inner dimension. I.e., output size of first layer in FF.\
                                        Note: Defaults to d_model, even though Transformer uses d_ff = 4*d_model.")
tf.app.flags.DEFINE_integer("num_heads", 8, "Number of heads in multi-head attention layer of an EncoderBlock.")
tf.app.flags.DEFINE_integer("kernel_size_emb", 5, "Kernel size for convolution in CharLevelEncoder (embedding layer).")
tf.app.flags.DEFINE_integer("kernel_size_enc", 7, "Kernel size for convolution in EncoderBlock (encoding layer).")
tf.app.flags.DEFINE_integer("num_blocks_enc", 1, "Number of blocks per EncoderBlock in the encoding layer.")
tf.app.flags.DEFINE_integer("num_conv_enc", 4, "Number of convolution layers per EncoderBlock in the encoding layer.")
tf.app.flags.DEFINE_integer("kernel_size_mod", 5, "Kernel size for convolution in EncoderBlock in the modeling layer.")
tf.app.flags.DEFINE_integer("num_blocks_mod", 7, "Number of blocks per EncoderBlock in the modeling layer.")
tf.app.flags.DEFINE_integer("num_conv_mod", 2, "Number of convolution layers per EncoderBlock in the modeling layer.")
tf.app.flags.DEFINE_integer("max_c_len", 400, "The maximum context length for input to the model at train time")
tf.app.flags.DEFINE_integer("max_q_len", 50, "The maximum question length for input to the model at train time")
tf.app.flags.DEFINE_integer("max_c_len_test", 750, "The maximum context length for input to the model at test time")
tf.app.flags.DEFINE_integer("max_q_len_test", 75, "The maximum question length for input to the model at test time")
tf.app.flags.DEFINE_integer("max_answer_len", 15, "The maximum answer length predicted by your model")
tf.app.flags.DEFINE_integer("max_w_len", 16, "The maximum word length for encoding as a char-level word vector.")
tf.app.flags.DEFINE_integer("char_emb_size", 200, "Size of the pretrained character vectors.")
tf.app.flags.DEFINE_integer("word_emb_size", 300, "Size of the pretrained GloVe word vectors (50/100/200/300).")

# How often to print, save, eval (original values: 1, 500, 500, 1).
tf.app.flags.DEFINE_integer("print_every", 10, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("save_every", 1000, "How many iterations to do per save.")
tf.app.flags.DEFINE_integer("eval_every", 2000, "How many iterations to do per calculating loss/f1/em on dev set.")
tf.app.flags.DEFINE_integer("keep_last", 1, "How many last checkpoints to keep. 0 indicates keep all.")
tf.app.flags.DEFINE_integer("keep_best", 1, "How many best checkpoints to keep. 0 indicates keep all.")
tf.app.flags.DEFINE_integer("num_eval", 300, "How many batches to run evaluation over.")

# Reading and saving data
tf.app.flags.DEFINE_string("train_dir", "", "Training directory to save the model parameters and other info. Defaults to logs/{name}")
tf.app.flags.DEFINE_string("glove_path", "", "Path to glove .txt file. Defaults to data/glove.840B.{word_emb_size}d.txt")
tf.app.flags.DEFINE_string("char_emb_path", "", "Path to char embedding .txt file. Defaults to data/char_emb_file.txt")
tf.app.flags.DEFINE_string("word_emb_path", "", "Path to preprocessed word embedding .txt file. Defaults to data/word_emb_file.txt")
tf.app.flags.DEFINE_string("train_rec_path", "", "Path to train record file data/train.tfrecords")
tf.app.flags.DEFINE_string("train_ans_path", "", "Path to train answer file data/train_ans.json")
tf.app.flags.DEFINE_string("train_info_path", "", "Path to train info file data/train_info.json")
tf.app.flags.DEFINE_string("dev_rec_path", "", "Path to dev record file data/dev.tfrecords")
tf.app.flags.DEFINE_string("dev_ans_path", "", "Path to dev answer file data/dev_ans.json")
tf.app.flags.DEFINE_string("dev_info_path", "", "Path to dev info file data/dev_info.json")
tf.app.flags.DEFINE_string("test_rec_path", "", "Path to test record file data/test.tfrecords")
tf.app.flags.DEFINE_string("test_ans_path", "", "Path to test answer file data/test_ans.json")
tf.app.flags.DEFINE_string("test_info_path", "", "Path to test info file data/test_info.json")
tf.app.flags.DEFINE_string("data_dir", DEFAULT_DATA_DIR, "Where to find preprocessed SQuAD data for training. Defaults to data/")
tf.app.flags.DEFINE_string("checkpoint_dir", "", "For official_eval mode, which directory to load the checkpoint from. \
                                                 Note: Use either checkpoint_dir (single model) or ensemble_path (ensemble), not both.)")
tf.app.flags.DEFINE_string("ensemble_path", "", "Path to .txt file containing paths of checkpoint directories to use in an ensemble.\
                                                Note: Use either checkpoint_dir (single model) or ensemble_path (ensemble), not both.")
tf.app.flags.DEFINE_string("json_in_path", "", "For official_eval mode, path to JSON input file. You need to specify this for official_eval_mode.")
tf.app.flags.DEFINE_string("json_out_path", "predictions.json", "Output path for official_eval mode. Defaults to predictions.json")

FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)


def initialize_model(session, model, train_dir, expect_exists, is_training=True):
    """
    Initializes model from train_dir.

    Inputs:
      session: TensorFlow session
      model: QAModel
      train_dir: path to directory where we'll look for checkpoint
      expect_exists: If True, throw an error if no checkpoint is found.
        If False, initialize fresh model if no checkpoint is found.
    """
    print("Looking for model at %s..." % train_dir)
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        # At test time, set each variable to its exponential moving average
        if not is_training:
            shadows = session.run(model.shadows)                           # Get all saved shadow variable values
            input_feed = {v: s for (v, s) in zip(model.shadows, shadows)}  # Assign to all shadow variables
            session.run(model.shadow_assign_ops, input_feed)               # Assign shadow variables to their globals
    else:
        if expect_exists:
            raise Exception("There is no saved checkpoint at %s" % train_dir)
        else:
            print("There is no saved checkpoint at %s. Creating model with fresh parameters." % train_dir)
            session.run(tf.global_variables_initializer())
            print('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))


def get_char_embs(char_emb_path, char_emb_size, alphabet_size=1422):
    """Get pretrained character embeddings and a dictionary mapping characters to their IDs.
    Skips IDs 0 and 1, since these are reserved for PAD and UNK, respectively.

    Input:
      char_emb_path: path to glove.840B.{char_embedding_size}d-char.txt. If None, use random initialization.
      char_embedding_size: Size of character embeddings

    Returns:
      char_emb_matrix: Numpy array shape (1426, char_embedding_size) containing char embeddings.
      char2id: dict. Maps chars (string) to their IDs (int).
    """
    print("Loading char embeddings from file: {}...".format(char_emb_path))

    char_emb_matrix = []
    char2id = {}
    idx = 0
    with open(char_emb_path, 'r') as fh:
        for line in tqdm(fh, total=alphabet_size):
            line = line.lstrip().rstrip().split(" ")
            char = line[0]
            vector = list(map(float, line[1:]))
            if char_emb_size != len(vector):
                raise Exception("Expected vector of size {}, but got vector of size {}.".format(char_emb_size, len(vector)))
            char_emb_matrix.append(vector)
            char2id[char] = idx
            idx += 1

    char_emb_matrix = np.array(char_emb_matrix, dtype=np.float32)
    print("Loaded char embedding matrix with shape {}.".format(char_emb_matrix.shape))

    return char_emb_matrix, char2id


def get_word_embs(word_emb_path, word_emb_size, vocabulary_size=99002):
    """Reads from preprocessed GloVe .txt file and returns embedding matrix and
    mappings from words to word ids.

    Input:
      word_emb_path: string. Path to preprocessed glove file.
      vec_size: int. Dimensionality of a word vector.
    Returns:
      word_emb_matrix: Numpy array shape (vocab_size, vec_size) containing word embeddings.
        Only includes embeddings for words that were seen in the dev/train sets.
      word2id: dictionary mapping word (string) to word id (int)
    """

    print("Loading word embeddings from file: {}...".format(word_emb_path))

    word_emb_matrix = []
    word2id = {}
    idx = 0
    with open(word_emb_path, 'r') as fh:
        for line in tqdm(fh, total=vocabulary_size):
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            vector = list(map(float, line[1:]))
            if word_emb_size != len(vector):
                raise Exception("Expected vector of size {}, but got vector of size {}.".format(word_emb_size, len(vector)))
            word_emb_matrix.append(vector)
            word2id[word] = idx
            idx += 1

    word_emb_matrix = np.array(word_emb_matrix, dtype=np.float32)
    print("Loaded word embedding matrix with shape {}.".format(word_emb_matrix.shape))

    return word_emb_matrix, word2id


def main(unused_argv):
    # Print an error message if you've entered flags incorrectly
    if len(unused_argv) != 1:
        raise Exception("There is a problem with how you entered flags: %s" % unused_argv)

    # Define train_dir
    if not FLAGS.name and not FLAGS.train_dir and FLAGS.mode != EVAL_MODE:
        raise Exception("You need to specify either --name or --train_dir")
    FLAGS.train_dir = FLAGS.train_dir or os.path.join(LOGS_DIR, FLAGS.name)

    # If not specified, set d_ff to match d_model
    if FLAGS.d_ff == 0:
        FLAGS.d_ff = FLAGS.d_model

    # Initialize best model directory
    best_model_dir = os.path.join(FLAGS.train_dir, "best_checkpoint")

    # Define path for glove vecs
    FLAGS.glove_path = FLAGS.glove_path or os.path.join(FLAGS.data_dir, "glove.840B.{}d.txt".format(FLAGS.word_emb_size))
    FLAGS.char_emb_path = os.path.join(FLAGS.data_dir, "char_emb_file.txt")
    FLAGS.word_emb_path = os.path.join(FLAGS.data_dir, "word_emb_file.txt")

    # Get file paths to train/dev/test datafiles for tokenized queries, contexts and answers
    FLAGS.train_rec_path = os.path.join(FLAGS.data_dir, "train.tfrecord")
    FLAGS.train_ans_path = os.path.join(FLAGS.data_dir, "train_ans.json")
    FLAGS.train_info_path = os.path.join(FLAGS.data_dir, "train_info.json")
    FLAGS.dev_rec_path = os.path.join(FLAGS.data_dir, "dev.tfrecord")
    FLAGS.dev_ans_path = os.path.join(FLAGS.data_dir, "dev_ans.json")
    FLAGS.dev_info_path = os.path.join(FLAGS.data_dir, "dev_info.json")

    # Load word embedding matrix and char embedding matrix.
    word_emb_matrix, word2id = get_word_embs(FLAGS.word_emb_path, FLAGS.word_emb_size)
    char_emb_matrix, char2id = get_char_embs(FLAGS.char_emb_path, FLAGS.char_emb_size)

    # Some GPU settings
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Split by mode
    if FLAGS.mode == TRAIN_MODE:

        # Load dataset info and answer files
        print("Loading train and dev datasets...")
        train_answers = data_from_json(FLAGS.train_ans_path)
        train_info = data_from_json(FLAGS.train_info_path)
        dev_answers = data_from_json(FLAGS.dev_ans_path)
        dev_info = data_from_json(FLAGS.dev_info_path)

        # Initialize data pipeline
        loader = get_data_loader(FLAGS, is_training=True)
        train_dataset = load_dataset(FLAGS, FLAGS.train_rec_path, loader, shuffle=True)
        train_iterator = train_dataset.make_one_shot_iterator()
        dev_dataset = load_dataset(FLAGS, FLAGS.dev_rec_path, loader, shuffle=True)
        dev_iterator = dev_dataset.make_one_shot_iterator()

        # Initialize the model
        input_handle = tf.placeholder(tf.string, shape=())
        input_iterator = tf.data.Iterator.from_string_handle(input_handle, train_dataset.output_types,
                                                             train_dataset.output_shapes)
        model = SQuADTransformer(FLAGS, input_iterator, input_handle, word_emb_matrix, char_emb_matrix)

        # Setup train dir and logfile
        if not os.path.exists(FLAGS.train_dir):
            os.makedirs(FLAGS.train_dir)
        file_handler = logging.FileHandler(os.path.join(FLAGS.train_dir, "log.txt"))
        logging.getLogger().addHandler(file_handler)

        # Make best model dir if necessary
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)

        with tf.Session(config=config) as sess:

            # Load most recent model
            initialize_model(sess, model, FLAGS.train_dir, expect_exists=False)

            # Train
            model.train(sess, train_iterator, train_answers, train_info, dev_iterator, dev_answers, dev_info)

    elif FLAGS.mode == EVAL_MODE:
        if FLAGS.json_in_path == "":
            raise Exception("For {} mode, you need to specify --json_in_path".format(EVAL_MODE))
        if FLAGS.checkpoint_dir == "" and FLAGS.ensemble_path == "":
            raise Exception("For {} mode, you need to specify --checkpoint_dir or --ensemble_path".format(EVAL_MODE))
        FLAGS.is_training = False

        # Read the JSON data from file
        print("Loading test dataset from {}...".format(FLAGS.json_in_path))
        test_data = data_from_json(FLAGS.json_in_path)
        test_examples, test_answers, test_info, _, _ = preprocess(test_data)

        # Get formatted examples in memory for creating a TF Dataset
        formatted_examples, output_types, output_shapes = get_formatted_examples(FLAGS, test_examples, word2id, char2id)

        # Construct a generator function for building TF dataset
        def gen():
            infinite_idx = 0
            while True:
                yield formatted_examples[infinite_idx]
                infinite_idx = (infinite_idx + 1) % len(formatted_examples)

        # Initialize data pipeline (repeat so we can use this multiple times in an ensemble).
        test_dataset = tf.data.Dataset.from_generator(gen, output_types, output_shapes).repeat().batch(FLAGS.batch_size)
        test_iterator = test_dataset.make_one_shot_iterator()
        input_handle = tf.placeholder(tf.string, shape=())
        input_iterator = tf.data.Iterator.from_string_handle(input_handle, test_dataset.output_types, test_dataset.output_shapes)

        # Ensemble or single eval.
        is_ensemble = FLAGS.ensemble_path != ""
        if is_ensemble:  # Path to file with a list of directories for ensemble
            with open(FLAGS.ensemble_path, 'r') as fh:
                checkpoint_paths = [line.strip() for line in fh.readlines() if line]
                if len(checkpoint_paths) == 0:
                    raise Exception("Ensemble path {} did not contain any checkpoint paths.".format(FLAGS.ensemble_path))
        else:
            checkpoint_paths = [FLAGS.checkpoint_dir]

        # Make predictions using all checkpoints specified in checkpoint_paths
        model = SQuADTransformer(FLAGS, input_iterator, input_handle, word_emb_matrix, char_emb_matrix)
        all_answers = defaultdict(list)  # Maps from UUID to list of (answer text, prob) pairs.
        for i in range(len(checkpoint_paths)):
            if is_ensemble:
                print("Ensemble model {} / {}...".format(i+1, len(checkpoint_paths)))
            with tf.Session(config=config) as sess:
                # Load model from checkpoint_dir
                initialize_model(sess, model, checkpoint_paths[i], expect_exists=True, is_training=False)

                # Get a predicted answer for each example in the data
                num_batches = test_info['num_examples'] // FLAGS.batch_size + 1
                answers_dict = model.get_answers(sess, test_iterator, test_answers, num_batches)

                # Add it to the combined answers
                for k, v in answers_dict.items():
                    all_answers[k].append(v)

        # Combine the results into a final prediction
        if is_ensemble:
            print("Combining answers with max-vote...")
        answers_dict = {}
        for k, v in tqdm(all_answers.items()):
            answers_dict[k] = ensemble_max_vote(all_answers[k])

        # Write the uuid->answer mapping a to json file in root dir
        print("Writing predictions to %s..." % FLAGS.json_out_path)
        with io.open(FLAGS.json_out_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(answers_dict, ensure_ascii=False))
            print("Wrote predictions to %s" % FLAGS.json_out_path)

    else:
        raise Exception("Unsupported mode: %s" % FLAGS.mode)


def ensemble_max_vote(predictions):
    """Given a list of (answer_text, prob) pairs, return the answer text that is most voted upon.
    Fall back to the maximum probability answer.
    """
    # Count up answer frequency and find the max
    answer_counts = defaultdict(int)
    answer_probs = {}
    for pred in predictions:
        answer_counts[pred[0]] += 1
        answer_probs[pred[0]] = pred[1]
    max_count = max(answer_counts.values())

    # Get all answers with max count. Might just have one candidate, but break ties with max prob.
    candidate_answers = [ans for ans in answer_counts.keys() if answer_counts[ans] == max_count]
    assert len(candidate_answers) > 0

    best_answer = None
    for answer in candidate_answers:
        # If candidate has higher probability, take that one
        if best_answer is None or answer_probs[answer] > answer_probs[best_answer]:
            best_answer = answer

    return best_answer


if __name__ == "__main__":
    tf.app.run()
