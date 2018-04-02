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

"""Download SQuAD train and dev sets.
Tokenize, construct, and write TFRecord files.
"""

import argparse
import json
import nltk
import numpy as np
import os
import random
import tensorflow as tf
from six.moves.urllib.request import urlretrieve
from tqdm import tqdm

PAD_ID = 0
UNK_ID = 1
NUM_RESERVED_IDS = 2
SQUAD_BASE_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--word_emb_file", default="word_emb_file.txt")
    parser.add_argument("--word_emb_size", default=300)
    parser.add_argument("--char_emb_file", default="char_emb_file.txt")
    parser.add_argument("--char_emb_size", default=200)
    parser.add_argument("--dev_rec_file", default="dev.tfrecord")
    parser.add_argument("--dev_ans_file", default="dev_ans.json")
    parser.add_argument("--dev_info_file", default="dev_info.json")
    parser.add_argument("--train_rec_file", default="train.tfrecord")
    parser.add_argument("--train_ans_file", default="train_ans.json")
    parser.add_argument("--train_info_file", default="train_info.json")
    parser.add_argument("--test_rec_file", default="test.tfrecord")
    parser.add_argument("--test_ans_file", default="test_ans.json")
    parser.add_argument("--test_info_file", default="test_info.json")
    parser.add_argument("--glove_file", default="glove.840B.300d.txt")
    parser.add_argument("--max_c_len", default=400)
    parser.add_argument("--max_q_len", default=50)
    parser.add_argument("--max_c_len_test", default=600)
    parser.add_argument("--max_q_len_test", default=80)
    parser.add_argument("--max_w_len", default=16)
    parser.add_argument("--is_training", dest='is_training', action='store_true')
    parser.set_defaults(is_training=True)

    return parser.parse_args()


def write_to_file(out_file, line):
    out_file.write(line.encode('utf8') + '\n')


def data_from_json(filename):
    """Load JSON data from filename."""
    with open(filename) as data_file:
        data = json.load(data_file)
    return data


def tokenize(sequence):
    tokens = [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sequence)]
    return tokens


def total_exs(dataset):
    """
    Returns the total number of (context, question, answer) triples,
    given the data read from the SQuAD json file.
    """
    total = 0
    for article in dataset['data']:
        for para in article['paragraphs']:
            total += len(para['qas'])
    return total


def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optional
            Number of blocks just transferred [default: 1].
        bsize: int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


def maybe_download(url, filename, prefix, num_bytes=None):
    """Takes an URL, a filename, and the expected bytes, download
    the contents and returns the filename.
    num_bytes=None disables the file size check."""
    local_filename = None
    output_path = os.path.join(prefix, filename)
    if not os.path.exists(output_path):
        try:
            print("Downloading file {} to {}...".format(url + filename, output_path))
            with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                local_filename, _ = urlretrieve(url + filename, output_path, reporthook=reporthook(t))
        except AttributeError as e:
            print("An error occurred when downloading the file! Please get the dataset using a browser.")
            raise e
    # We have a downloaded file
    # Check the stats and make sure they are ok
    file_stats = os.stat(os.path.join(prefix, filename))
    if num_bytes is None or file_stats.st_size == num_bytes:
        print("File {} successfully downloaded to {}.".format(filename, output_path))
    else:
        raise Exception("Unexpected dataset size. Please get the dataset using a browser.")

    return local_filename


def get_ids(tokens, max_seq_len, max_word_len, word2id, char2id):
    """Turns an already-tokenized sentence string into word indices
    and char indices.
    """
    ids = np.zeros((max_seq_len,), dtype=np.int32)
    char_ids = np.zeros((max_seq_len, max_word_len), dtype=np.int32)
    for i, token in enumerate(tokens):
        if i >= max_seq_len:
            break
        # Check all capitalization variants of the token.
        found_id = False
        for variant in (token, token.lower(), token.capitalize(), token.upper()):
            if variant in word2id:
                ids[i] = word2id[variant]
                found_id = True
                break
        if not found_id:
            ids[i] = UNK_ID

        # Add char IDs.
        for j, c in enumerate(token):
            if j >= max_word_len:
                break
            if c in char2id:
                char_ids[i, j] = char2id[c]
            else:
                char_ids[i, j] = UNK_ID

    return ids, char_ids


def get_one_hots(ans_starts, ans_ends, max_len):
    """Get one-hot answer vectors from a list of answer starts and answer ends.
    """
    ans_start = np.zeros((max_len,), dtype=np.float32)
    ans_start[ans_starts[-1]] = 1.  # Note: We just take the last answer
    ans_end = np.zeros((max_len,), dtype=np.float32)
    ans_end[ans_ends[-1]] = 1.

    return ans_start, ans_end


def get_tf_features(c_ids, c_char_ids, q_ids, q_char_ids, ans_start, ans_end, example_id):
    """Convert our own representation of an example's features to Features class for TensorFlow dataset.
    """
    features = tf.train.Features(feature={
        "example_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example_id])),
        "c_ids": tf.train.Feature(bytes_list=tf.train.BytesList(value=[c_ids.tostring()])),
        "c_char_ids": tf.train.Feature(bytes_list=tf.train.BytesList(value=[c_char_ids.tostring()])),
        "q_ids": tf.train.Feature(bytes_list=tf.train.BytesList(value=[q_ids.tostring()])),
        "q_char_ids": tf.train.Feature(bytes_list=tf.train.BytesList(value=[q_char_ids.tostring()])),
        "ans_start": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ans_start.tostring()])),
        "ans_end": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ans_end.tostring()]))
      })
    return features


def write_tf_record(args, examples, word2id, char2id, save_path):
    tf_record_writer = tf.python_io.TFRecordWriter(save_path)
    num_written = 0
    max_c_len = args.max_c_len if args.is_training else args.max_c_len_test
    max_q_len = args.max_q_len if args.is_training else args.max_q_len_test
    max_w_len = args.max_w_len
    print("Writing {} examples to TF record file: {}...".format(len(examples), save_path))
    for example in tqdm(examples):
        if len(example['c_tokens']) > max_c_len or len(example['q_tokens']) > max_q_len:
            continue
        num_written += 1
        c_ids, c_char_ids = get_ids(example['c_tokens'], max_c_len, max_w_len, word2id, char2id)
        q_ids, q_char_ids = get_ids(example['q_tokens'], max_q_len, max_w_len, word2id, char2id)
        ans_start, ans_end = get_one_hots(example['ans_starts'], example['ans_ends'], max_c_len)
        example_id = example['example_id']

        features = get_tf_features(c_ids, c_char_ids, q_ids, q_char_ids, ans_start, ans_end, example_id)
        tf_record = tf.train.Example(features=features)
        tf_record_writer.write(tf_record.SerializeToString())
    print("Wrote {} / {} records (discarded {} examples due to max length)."
          .format(num_written, len(examples), len(examples) - num_written, max_c_len))
    tf_record_writer.close()


def get_formatted_examples(args, examples, word2id, char2id):
    """Parse examples into a format that can be passed directly to the model.
    Returns:
         - list of these formatted examples, where each example is a tuple.
         - output_types
         - output_shapes
    """
    max_c_len = args.max_c_len if args.is_training else args.max_c_len_test
    max_q_len = args.max_q_len if args.is_training else args.max_q_len_test
    max_w_len = args.max_w_len
    print("Formatting {} examples for in-memory dataset...".format(len(examples)))
    formatted_examples = []
    for example in tqdm(examples):
        if len(example['c_tokens']) > max_c_len or len(example['q_tokens']) > max_q_len:
            continue
        c_ids, c_char_ids = get_ids(example['c_tokens'], max_c_len, max_w_len, word2id, char2id)
        q_ids, q_char_ids = get_ids(example['q_tokens'], max_q_len, max_w_len, word2id, char2id)
        ans_start, ans_end = get_one_hots(example['ans_starts'], example['ans_ends'], max_c_len)
        example_id = example['example_id']
        formatted_examples.append((example_id, c_ids, c_char_ids, q_ids, q_char_ids, ans_start, ans_end))

    # Need output types and shapes for constructing TF Dataset
    output_types = (tf.int64, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32, tf.float32)
    output_shapes = ([], [max_c_len], [max_c_len, max_w_len], [max_q_len], [max_q_len, max_w_len], [max_c_len], [max_c_len])

    return formatted_examples, output_types, output_shapes


def get_token_bounds(string, tokens):
    """Get list of tuples (start_idx, end_idx) with the starting char loc
    and ending char loc of each token in context.
    """
    start_idx = 0
    bounds = []
    for token in tokens:
        start_idx = string.find(token, start_idx)
        if start_idx < 0:
            return None
        bounds.append((start_idx, start_idx + len(token)))
        start_idx += len(token)
    return bounds


def preprocess(dataset):
    """Reads the dataset, extracts context, question, answer, tokenizes them,
    and calculates answer span in terms of token indices.

    Returns:
      examples: dict. Parsed examples from the dataset.
      words_seen: set. Contains all words seen.
      chars_seen: set. Contains all chars seen.
    """

    num_errors = 0
    examples = []
    answers = {}
    words_seen = set()  # Keep track of words to trim down the word embeddings
    chars_seen = set()  # Keep track of chars to know which char embeddings we need

    for article in tqdm(dataset['data']):
        for p in article['paragraphs']:
            # Tokenize the context
            context = p['context']
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')
            c_tokens = tokenize(context)
            c_chars = [list(t) for t in c_tokens]
            c_token_bounds = get_token_bounds(context, c_tokens)
            if c_token_bounds is None:
                num_errors += 1
                continue
            # Keep track of words and chars seen in contexts
            for token in c_tokens:
                words_seen.add(token)
                for char in token:
                    chars_seen.add(char)

            for qa_pair in p['qas']:
                # Tokenize each question/answer pair associated with this context
                example_id = len(examples) + 1
                question = qa_pair['question']
                q_tokens = tokenize(question)
                q_chars = [list(t) for t in q_tokens]
                # Keep track of words and chars seen in questions
                for token in q_tokens:
                    words_seen.add(token)
                    for char in token:
                        chars_seen.add(char)
                # Get word indices for start and end of each answer, as well as the answer text itself
                ans_starts = []
                ans_ends = []
                ans_texts = []
                for a in qa_pair['answers']:
                    ans_texts.append(a['text'])
                    a_start_char = a['answer_start']
                    a_end_char = a_start_char + len(a['text'])
                    a_word_idxs = []
                    for word_idx, bounds in enumerate(c_token_bounds):
                        c_token_start, c_token_end = bounds
                        if c_token_start < a_end_char and c_token_end > a_start_char:
                            a_word_idxs.append(word_idx)
                    ans_starts.append(a_word_idxs[0])
                    ans_ends.append(a_word_idxs[-1])

                examples.append({
                    "example_id": example_id,
                    "c_tokens": c_tokens,
                    "c_chars": c_chars,
                    "q_tokens": q_tokens,
                    "q_chars": q_chars,
                    "ans_starts": ans_starts,
                    "ans_ends": ans_ends
                })
                answers[str(example_id)] = {
                    "id": qa_pair['id'],
                    "c": context,
                    "bounds": c_token_bounds,
                    "a": ans_texts
                }

    random.shuffle(examples)
    print("Processed {} examples. Encountered {} errors.".format(len(examples), num_errors))
    info_dict = {"num_examples": len(examples)}

    return examples, answers, info_dict, words_seen, chars_seen


def preprocess_word_embs(glove_path, glove_dim, words_seen, output_path):
    """Reads from a GloVe-style .txt file and constructs an embedding matrix and
    mappings from words to word ids. The resulting embedding matrix only includes
    words seen in the example text, saving on memory so we can use the 840B corpus.

    This function produces a word embedding file, and writes it to output_path.

    Input:
      glove_path: path to glove.840B.{glove_dim}d.txt
      glove_dim: integer; needs to match the dimension in glove_path
      words_seen: set. Words seen in the example contexts and questions.
    """

    print("Loading GloVe vectors from file: %s" % glove_path)
    vocab_size = 2196017  # Estimated number of tokens with GloVe Common Crawl vectors
    emb_dict = {}
    glove_dict = {}
    # First pass: Go through glove vecs and add exact word matches.
    print("First pass: Adding exact matches...")
    with open(glove_path, 'r') as fh:
        for line in tqdm(fh, total=vocab_size):
            line = line.lstrip().rstrip().split()
            word = "".join(line[0:-glove_dim])  # Word may have multiple components
            vector = list(map(float, line[-glove_dim:]))
            if word in words_seen:
                emb_dict[word] = vector
            glove_dict[word] = vector

    # Second pass: Go through glove vecs and add capitalization variants that we don't already have.
    print("Second pass: Adding capitalization variants...")
    for word, vector in tqdm(glove_dict.items(), total=len(glove_dict)):
        for variant in (word, word.lower(), word.capitalize(), word.upper()):
            if variant in words_seen and variant not in emb_dict:
                emb_dict[variant] = vector

    print("Found embeddings for {} out of {} words.".format(len(emb_dict), len(words_seen)))

    # Assign IDs to all words seen in the examples.
    pad_word = "__PAD__"
    unk_word = "__UNK__"
    word2id = {word: i for i, word in enumerate(emb_dict.keys(), NUM_RESERVED_IDS)}
    word2id[pad_word] = PAD_ID
    word2id[unk_word] = UNK_ID
    emb_dict[pad_word] = [0.0 for _ in range(glove_dim)]
    emb_dict[unk_word] = [0.0 for _ in range(glove_dim)]

    # Construct the embedding matrix and write to output file
    print("Creating word embedding file at {}...".format(output_path))
    id2word = {i: word for word, i in word2id.items()}
    with open(output_path, 'w') as fh:
        for i in range(len(id2word)):
            word = id2word[i]
            tokens = [word] + ["{:.5f}".format(x_i) for x_i in emb_dict[word]]
            fh.write(" ".join(tokens) + "\n")

    return word2id


def preprocess_char_embs(char_emb_size, chars_seen, output_path):
    """Constructs random character embeddings for all characters in chars_seen.
    Write a char embedding file to output_path.
    """

    print("Creating character embedding file at {}...".format(output_path))
    emb_dict = {}
    for char in chars_seen:
        emb_dict[char] = [np.random.normal(scale=0.1) for _ in range(char_emb_size)]

    # Assign IDs to all words seen in the examples.
    pad_char = "__PAD__"
    unk_char = "__UNK__"
    char2id = {word: i for i, word in enumerate(emb_dict.keys(), 2)}
    char2id[pad_char] = 0
    char2id[unk_char] = 1
    emb_dict[pad_char] = [0.0 for _ in range(char_emb_size)]
    emb_dict[unk_char] = [0.0 for _ in range(char_emb_size)]

    # Construct the embedding matrix and write to output file
    id2char = {i: word for word, i in char2id.items()}
    with open(output_path, 'w') as fh:
        for i in range(len(id2char)):
            word = id2char[i]
            tokens = [word] + ["{:.5f}".format(x_i) for x_i in emb_dict[word]]
            fh.write(" ".join(tokens) + "\n")

    return char2id


def save_to_disk(obj, path, name="info"):
    print("Writing {} to file: {}".format(name, path))
    with open(path, 'w') as fh:
        json.dump(obj, fh)


def main(args):
    print("Downloading and processing SQuAD datasets: {}".format(args.data_dir))
    os.makedirs(args.data_dir, exist_ok=True)

    train_filename = "train-v1.1.json"
    dev_filename = "dev-v1.1.json"

    # Training set
    maybe_download(SQUAD_BASE_URL, train_filename, args.data_dir, num_bytes=30288272)
    train_data = data_from_json(os.path.join(args.data_dir, train_filename))
    print("Training set has {} examples total".format(total_exs(train_data)))
    train_examples, train_answers, train_info, train_words, train_chars = preprocess(train_data)

    # Dev set
    maybe_download(SQUAD_BASE_URL, dev_filename, args.data_dir, num_bytes=4854279)
    dev_data = data_from_json(os.path.join(args.data_dir, dev_filename))
    print("Dev set has {} examples total".format(total_exs(dev_data)))
    dev_examples, dev_answers, dev_info, dev_words, dev_chars = preprocess(dev_data)

    # Preprocess word embeddings and write to file
    output_path = os.path.join(args.data_dir, args.word_emb_file)
    glove_path = os.path.join(args.data_dir, args.glove_file)
    word2id = preprocess_word_embs(glove_path, args.word_emb_size, train_words | dev_words, output_path)

    # Preprocess char embeddings and write to file
    output_path = os.path.join(args.data_dir, args.char_emb_file)
    char2id = preprocess_char_embs(args.char_emb_size, train_chars | dev_chars, output_path)

    # Construct dataset features and write to files
    train_rec_path = os.path.join(args.data_dir, args.train_rec_file)
    write_tf_record(args, train_examples, word2id, char2id, train_rec_path)
    train_ans_path = os.path.join(args.data_dir, args.train_ans_file)
    save_to_disk(train_answers, train_ans_path, name="train answers")
    train_info_path = os.path.join(args.data_dir, args.train_info_file)
    save_to_disk(train_info, train_info_path, name="train info")

    dev_rec_path = os.path.join(args.data_dir, args.dev_rec_file)
    write_tf_record(args, dev_examples, word2id, char2id, dev_rec_path)
    dev_ans_path = os.path.join(args.data_dir, args.dev_ans_file)
    save_to_disk(dev_answers, dev_ans_path, name="dev answers")
    dev_info_path = os.path.join(args.data_dir, args.dev_info_file)
    save_to_disk(dev_info, dev_info_path, name="dev info")


if __name__ == '__main__':
    parsed_args = setup_args()
    main(parsed_args)
