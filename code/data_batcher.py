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

"""This file contains code to read tokenized data from file,
truncate, pad and process it into batches ready for training"""

import tensorflow as tf


def get_data_loader(flags, is_training):
    """Get a data loader to load examples from TF record format.
    """
    def load_fn(example):
        """Load an example from TF record format."""
        max_c_len = flags.max_c_len if is_training else flags.max_c_len_test
        max_q_len = flags.max_q_len if is_training else flags.max_q_len_test
        max_w_len = flags.max_w_len

        features = {"example_id": tf.FixedLenFeature([], tf.int64),
                   "c_ids": tf.FixedLenFeature([], tf.string),
                   "c_char_ids": tf.FixedLenFeature([], tf.string),
                   "q_ids": tf.FixedLenFeature([], tf.string),
                   "q_char_ids": tf.FixedLenFeature([], tf.string),
                   "ans_start": tf.FixedLenFeature([], tf.string),
                   "ans_end": tf.FixedLenFeature([], tf.string)}
        parsed_example = tf.parse_single_example(example, features=features)
        example_id = parsed_example["example_id"]
        c_ids = tf.reshape(tf.decode_raw(parsed_example["c_ids"], tf.int32), [max_c_len])
        c_char_ids = tf.reshape(tf.decode_raw(parsed_example["c_char_ids"], tf.int32), [max_c_len, max_w_len])
        q_ids = tf.reshape(tf.decode_raw(parsed_example["q_ids"], tf.int32), [max_q_len])
        q_char_ids = tf.reshape(tf.decode_raw(parsed_example["q_char_ids"], tf.int32), [max_q_len, max_w_len])
        ans_start = tf.reshape(tf.decode_raw(parsed_example["ans_start"], tf.float32), [max_c_len])
        ans_end = tf.reshape(tf.decode_raw(parsed_example["ans_end"], tf.float32), [max_c_len])

        return example_id, c_ids, c_char_ids, q_ids, q_char_ids, ans_start, ans_end

    return load_fn


def load_dataset(flags, tf_record_file, loader, shuffle=True):
    """Get dataset in batches using shuffling within windowed lengths.

    Reference: https://stackoverflow.com/questions/45292517/how-do-i-use-the-group-by-window-function-in-tensorflow
    """
    windows = [tf.constant(x) for x in range(50, 401, 50)]  # Group by length in windows [50, 100, 150, ..., 400]

    def key_fn(*args):
        """Get the index of the window to which an example belongs."""
        c_ids = args[1]
        c_mask = tf.cast(c_ids, tf.bool)
        c_len = tf.reduce_sum(tf.cast(c_mask, tf.int32))
        return tf.argmax(tf.clip_by_value(windows, 0, c_len))

    def reduce_fn(*args):
        return args[1].batch(flags.batch_size)

    if shuffle:
        dataset = tf.data.TFRecordDataset(tf_record_file).map(loader, num_parallel_calls=flags.num_loader_threads)\
            .shuffle(flags.shuffle_buffer).repeat()
        buffer_size = len(windows) * 32
        dataset = dataset.apply(tf.contrib.data.group_by_window(key_fn, reduce_fn, window_size=6 * flags.batch_size))\
            .shuffle(buffer_size)
    else:
        dataset = tf.data.TFRecordDataset(tf_record_file).map(loader, num_parallel_calls=flags.num_loader_threads)\
            .repeat()
        dataset = dataset.batch(flags.batch_size)

    return dataset

