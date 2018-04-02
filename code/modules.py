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

"""Modules used by the Fast SQuAD model.
"""

import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from math import log as math_log


class BiDAFAttn(object):
    """Module for Bi-Directional Attention Flow attention.
     Based on the paper "Bi-Directional Attention Flow for Machine Comprehension" by Seo et al., 2017.
    (https://arxiv.org/pdf/1611.01603.pdf).

    Computes a similarity matrix with tri-linear function of context, question, and context * question.
    This similarity matrix is then used to compute context-to-question (C2Q) attention and
    (optionally) question-to-context (Q2C) attention.
    """

    def __init__(self, keep_prob, l2_lambda=3e-7):
        """Initialize a BiDAFAttn module.

        Inputs:
          keep_prob: float. Probability of keeping a node, passed to tf.nn.dropout.
          l2_lambda: float. L2 regularization factor.
        """
        self.keep_prob = keep_prob
        self.l2_lambda = l2_lambda

    def _similarity_matrix(self, c_vecs, num_c_vecs, q_vecs, num_q_vecs, vec_size):
        """Build similarity matrix for Bidirectional attention.
        Use tri-linear function of context, question, and context * question.

        Inputs:
          c_vecs: tensor. Context vectors. Shape (batch_size, c_len, num_c_vecs).
          num_c_vecs: tensor. Number of context vectors.
          q_vecs: tensor. Question vectors. Shape (batch_size, q_len, num_q_vecs).
          num_q_vecs: tensor. Number of question vectors.
          vec_size: int. Size of each context and question vector.
        Returns:
          sim_mat: tensor. Similarity matrix of shape (batch_size, num_c_vecs, num_q_vecs).
        """
        # Prepare each of the inputs for the linearity.
        x_c = tf.reshape(c_vecs, [-1, vec_size])
        x_q = tf.reshape(q_vecs, [-1, vec_size])
        x_cq = tf.reshape(tf.expand_dims(c_vecs, 2) * tf.expand_dims(q_vecs, 1), [-1, vec_size])

        # Perform dropout on each input.
        x_c = tf.nn.dropout(x_c, self.keep_prob)
        x_q = tf.nn.dropout(x_q, self.keep_prob)
        x_cq = tf.nn.dropout(x_cq, self.keep_prob)

        # For memory efficiency, we compute the linearity piecewise over c, q, and c*q.
        y_c = tf_layers.fully_connected(x_c, 1, activation_fn=None,
                                        weights_regularizer=tf_layers.l2_regularizer(scale=self.l2_lambda))
        y_q = tf_layers.fully_connected(x_q, 1, activation_fn=None,
                                        weights_regularizer=tf_layers.l2_regularizer(scale=self.l2_lambda))
        y_cq = tf_layers.fully_connected(x_cq, 1, activation_fn=None,
                                         weights_regularizer=tf_layers.l2_regularizer(scale=self.l2_lambda))

        # Prepare to add each component together.
        y_c = tf.reshape(y_c, [-1, num_c_vecs, 1])
        y_q = tf.reshape(y_q, [-1, 1, num_q_vecs])
        y_cq = tf.reshape(y_cq, [-1, num_c_vecs, num_q_vecs])

        # Combine the piecewise linearities to get the full product W_sim^T [c; q; c*q].
        sim_mat = y_c + y_q + y_cq

        return sim_mat

    @staticmethod
    def _beta_fn(c_vecs, c2q_attn, q2c_attn):
        """Apply the beta function to combine the context and attention matrices.

        Based on beta from the paper "Bidirectional Attention Flow for Machine Comprehension" by Seo et al., 2017.
        (https://arxiv.org/pdf/1611.01603.pdf).
        They simply concatenate [c, c2q_attn, c * c2q_attn, c * q2c_attn].

        Inputs:
          c_vecs: tensor. Context vectors.
          c2q_attn: tensor. Context-to-query attention.
          q2c_attn: tensor or None. Query-to-context attention.
        Returns:
          beta: tensor. Result of applying the chosen beta function to the inputs.
        """
        if q2c_attn is None:
            return tf.concat([c_vecs, c2q_attn, c_vecs * c2q_attn], axis=2)
        return tf.concat([c_vecs, c2q_attn, c_vecs * c2q_attn, c_vecs * q2c_attn], axis=2)

    def build_graph(self, c_vecs, c_mask, num_c_vecs, q_vecs, q_mask, num_q_vecs, use_q2c=True, scope="BiDAFAttn"):
        """Build the BiDAF attention layer component for the compute graph.

        Inputs:
          c_vecs: tensor. Context vectors. Shape (batch_size, num_c_vecs, vec_size).
          c_mask: tensor. Mask for the context vectors. Shape (batch_size, num_c_vecs).
          num_c_vecs: tensor. Number of context vectors. Shape ().
          q_vecs: tensor. Question vectors. Shape (batch_size, num_q_vecs, vec_size).
          q_mask: tensor. Mask for the question vectors. Shape (batch_size, num_q_vecs).
          num_q_vecs: tensor. Number of question vectors. Shape ().
          use_q2c: bool. If true, include both C2Q and Q2C attention. If false, only use C2Q attention.
          scope: string. Name of scope to use for TensorFlow variables.
        Returns:
          attn_outputs: Tensor. Shape (batch_size, num_c_vecs, n*vec_size). If use_q2c, then n=4. Else n=3.
        """
        with tf.variable_scope(scope):
            vec_size = c_vecs.get_shape().as_list()[2]

            # Calculate similarity matrix (batch_size, num_c_vecs, num_q_vecs)
            sim_mat = self._similarity_matrix(c_vecs, num_c_vecs, q_vecs, num_q_vecs, vec_size)

            # Calculate context-to-query attention
            q_mask_expanded = tf.expand_dims(q_mask, axis=1)
            _, sim_bar = masked_softmax(sim_mat, q_mask_expanded, 2)  # (batch_size, num_c_vecs, num_q_vecs)
            c2q_attn = tf.matmul(sim_bar, q_vecs)                     # (batch_size, num_c_vecs, vec_size)

            # Calculate query-to-context attention
            if use_q2c:
                c_mask_expanded = tf.expand_dims(c_mask, axis=2)
                _, sim_dbl_bar = masked_softmax(sim_mat, c_mask_expanded, 1)  # (batch_size, num_c_vecs, num_q_vecs)
                sim_dbl_bar = tf.transpose(sim_dbl_bar, (0, 2, 1))            # (batch_size, num_q_vecs, num_c_vecs)
                sim_sim = tf.matmul(sim_bar, sim_dbl_bar)                     # (batch_size, num_c_vecs, num_c_vecs)
                q2c_attn = tf.matmul(sim_sim, c_vecs)                         # (batch_size, num_c_vecs, vec_size)
            else:
                q2c_attn = None

            # Apply beta function to combine the context with the attention outputs
            outputs = self._beta_fn(c_vecs, c2q_attn, q2c_attn)

        return outputs


class CharLevelEncoder(object):
    """
    Module for encoding a words based on their character embeddings.
    Performs a 1D convolution over the character embeddings, then performs max-pooling
    over each output feature for each word.

    Based on the paper "Character-Aware Neural Language Models" by Kim et al., 2015
    (https://arxiv.org/pdf/1508.06615.pdf).
    """

    def __init__(self, char_emb_size, kernel_size, keep_prob):
        self.char_emb_size = char_emb_size
        self.kernel_size = kernel_size
        self.keep_prob = keep_prob
        self.is_training = self.keep_prob < (1. - 1e-5)

    def build_graph(self, char_embeddings, seq_len, word_len, scope="CharLevelEncoder", reuse=None):
        """Compute the char-level word embeddings for the given char_embeddings sequence.

        Inputs:
          char_embeddings: tensor. Shape (batch_size, seq_len, word_len, char_emb_size).
          seq_len: tensor. Max sequence length. Shape ().
          word_len: int. Max word length.
        Return:
          Tensor shape (batch_size, seq_len, char_embedding_size). The char-level word embedding
          for each word in the input tensor.
        """
        with tf.variable_scope(scope, reuse=reuse):
            char_embeddings = tf.reshape(char_embeddings, [-1, word_len, self.char_emb_size])
            char_embeddings = tf.nn.dropout(char_embeddings, self.keep_prob)
            # VALID padding outperforms SAME here. Perhaps it dilutes the importance of prefixes and suffixes in words.
            char_embeddings = std_conv(char_embeddings, self.char_emb_size, self.kernel_size, padding="VALID",
                                       activation_fn=tf.nn.relu, reuse=reuse)  # (bs * seq_len, word_len, emb_size)
            char_embeddings = tf.reduce_max(char_embeddings, axis=1)
            char_embeddings = tf.reshape(char_embeddings, [-1, seq_len, self.char_emb_size])

        return char_embeddings


class EncoderBlock(object):
    """Module for an encoder block, which uses convolution and self-attention to encode a sequence.
    Based on the paper "Fast and Accurate Reading Comprehension" by Yu et al.
    (https://openreview.net/pdf?id=B14TlG-RW).

    Each block consists of [CONV x *] + [SELF-ATTN] + [FEED-FWD], where each sublayer in brackets
    is a layer-norm residual block mapping x to sublayer(layer_norm(x) + x).
    We apply layer normalization at the pre-processing step of each layer.
    We apply dropout, a residual connection, and stochastic depth dropout at the post-processing step of each sublayer.
    """
    def __init__(self, num_blocks, keep_prob, kernel_size, d_model, num_conv_layers, num_heads, d_ff, l2_lambda=3e-7):
        self.num_blocks = num_blocks
        self.keep_prob = keep_prob
        self.kernel_size = kernel_size
        self.d_model = d_model
        self.num_conv_layers = num_conv_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.l2_lambda = l2_lambda
        self.num_sublayers = self.num_blocks * (self.num_conv_layers + 2)

    @staticmethod
    def _sublayer_pre_process(layer_inputs, reuse=None):
        """Perform sublayer pre-processing steps. We only apply layer_norm.

        A note from Google's tensor2tensor repo:
        "The current settings ("", "dan") are the published version
        of the transformer.  ("n", "da") seems better for harder-to-learn
        models, so it should probably be the default."
        """
        return tf_layers.layer_norm(layer_inputs, scope="LayerNorm", reuse=reuse)

    def _sublayer_post_process(self, layer_inputs, layer_outputs, sublayer_id, use_dropout=True):
        """Perform sublayer pre-processing steps. We apply dropout and residual connection,
        followed by stochastic depth dropout.

        A note from Google's tensor2tensor repo:
        "The current settings ("", "dan") are the published version
        of the transformer.  ("n", "da") seems better for harder-to-learn
        models, so it should probably be the default."
        """
        if use_dropout:
            layer_outputs = tf.nn.dropout(layer_outputs, self.keep_prob)
        layer_outputs += layer_inputs
        return self._stochastic_depth_dropout(layer_inputs, layer_outputs, sublayer_id)

    def _stochastic_depth_dropout(self, sublayer_inputs, sublayer_outputs, sublayer_id):
        """Perform stochastic depth dropout on a sublayer. Earlier layers are more likely to survive.

        Sublayer l survives with probability 1 - (l/L) * (1 - p_L), where L is the total
        number of sublayers and p_L = self.keep_prob (defaults to 0.9).
        If the sublayer survives, returns layer_outputs. Else returns layer_inputs.
        """
        assert 0 < sublayer_id <= self.num_sublayers, "Invalid sublayer ID: {}.".format(sublayer_id)
        # Earlier layers are more likely to be kept than later layers.
        keep_prob = 1. - float(sublayer_id) / float(self.num_sublayers) * (1. - self.keep_prob)
        do_keep = tf.random_uniform([]) < keep_prob
        return tf.cond(do_keep, lambda: sublayer_outputs, lambda: sublayer_inputs)

    def _build_positional_encoding_sublayer(self, inputs, seq_len, scope="PositionalEncoding"):
        """Add positional encoding to the inputs.

        Based on the paper "Attention is all you need" by Vaswani et al.
        (https://arxiv.org/pdf/1706.03762.pdf).
        Code adapted from Google's Tensor2Tensor repo on GitHub.
        """
        with tf.variable_scope(scope):
            vec_size = inputs.get_shape().as_list()[-1]
            positions = tf.expand_dims(tf.to_float(tf.range(seq_len)), 1)  # shape (seq_len, 1)
            d_model = vec_size // 2
            exponent_step = math_log(10000.) / (tf.to_float(d_model) - 1)
            pos_multiplier = tf.exp(tf.to_float(tf.range(d_model)) * -exponent_step)
            pos_encoded = positions * tf.expand_dims(pos_multiplier, 0)  # shape (seq_len, d_model)
            pos_encoded = tf.concat([tf.sin(pos_encoded), tf.cos(pos_encoded)], axis=1)
            pos_encoded = tf.pad(pos_encoded, [[0, 0], [0, tf.mod(vec_size, 2)]])
            pos_encoded = tf.reshape(pos_encoded, [1, seq_len, vec_size])
            outputs = inputs + pos_encoded
            outputs = tf.nn.dropout(outputs, self.keep_prob)

        return outputs

    @staticmethod
    def _ds_conv(inputs, num_filters, kernel_size, l2_lambda=3e-7, scope="DSConv", reuse=None):
        """Depthwise-separable 1D convolution.
        Based on the paper "Xception: Deep Learning with Depthwise Separable Convolutions" by Francois Chollet.
        (https://arxiv.org/pdf/1610.02357).

        Inputs:
          inputs: tensor. Rank 3, will be expanded along dimension 2 then squeezed back.
          num_filters: int. Number of filters to use in convolution.
          kernel_size: int. Size of kernel to use in convolution.
          l2_lambda: float. L2 regularization factor for filters.
        """
        with tf.variable_scope(scope, reuse=reuse):
            vec_size = inputs.get_shape().as_list()[-1]
            # Depth-wise filter. Use He initializer because a ReLU activation follows immediately.
            d_filter = tf.get_variable("d_filter",
                                       shape=(kernel_size, 1, vec_size, 1),
                                       dtype=tf.float32,
                                       regularizer=tf_layers.l2_regularizer(scale=l2_lambda),
                                       initializer=tf_layers.variance_scaling_initializer())
            # Point-wise filter. Use He initializer because we use ReLU activation.
            p_filter = tf.get_variable("p_filter",
                                       shape=(1, 1, vec_size, num_filters),
                                       dtype=tf.float32,
                                       regularizer=tf_layers.l2_regularizer(scale=l2_lambda),
                                       initializer=tf_layers.variance_scaling_initializer())
            # Expand dims so we can use the TensorFlow separable Conv2D implementation.
            # Note: Standard tf.nn.conv1D does an analogous thing, reshaping its inputs and calling tf.nn.conv2D.
            inputs = tf.expand_dims(inputs, axis=2)
            outputs = tf.nn.separable_conv2d(inputs, d_filter, p_filter, strides=(1, 1, 1, 1), padding="SAME")
            # Bias
            b = tf.get_variable("b", shape=(outputs.shape[-1],), initializer=tf.zeros_initializer())
            # Activation
            outputs = tf.nn.relu(outputs + b)
            outputs = tf.squeeze(outputs, axis=2)

        return outputs

    def _build_conv_sublayer(self, inputs, sublayer_id, scope=None, reuse=None):
        """Compute layer_norm(x + conv(x)), where conv is depthwise-separable convolution

        Inputs:
          inputs: tensor. The input sequence to this sublayer. Shape (batch_size, seq_len, num_filters).
          sublayer_id: int. ID for this sublayer, used for stochastic depth dropout. Bounds: [1, self.num_sublayers].
        Returns:
          Tensor shape (batch_size, seq_len, num_filters). Result of applying the sublayer operations.
        """
        scope = scope or "ConvSublayer{}".format(sublayer_id)
        with tf.variable_scope(scope, reuse=reuse):
            outputs = self._sublayer_pre_process(inputs, reuse=reuse)
            outputs = self._ds_conv(outputs, self.d_model, self.kernel_size, self.l2_lambda, reuse=reuse)

        return self._sublayer_post_process(inputs, outputs, sublayer_id)

    def _build_self_attn_sublayer(self, inputs, seq_len, mask, sublayer_id, scope="SelfAttnSublayer", reuse=None):
        """Compute self_attn(layer_norm(x)) + x, where self_attn is multi-head self-attention
        as described in Vaswani et al., 2017.

        Inputs:
          inputs: tensor. The input sequence to this sublayer. Shape (batch_size, seq_len, num_filters).
          sublayer_id: int. ID for this sublayer, used for stochastic depth dropout. Bounds: [1, self.num_sublayers].
        Returns:
          Tensor shape (batch_size, seq_len, num_filters). Result of applying the sublayer operations.
        """
        with tf.variable_scope(scope, reuse=reuse):
            outputs = self._sublayer_pre_process(inputs, reuse=reuse)
            attn = MultiHeadAttn(self.num_heads, self.d_model, l2_lambda=self.l2_lambda)
            outputs = attn.build_graph(outputs, outputs, outputs, seq_len, mask, reuse=reuse)

        return self._sublayer_post_process(inputs, outputs, sublayer_id)

    def _build_feed_fwd_sublayer(self, inputs, sublayer_id, scope="FeedForwardSublayer", reuse=None):
        """Compute feed_fwd(layer_norm(x)) + x, where feed_fwd is a feed-forward of two 1x1 conv layers.

        Inputs:
          inputs: tensor. The input sequence to this sublayer. Shape (batch_size, seq_len, num_filters).
          sublayer_id: int. ID for this sublayer, used for stochastic depth dropout. Bounds: [1, self.num_sublayers].
        Returns:
          Tensor shape (batch_size, seq_len, num_filters). Result of applying the sublayer operations.
        """
        with tf.variable_scope(scope, reuse=reuse):
            outputs = self._sublayer_pre_process(inputs, reuse=reuse)
            outputs = std_conv(outputs, self.d_ff, activation_fn=tf.nn.relu, scope="Conv1", reuse=reuse)
            outputs = std_conv(outputs, self.d_model, activation_fn=None, scope="Conv2", reuse=reuse)

        return self._sublayer_post_process(inputs, outputs, sublayer_id)

    def _build_input_reduction_sublayer(self, inputs, scope="ReduceInputDim", reuse=None):
        """Project inputs down to dimension d_model. Unlike other sublayers, does not have a
        residual connection, since inputs and outputs have different dimension.
        """
        with tf.variable_scope(scope, reuse=reuse):
            outputs = std_conv(inputs, self.d_model, scope="Conv", reuse=reuse)

        return outputs

    def build_graph(self, inputs, seq_len, mask, reduce_input_dim=False, scope="EncoderBlock", reuse=None):
        """
        Build the compute graph for an EncoderBlock.

        EncoderBlocks are described in Yu et al., 2018 (https://openreview.net/forum?id=B14TlG-RW).
        An EncoderBlock consists of n blocks, where each block consists of:
          * m sublayers: depthwise-separable convolution
          * 1 sublayer: self-attention using multi-head attention (cf. Vaswani et al., 2017).
          * 1 sublayer: feed forward network-in-network (two standard convolutional layers with filter size of 1).
        Each sublayer computes sublayer(layer_norm(x)) + x, and we perform stochastic depth dropout on each sublayer.

        Inputs:
          inputs: tensor. The input sequence to this EncoderBlock. Shape (batch_size, seq_len, vec_size).
          seq_len: tensor. Maximal length of each sequence. Shape ().
          mask: tensor. Mask for the sequence. Shape (batch_size,).
          reduce_input_dim: bool. If true, project the last input dimension down from vec_size to d_model.
        Returns:
            tensor. Output of this EncoderBlock. Shape (batch_size, seq_len, d_model).
        """
        with tf.variable_scope(scope, reuse=reuse):
            # Reduce input dimension (happens only once per EncoderBlock)
            if reduce_input_dim:
                outputs = self._build_input_reduction_sublayer(inputs)
            else:
                outputs = inputs

            # Keep track of sublayer ID for computing stochastic depth dropout probability
            sublayer_id = 1
            for block_id in range(self.num_blocks):
                with tf.variable_scope("Block{}".format(block_id + 1), reuse=reuse):
                    # Add positional encoding
                    outputs = self._build_positional_encoding_sublayer(outputs, seq_len)

                    # Add convolution sublayers
                    for _ in range(self.num_conv_layers):
                        outputs = self._build_conv_sublayer(outputs, sublayer_id, reuse=reuse)
                        sublayer_id += 1

                    # Add self-attention sublayer
                    outputs = self._build_self_attn_sublayer(outputs, seq_len, mask, sublayer_id, reuse=reuse)
                    sublayer_id += 1

                    # Add feed-forward sublayer
                    outputs = self._build_feed_fwd_sublayer(outputs, sublayer_id, reuse=reuse)
                    sublayer_id += 1

        return outputs


class HighwayEncoder(object):
    """
    Encode an input sequence using a highway network.

    Based on the paper "Highway Networks" by Srivastava et al.
    (https://arxiv.org/pdf/1505.00387.pdf).
    """
    def __init__(self, num_layers, keep_prob, l2_lambda=3e-7):
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        self.l2_lambda = l2_lambda

    def build_graph(self, inputs, scope="HighwayEncoder", reuse=None):
        """
        Build a highway network with the number of layers specified in the initializer.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, vec_size)
        Return: Tensor shape (batch_size, seq_len, vec_size) after passing through
          num_layers highway network layers.
        """
        with tf.variable_scope(scope, reuse=reuse):
            outputs = inputs
            for l in range(self.num_layers):
                # Each layer computes [carry * transform + (1-carry) * inputs]
                with tf.variable_scope("Layer{}".format(l+1), reuse=reuse):
                    vec_size = inputs.get_shape().as_list()[-1]

                    # Compute non-linear transform with 1x1 feed-forward Conv layer and ReLU activation
                    h = std_conv(outputs, vec_size, activation_fn=tf.nn.relu, scope="NonLinearTransform", reuse=reuse)

                    # Compute transform gate with 1x1 feed-forward Conv layer and sigmoid activation
                    t = std_conv(outputs, vec_size, activation_fn=tf.nn.sigmoid, scope="TransformGate", reuse=reuse)

                    # Combine non-linear transformation with the inputs, gated by the transform gate (we set C = 1-T).
                    outputs = h * t + outputs * (1. - t)

                    outputs = tf.nn.dropout(outputs, self.keep_prob)

        return outputs


class ScaledDotProductAttn(object):
    """Module for scaled dot-product attention.
    Based on the attention mechanism described in the paper "Attention Is All You Need" by Vaswani et al., 2017.
    (https://arxiv.org/pdf/1706.03762.pdf).

    Terminology: "Map[s] a query and a set of key-value pairs to an output, where the query, keys, values, and output
    are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value
    is computed by a compatibility function of the query with the corresponding key."
    """
    def __init__(self):
        pass

    @staticmethod
    def build_graph(queries, keys, values, mask):
        """Compute scaled dot-product attention, a weighted sum of the values, where the weight
        assigned to each value is computed by a compatibility function f the query with the corresponding key.

        Inputs:
          queries: tensor. Shape (batch_size, num_heads, max_seq_len, d_k).
          keys: tensor. Shape (batch_size, num_heads, max_seq_len, d_k).
          values: tensor. Shape (batch_size, num_heads, max_seq_len, d_v).
          mask: tensor. Shape (batch_size, max_seq_len).
        Returns:
          tensor. Weighted sum of the values, where each weight is computed by a compatibility function
          of a query vector and the key corresponding to a value. Shape matches that of values.
        """
        d_k = tf.shape(keys)[-1]
        # Compute QK^T
        qk_t = tf.matmul(queries, keys, transpose_b=True)  # Shape (bs, num_heads, max_seq_len, max_seq_len)
        # Scale by 1/sqrt(d_k) to give the dot products unit variance (hence the name "scaled dot-product")
        qk_t = qk_t / tf.sqrt(tf.cast(d_k, tf.float32))
        # Expand mask from shape (batch_size, max_seq_len) to (batch_size, 1, 1, max_seq_len).
        # We want to give out-of-range values 0 weight, and we sum over the last dimension for the softmax.
        mask = tf.expand_dims(mask, axis=1)
        mask = tf.expand_dims(mask, axis=2)
        # Compute the weights with softmax.
        _, weights = masked_softmax(qk_t, mask, -1)
        # Compute weighted sum of the values
        attn_outputs = tf.matmul(weights, values)

        return attn_outputs


class MultiHeadAttn(object):
    """Module for multi-head attention.
    Based on the attention mechanism described in the paper "Attention Is All You Need" by Vaswani et al., 2017.
    (https://arxiv.org/pdf/1706.03762.pdf).

    Calls the ScaledDotProductAttn module in parallel over a number of heads.
    """
    def __init__(self, num_heads, d_model, l2_lambda=3e-7):
        assert d_model % num_heads == 0, "MultiHeadAttn: d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.l2_lambda = l2_lambda

    def _project(self, q, k, v, scope="Linearity", reuse=None):
        """Project queries, keys, values with a linear layer.

        Note: We project the inputs for q, k, v *before* splitting to prepare inputs for each head.
        This differs from the order in "Attention Is All You Need," but is functionally equivalent.
        """
        def _project_one(x, d, inner_scope):
            return tf_layers.fully_connected(x, d, activation_fn=None, biases_initializer=None,
                                             weights_regularizer=tf_layers.l2_regularizer(scale=self.l2_lambda),
                                             scope=inner_scope, reuse=reuse)

        with tf.variable_scope(scope, reuse=reuse):
            q_projected = _project_one(q, self.d_model, "q")
            k_projected = _project_one(k, self.d_model, "k")
            v_projected = _project_one(v, self.d_model, "v")

        return q_projected, k_projected, v_projected

    def _split(self, q, k, v, seq_len):
        """Split queries, keys, values into pieces to prepare for multi-head scaled dot product.
        """
        def _split_one(x, d):
            x = tf.reshape(x, [-1, seq_len, self.num_heads, d])
            x = tf.transpose(x, [0, 2, 1, 3])  # Shape: (batch_size, num_heads, seq_len, d_x)
            return x

        q_split, k_split, v_split = _split_one(q, self.d_k), _split_one(k, self.d_k), _split_one(v, self.d_v)
        return q_split, k_split, v_split

    def _concat(self, attn_outputs, seq_len):
        """Concatenate heads of attention outputs.

        This happens at the end, after each head has performed scaled dot-product attention.
        """
        attn_outputs = tf.transpose(attn_outputs, [0, 2, 1, 3])
        attn_outputs = tf.reshape(attn_outputs, [-1, seq_len, self.num_heads * self.d_v])
        return attn_outputs

    def build_graph(self, q, k, v, seq_len, mask, scope="MultiHeadAttn", reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            # Project each of q, k, v linearly
            q, k, v = self._project(q, k, v, reuse=reuse)
            # Split each of q, k, v to prepare for scaled dot product in parallel
            q, k, v = self._split(q, k, v, seq_len)
            # Perform scaled dot-product attention on q, k, v
            sdp_attn = ScaledDotProductAttn()
            attn_outputs = sdp_attn.build_graph(q, k, v, mask)
            # Merge the outputs of each head
            attn_outputs = self._concat(attn_outputs, seq_len)
            # Linear transform to project back to model dimension
            attn_outputs = tf_layers.fully_connected(attn_outputs,
                                                     self.d_model,
                                                     biases_initializer=None,
                                                     activation_fn=None,
                                                     weights_regularizer=tf_layers.l2_regularizer(scale=self.l2_lambda),
                                                     scope="OutputTransform",
                                                     reuse=reuse)

        return attn_outputs


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self, l2_lambda=3e-7):
        self.l2_lambda = l2_lambda

    def build_graph(self, inputs, masks, scope="Softmax", reuse=None):
        """
        Applies 1D convolutional down-projection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Returns:
          logits: Tensor shape (batch_size, seq_len).
            logits is the result of the down-projection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with tf.variable_scope(scope, reuse=reuse):

            # Convolutional down-projection layer
            logits = std_conv(inputs, 1, l2_lambda=self.l2_lambda, use_bias=False, scope="Logits", reuse=reuse)
            logits = tf.squeeze(logits, axis=2)  # shape: (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30)  # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask)  # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist


def std_conv(inputs, num_filters, kernel_size=1, padding="SAME", activation_fn=None, l2_lambda=3e-7, use_bias=True,
             scope="Conv", reuse=None):
    """Standard 1D convolution, using SAME padding.

    Inputs:
      inputs: tensor. Input to the 1D conv layer. Shape (batch_size, seq_len, vec_size).
      num_filters: int. Depth of filter stack to use in 1D conv.
      kernel_size: int. Spatial extent of 1D kernel (i.e., number of timesteps the kernel covers per application).
      padding: string. Padding to use for 1D convolution. Defaults to "SAME".
      activation_fn: function. Activation function to apply to outputs before returning. If None, no activation.
      l2_lambda: float. L2 regularization factor to apply to the kernel weights.
      use_bias: bool. If true, apply a bias to the convolution outputs. Else, no bias.
    Returns:
      outputs: tensor. Outputs after convolution, bias (if any), and activation (if any) are applied.
      Shape (batch_size, out_seq_len, num_filters), where out_seq_len depends on the padding.
    """
    with tf.variable_scope(scope, reuse=reuse):
        vec_size = inputs.get_shape()[-1]
        # Use Xavier initializer if no activation, otherwise use He.
        initializer = tf_layers.xavier_initializer if activation_fn is None else tf_layers.variance_scaling_initializer
        filters = tf.get_variable("filters",
                                  shape=(kernel_size, vec_size, num_filters),
                                  dtype=tf.float32,
                                  regularizer=tf_layers.l2_regularizer(scale=l2_lambda),
                                  initializer=initializer())
        outputs = tf.nn.conv1d(inputs, filters, stride=1, padding=padding)
        if use_bias:
            b = tf.get_variable("b", shape=(num_filters,), dtype=tf.float32, initializer=tf.zeros_initializer())
            outputs += b

    return outputs if activation_fn is None else activation_fn(outputs)
