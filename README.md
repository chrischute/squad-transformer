# Attention Is All SQuAD Needs


## Overview
Effective reading comprehension models often rely on recurrent neural networks (RNNs) to capture positional dependencies in text. However, RNNs are sequential by construction, which limits the amount of parallel execution that can take place during training and inference. In [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf), the authors experiment with replacing RNN modules entirely with self-attention in a neural machine translation setting. Their replacement for recurrent cells is called the *Transformer.*

In the spirit of this research, we follow [this paper](https://openreview.net/pdf?id=B14TlG-RW) and implement an RNN-free, attention-based model that performs competitively on the Stanford Question Answering Dataset (SQuAD). In particular, we use only convolution, dense layers, and self-attention, allowing our model to train ~4.5x faster than an analogous RNN-based model. This speedup comes without sacrificing effectiveness: A single model achieves 67.8/77.6 EM/F1 score, and an ensemble of 5 models achieves 69.7/78.8 EM/F1 score in official evaluation on the dev set.


## Encoder Block
The following diagram shows a Transformer (left) and an Encoder Block (right). Note where the Encoder Block draws inspiration from the Transformer: The two modules are similar in their use of positional encoding, residual connections, [layer normalization](https://arxiv.org/pdf/1607.06450.pdf), self-attention sublayers, and feed-forward sublayers.

![alt text](https://raw.githubusercontent.com/chrischute/squad-transformer/imgs/transformer_vs_encoder_block.png)

An Encoder Block differs from the Transformer in its use of stacked convolutional sublayers, which use [depthwise-separable convolution](https://arxiv.org/pdf/1610.02357.pdf) to capture local dependencies in the input sequence. Also note that the sublayer pre- and post-processing steps are rearranged. An Encoder Block uses layer norm in the pre-processing step, and performs dropout and adds the residual connection in the post-processing step.


## Model
Our model, based off [this paper](https://openreview.net/pdf?id=B14TlG-RW), follows a grouping common to SQuAD models: An embedding layer, followed by encoding, context-query attention, modeling, and output layers.

![alt text](https://raw.githubusercontent.com/chrischute/squad-transformer/imgs/model.png)

  1. **Embedding Layer.** The embedding layer maps context and query words to [GloVe](https://nlp.stanford.edu/projects/glove/) 300-dimensional word embeddings (Common Crawl 840B corpus), and maps characters to trainable 200-dimensional character embeddings. The character embeddings are passed through a convolutional layer and a max-pooling layer, as described in [this paper](https://arxiv.org/pdf/1508.06615.pdf), to produce 200-dimensional character-level word embeddings. We concatenate the word embeddings and pass them through a two-layer highway network [9], which outputs a 500- dimensional encoding of each input position.
  2. **Encoding Layer.** The encoding layer consists of a single Encoder Block, as shown in Figure 1, applied after a linear down-projection of the embeddings to size `d_model = 128`. An Encoder Block stacks `B` blocks of [`C` convolutional sublayers, a single self-attention sublayer, a feed-forward layer]. As mentioned previously, the convolutional sublayers use depthwise-separable convolution. Self-attention is implemented with multi-head, scaled dot-product attention as described in [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf), with 8 heads applied in parallel over the input. The feed-forward sublayer is a pair of time-distributed linear mappings (equivalently, convolution with kernel size 1), with ReLU activation in between. For the encoding layer’s Encoder Block, each convolution has kernel size `k = 7` and applies `d_model = 128` filters. We set `B = 1` and `C = 4`. We share weights across applications of the encoding layer to the context and question embeddings.
  3. **Context-Query Attention Layer (BiDAF).** The output of the encoding layer is fed to a bidirectional context-to-query (C2Q) and query-to-context (Q2C) layer, as described in [this paper](https://arxiv.org/pdf/1611.01603.pdf) (BiDAF). This layer computes a similarity matrix `S` of shape `(n, m)`, where `n` is the context length and `m` is the question length. The `(i, j)` entry of `S` is given by `S(i,j) = W [c_i, q_j, c_i * c_j ]`. We apply softmax to the rows and multiply by the query vectors to get the C2Q attention. Similarly, we then apply softmax to the columns, multiply by the C2Q attention matrix, followed by the matrix of context vectors to get the Q2C attention. As in the BiDAF paper, we output `[C, C2Q, C * C2Q, C * Q2C]` from this layer.
  4. **Modeling Layer.** For the modeling layer, we use an Encoder Block with `B = 7` blocks and `C = 2` convolutional sublayers per block. We use `k = 7` and `d_model = 128`, as in the encoding layer. We apply a single encoder block three times, getting outputs `M1`, `M2`, `M3`.
  5. **Output Layer.** For the output layer, we predict the answer start and end probability distributions independently with two heads, each performing a linear down-projection followed by softmax. The start predictor takes `[M1, M2]` as input, and the end predictor takes `[M1, M3]` as input.


## Training
For our optimizer, we use [Adam](https://arxiv.org/pdf/1412.6980.pdf) with a learning rate that increases linearly from 0 to `sqrt(d_model)` for 2000 warmup steps, followed by a decay proportional to the inverse square root of the step number. We use `beta_1 = 0.8`, `beta_2 = 0.999`, and `epsilon = 1e-7` for the Adam hyperparameters and train with a batch size of 32.

During training, we apply multiple forms of regularization. We use L2 regularization on all trainable kernels with `lambda = 3e-7`, and we apply dropout with `p_drop = 0.1` after every layer of the model (including both the word- and character-level embeddings). As a third form of regularization, we use [stochastic depth dropout](https://arxiv.org/pdf/1603.09382.pdf) on every sublayer in an Encoder Block. In an Encoder Block with `L` total sublayers, during training the `l`-th sublayer has probability `1 - (l/L) * p_drop` of survival (*i.e.,* earlier sublayers are more likely to survive).


## Results
We find that our model achieves over a four-fold speedup over an analogous RNN-based architecture. Moreover, our model is competitive on SQuAD. A single model achieves 67.8/77.6 EM/F1 score, and an ensemble of 5 models achieves 69.7/78.8 EM/F1 score in official evaluation on the dev set.


## Usage
To train this model for yourself, first get access to a machine with a GPU. We recommend a GPU with at least 12 GB of memory, or you may have to decrease `d_model` to fit.

  1. Run `git clone https://github.com/chrischute/squad-transformer.git`.
  2. Navigate to the `squad-transformer` directory.
  3. Run `./setup_repo.sh` to set up the virtual environment and download/pre-process the dataset and word vectors. This may take 30 minutes or longer, but only needs to be done once.
  4. Run `source activate squad` to activate the virtual environment.
  5. Run `python code/main.py --mode=train --name=squad-transformer` to begin training.

We suggest training in a separate `tmux` session, then launching another session to run TensorBoard. To run TensorBoard, navigate to `squad-transformer/logs/` and run `tensorboard --logdir=. --port=5678`. Then you will be able to see plots in your browser at http://localhost:5678/ (if you're on a cloud instance, you may need to setup [SSH port forwarding](https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server)).

On an NVIDIA Tesla K80 GPU, training takes about one hour per 10,000 iterations.


### Acknowledgements
This began as a final project for [Stanford CS224n](http://web.stanford.edu/class/cs224n/), and was supported by the Winter 2018 teaching staff. Microsoft Azure generously provided GPU instance credits used during development.
