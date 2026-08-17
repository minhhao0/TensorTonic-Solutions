# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# Hào Trần's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/haoaka.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Bag-of-Words Vector | Build a NumPy bag-of-words count vector from an ordered vocabulary while ignoring out-of-vocabulary tokens. | https://www.tensortonic.com/problems/bag-of-words |
| Cosine Annealing LR Scheduler | Compute a cosine-annealed learning rate between configured maximum and minimum values across training steps. | https://www.tensortonic.com/problems/cosine-annealing-lr |
| Cosine Embedding Loss | Compute cosine embedding loss for similar and dissimilar vector pairs using labels and a configurable margin. | https://www.tensortonic.com/problems/cosine-embedding-loss |
| Implement Cosine Similarity | Compute cosine similarity between NumPy vectors with dot products, Euclidean norms, and zero-vector handling. | https://www.tensortonic.com/problems/cosine-similarity |
| Implement Dot Product | Implement the dot product of equal-length numeric vectors by summing element-wise products without library shortcuts. | https://www.tensortonic.com/problems/dot-product |
| ELU Activation | Apply the ELU activation element-wise, retaining positive inputs and exponentially transforming negative values. | https://www.tensortonic.com/problems/elu-activation |
| Implement Euclidean Distance | Compute Euclidean distance between equal-length NumPy vectors as the square root of summed squared differences. | https://www.tensortonic.com/problems/euclidean-distance |
| Implement GELU Activation (Gaussian Error Linear Unit) | Implement the Gaussian Error Linear Unit activation element-wise using the required GELU approximation. | https://www.tensortonic.com/problems/gelu |
| Implement Gradient Descent for a 1D Quadratic | Optimize a one-dimensional quadratic with iterative gradient descent and return the parameter trajectory. | https://www.tensortonic.com/problems/gradient-descent-quadratic |
| Hit Rate at K | Calculate recommendation hit rate at K by checking whether each user's relevant items appear in top-ranked results. | https://www.tensortonic.com/problems/hit-rate-at-k |
| Image Histogram | Count grayscale image pixels into intensity bins and return the histogram in ascending intensity order. | https://www.tensortonic.com/problems/image-histogram |
| Intersection over Union (IoU) | Compute intersection over union for two axis-aligned bounding boxes from overlap and combined area. | https://www.tensortonic.com/problems/iou-bounding-box |
| Implement Leaky ReLU (with α) | Apply Leaky ReLU element-wise with a configurable negative slope while retaining positive inputs. | https://www.tensortonic.com/problems/leaky-relu |
| Linear Interpolation | Interpolate values between known one-dimensional points while handling exact coordinates and valid boundaries. | https://www.tensortonic.com/problems/linear-interpolation |
| Learning Rate Scheduler (Linear Decay) | Compute a linearly decaying learning rate across training steps between configured start and end values. | https://www.tensortonic.com/problems/linear-lr-scheduler |
| Log Loss (Per-Sample) | Compute binary log loss for each prediction with clipped probabilities to prevent undefined logarithms. | https://www.tensortonic.com/problems/log-loss-per-sample |
| Log Transform | Apply a numerically safe logarithmic transform to numeric features using the required offset or base. | https://www.tensortonic.com/problems/log-transform |
| Implement Manhattan Distance | Compute Manhattan distance between equal-length vectors by summing absolute coordinate differences. | https://www.tensortonic.com/problems/manhattan-distance |
| Matrix Transpose | Implement matrix transpose in NumPy without built-in transpose helpers, preserving rectangular shapes and the original input. | https://www.tensortonic.com/problems/matrix-transpose |
| Max Pooling Forward | Apply 2D max pooling to a numeric matrix using a configurable square window and stride. | https://www.tensortonic.com/problems/maxpool-forward |
| Mean, Median, Mode | Calculate the mean, median, and deterministic mode of a numeric collection, including tied frequencies. | https://www.tensortonic.com/problems/mean-median-mode |
| Mean Squared Error (MSE) | Compute mean squared error between predictions and targets by averaging their squared element-wise differences. | https://www.tensortonic.com/problems/mean-squared-error |
| Implement Min-Max Normalization | Normalize each NumPy feature to the zero-to-one range with explicit handling for constant columns. | https://www.tensortonic.com/problems/minmax-normalization |
| Perplexity Computation | Compute language-model perplexity from token probability distributions and the observed token indices. | https://www.tensortonic.com/problems/perplexity-computation |
| Implement Positional Encoding (sin/cos) | Generate sinusoidal Transformer positional encodings across sequence positions and embedding dimensions. | https://www.tensortonic.com/problems/positional-encoding |
| Precision and Recall at K | Compute recommendation precision and recall at K by comparing ranked predictions with relevant items. | https://www.tensortonic.com/problems/precision-recall-at-k |
| Implement R² Score (Coefficient of Determination) | Compute the coefficient of determination from targets and predictions with explicit constant-target handling. | https://www.tensortonic.com/problems/r2-score |
| Implement ReLU Activation | Apply the ReLU activation element-wise by replacing negative values with zero and preserving nonnegative inputs. | https://www.tensortonic.com/problems/relu-activation |
| Remove Stopwords | Remove tokens found in a supplied stopword collection while preserving the order of remaining words. | https://www.tensortonic.com/problems/remove-stopwords |
| SELU Activation | Apply SELU activation element-wise with scaled positive values and exponential negative values. | https://www.tensortonic.com/problems/selu-activation |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| Implement Softmax Function | Implement numerically stable softmax by shifting logits before exponentiation and normalizing probabilities. | https://www.tensortonic.com/problems/softmax-function |
| Implement Swish Activation | Apply the Swish activation element-wise by multiplying each input by its sigmoid value. | https://www.tensortonic.com/problems/swish-activation |
| Implement Tanh Activation | Implement the hyperbolic tangent activation element-wise with outputs bounded between minus one and one. | https://www.tensortonic.com/problems/tanh-activation |
| Text Chunking | Split text into ordered chunks under the requested size and overlap rules without dropping content. | https://www.tensortonic.com/problems/text-chunking |
| Bottleneck Layer (DenseNet-B) | Build a DenseNet-B bottleneck layer with 1x1 channel reduction before the 3x3 feature-producing convolution. | https://www.tensortonic.com/research/densenet/densenet-bottleneck |
| Channel Growth and Compression | Compute DenseNet channel growth across dense blocks and transition compression from the initial channels and growth rate. | https://www.tensortonic.com/research/densenet/densenet-channels |
| Composite Layer (BN-ReLU-Conv) | Implement a DenseNet composite layer with batch normalization, ReLU, convolution, and feature-map concatenation. | https://www.tensortonic.com/research/densenet/densenet-composite-layer |
| Dense Block (Concatenative Connectivity) | Implement a DenseNet block that repeatedly concatenates every new layer output with all preceding feature maps. | https://www.tensortonic.com/research/densenet/densenet-dense-block |
| Full DenseNet Forward Pass | Assemble a DenseNet forward pass with dense blocks, transitions, final normalization, global pooling, and classification. | https://www.tensortonic.com/research/densenet/densenet-forward |
| Transition Layer | Implement a DenseNet transition layer with batch normalization, ReLU, 1x1 compression, and average pooling. | https://www.tensortonic.com/research/densenet/densenet-transition |
| GAN Discriminator | Implement a GAN discriminator that maps input samples through dense layers to real-versus-fake probabilities. | https://www.tensortonic.com/research/gan/gan-discriminator |
| Complete GAN System | Assemble a complete GAN system that generates samples, scores them with the discriminator, and returns both outputs. | https://www.tensortonic.com/research/gan/gan-full-network |
| GAN Generator | Implement a GAN generator that transforms latent noise through learned dense layers into generated samples. | https://www.tensortonic.com/research/gan/gan-generator |
| GAN Loss Functions | Compute numerically stable binary cross-entropy losses for the GAN generator and discriminator objectives. | https://www.tensortonic.com/research/gan/gan-loss |
| Mode Collapse Detection | Detect GAN mode collapse by measuring diversity across generated samples and flagging low-variance outputs. | https://www.tensortonic.com/research/gan/gan-mode-collapse |
| GAN Training Loop | Implement one GAN training iteration with separate discriminator and generator forward and update steps. | https://www.tensortonic.com/research/gan/gan-training-loop |
| Complete LSTM Cell | Build a complete LSTM cell with forget, input, candidate, cell-state, output, and hidden-state calculations. | https://www.tensortonic.com/research/lstm/lstm-cell |
| Cell State Update | Implement the LSTM cell-state update by combining retained memory with input-gated candidate information. | https://www.tensortonic.com/research/lstm/lstm-cell-state |
| Forget Gate | Implement an LSTM forget gate by combining the previous hidden state and current input with a sigmoid projection. | https://www.tensortonic.com/research/lstm/lstm-forget-gate |
| Complete LSTM Network | Assemble an LSTM sequence forward pass that carries hidden and cell states across every time step. | https://www.tensortonic.com/research/lstm/lstm-full-network |
| Input Gate | Implement the LSTM input gate and candidate activation that control new information written to the cell state. | https://www.tensortonic.com/research/lstm/lstm-input-gate |
| Output Gate | Implement the LSTM output gate and expose the current hidden state from the updated cell memory. | https://www.tensortonic.com/research/lstm/lstm-output-gate |
| BatchNorm in ResNet | Implement ResNet batch normalization with channel statistics, learned scale and bias, and training or inference behavior. | https://www.tensortonic.com/research/resnet/resnet-batch-norm |
| Bottleneck Block | Build a ResNet bottleneck block using 1x1 channel reduction, 3x3 convolution, and 1x1 channel expansion. | https://www.tensortonic.com/research/resnet/resnet-bottleneck |
| Convolutional Block | Implement a ResNet convolutional block with a projected shortcut that matches changed spatial and channel dimensions. | https://www.tensortonic.com/research/resnet/resnet-conv-block |
| Full ResNet Assembly | Assemble a ResNet forward pass from the stem, residual stages, global average pooling, and the classification head. | https://www.tensortonic.com/research/resnet/resnet-full-network |
| Identity Block | Implement a ResNet identity block with a three-layer bottleneck branch, batch normalization, ReLU, and an unchanged skip path. | https://www.tensortonic.com/research/resnet/resnet-identity-block |
| Skip Connection Analysis | Analyze ResNet skip connections by combining residual and identity tensors and tracking gradient flow through the addition. | https://www.tensortonic.com/research/resnet/resnet-skip-connection |
| Backpropagation Through Time | Implement one backpropagation-through-time step using the tanh derivative and hidden-to-hidden weight gradients. | https://www.tensortonic.com/research/rnn/rnn-bptt |
| RNN Cell | Implement an Elman RNN cell that combines the current input and previous hidden state before applying tanh. | https://www.tensortonic.com/research/rnn/rnn-cell |
| Forward Through Sequence | Implement a vanilla RNN forward pass that updates and returns hidden states across every sequence time step. | https://www.tensortonic.com/research/rnn/rnn-forward-sequence |
| Complete Vanilla RNN | Assemble a vanilla RNN that processes sequences into recurrent hidden states and per-time-step output logits. | https://www.tensortonic.com/research/rnn/rnn-full-network |
| Hidden State | Initialize a vanilla RNN hidden state as a floating-point zero matrix for the requested batch and hidden dimensions. | https://www.tensortonic.com/research/rnn/rnn-hidden-state |
| Vanishing Gradients | Simulate vanishing or exploding RNN gradients by repeatedly applying the hidden matrix's spectral norm. | https://www.tensortonic.com/research/rnn/rnn-vanishing-gradients |
| Scaled Dot-Product Attention | Implement scaled dot-product attention in PyTorch using query-key scores, softmax weights, and value aggregation. | https://www.tensortonic.com/research/transformer/transformers-attention |
| Embedding Layer | Create PyTorch token embeddings and scale each lookup by the square root of the Transformer model dimension. | https://www.tensortonic.com/research/transformer/transformers-embedding |
| Encoder Block | Assemble a Transformer encoder block with multi-head attention, residual paths, layer normalization, and a feed-forward network. | https://www.tensortonic.com/research/transformer/transformers-encoder-block |
| Feed-Forward Network | Implement the Transformer's position-wise feed-forward network with two linear projections and a ReLU activation. | https://www.tensortonic.com/research/transformer/transformers-feed-forward |
| Layer Normalization | Implement Transformer layer normalization in NumPy using per-token mean, variance, scale, and bias. | https://www.tensortonic.com/research/transformer/transformers-layer-normalization |
| Multi-Head Attention | Build NumPy multi-head attention with learned projections, per-head scaled attention, concatenation, and output projection. | https://www.tensortonic.com/research/transformer/transformers-multi-head-attention |
| Positional Encoding | Implement sinusoidal Transformer positional encodings in NumPy with alternating sine and cosine dimensions. | https://www.tensortonic.com/research/transformer/transformers-positional-encoding |
| Tokenization | Build a word-level Transformer tokenizer with fixed special-token IDs, sorted vocabulary entries, encoding, and decoding. | https://www.tensortonic.com/research/transformer/transformers-tokenization |
| U-Net Bottleneck | Implement the U-Net bottleneck shape transformation for two unpadded convolutions at the deepest encoder stage. | https://www.tensortonic.com/research/unet/unet-bottleneck |
| U-Net Decoder Block | Implement U-Net decoder shape transformations for up-convolution, cropped skip concatenation, and two valid convolutions. | https://www.tensortonic.com/research/unet/unet-decoder-block |
| U-Net Encoder Block | Implement U-Net encoder shape transformations for two unpadded 3x3 convolutions, a skip output, and 2x2 pooling. | https://www.tensortonic.com/research/unet/unet-encoder-block |
| U-Net Skip Connections | Implement U-Net skip connections by center-cropping encoder features and concatenating them with decoder features. | https://www.tensortonic.com/research/unet/unet-skip-connection |
| Class Token [CLS] | Prepend a learned classification token to each Vision Transformer patch sequence for image-level prediction. | https://www.tensortonic.com/research/vit/vit-class-token |
| ViT Encoder Block | Build a Vision Transformer encoder block with layer normalization, multi-head attention, MLP, and residual connections. | https://www.tensortonic.com/research/vit/vit-encoder-block |
| Complete Vision Transformer | Assemble a complete Vision Transformer from patch embedding, class and position tokens, encoder blocks, and classifier. | https://www.tensortonic.com/research/vit/vit-full-network |
| Classification Head | Implement the Vision Transformer classification head by normalizing and projecting the final class-token representation. | https://www.tensortonic.com/research/vit/vit-mlp-head |
| Patch Embedding | Implement Vision Transformer patch embeddings by splitting images into fixed patches and linearly projecting each patch. | https://www.tensortonic.com/research/vit/vit-patch-embedding |
| Position Embedding | Add learned positional embeddings to Vision Transformer patch-token sequences while preserving batch dimensions. | https://www.tensortonic.com/research/vit/vit-position-embedding |
| CBOW Forward Pass | Implement the Word2Vec CBOW forward pass by averaging context embeddings and producing vocabulary logits. | https://www.tensortonic.com/research/word2vec/word2vec-cbow-forward |
| Negative Sampling Distribution | Build the Word2Vec negative-sampling distribution from unigram counts raised to the three-quarter power. | https://www.tensortonic.com/research/word2vec/word2vec-noise-dist |
| Skip-gram Negative Sampling Loss | Implement skip-gram negative-sampling loss from center, positive-context, and negative-word embedding scores. | https://www.tensortonic.com/research/word2vec/word2vec-sgns-loss |
| SGNS Gradient Step | Implement one SGNS optimization step with positive and negative samples and in-place embedding gradient updates. | https://www.tensortonic.com/research/word2vec/word2vec-sgns-step |
| Skip-gram Pair Generation | Generate Word2Vec skip-gram training pairs by pairing each center token with words inside its context window. | https://www.tensortonic.com/research/word2vec/word2vec-skipgram-pairs |
| Frequent-Word Subsampling | Implement Word2Vec frequent-word subsampling by computing token retention probabilities from corpus frequencies. | https://www.tensortonic.com/research/word2vec/word2vec-subsampling |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/haoaka)
<!-- tensortonic:end -->
