# Neurix v1 Technical Documentation

System Specification, Mathematical Foundations, Architecture, and API Reference

---

## 1. Executive Summary and Architecture Overview

Neurix v1 is a modular, native C library and executable suite designed for training and executing Recurrent Neural Network (RNN) language models. Representing an advanced development and evolution of the neuralc C Neural Network framework, Neurix v1 is engineered from first principles without external linear algebra or deep learning library dependencies (such as BLAS, PyTorch, or TensorFlow).

The architecture consists of four distinct layers:

1. Subsystem Layer: Memory management, multi-threading (OpenMP), and hardware probing routines.
2. Tensor & Math Layer: N-dimensional floating-point array operations, strides, and memory views.
3. Neural Layer: Dense layers, activations (Softmax, ReLU, Linear), and Recurrent (RNN) layers with Backpropagation Through Time (BPTT).
4. Application & Interface Layer: Interactive TUI configuration utility (menuconfig), binary checkpoint serializer, and command-line inference engine.

---

## 2. Tensor Subsystem Architecture

### 2.1 Storage and Data Layout
The core array representation is defined by the `Tensor` structure in `include/tensor.h`. Tensors are allocated in continuous row-major 1D memory buffers (`float* data`), accompanied by explicit dimension arrays (`int shape[MAX_DIMS]`) and dimension counts (`int ndim`). Total elements (`size_t size`) are calculated dynamically upon allocation.

```c
typedef struct Tensor {
    float  *data;
    int     shape[MAX_DIMS];
    int     ndim;
    size_t  size;
} Tensor;
```

### 2.2 Views and Zero-Copy Reshaping
To prevent excessive memory allocation during sequence processing, Neurix v1 supports zero-copy tensor reshaping via `tensor_reshape()`. When converting between a 3D batch sequence tensor `[batch_size, seq_len, hidden_size]` and a 2D matrix view `[batch_size * seq_len, hidden_size]`, `tensor_reshape()` creates a lightweight view structure pointing to the same underlying raw `data` pointer.

---

## 3. Neural Network Layers and Activations

### 3.1 Dense Layer (Linear Transformation)
The `DenseLayer` structure executes affine transformations followed by non-linear activation functions:

$$Y = \text{Activation}(X \cdot W + b)$$

Where:
- $X$ is the input tensor of shape $[N, D_{in}]$.
- $W$ is the weight matrix of shape $[D_{in}, D_{out}]$.
- $b$ is the bias vector of shape $[D_{out}]$.

During backward propagation (`dense_backward()`), input gradients $dX$, weight gradients $dW$, and bias gradients $db$ are accumulated:

$$dW = X^T \cdot dY$$
$$db = \sum_{i=1}^N dY_i$$
$$dX = dY \cdot W^T$$

### 3.2 Activation Functions
Supported activation types:
- `ACT_NONE`: Linear pass-through.
- `ACT_RELU`: Rectified Linear Unit ($f(x) = \max(0, x)$).
- `ACT_SOFTMAX`: Normalized exponential probability distribution over logits.

---

## 4. Recurrent Neural Network (RNN) and BPTT

### 4.1 Forward Pass Equations
The `RNNLayer` processes input sequence tokens over discrete timesteps $t \in [1, T]$. For input sequence $X_t$ and previous hidden state $h_{t-1}$, the recurrent state transition is defined as:

$$h_t = \tanh(X_t \cdot W_{xh} + h_{t-1} \cdot W_{hh} + b_h)$$

Where:
- $W_{xh}$ represents input-to-hidden transformation weights.
- $W_{hh}$ represents hidden-to-hidden recurrent weights.
- $b_h$ represents hidden bias terms.

### 4.2 Backpropagation Through Time (BPTT)
During backward propagation (`rnn_backward()`), gradients flow backward from $t = T$ down to $t = 1$. The error gradient with respect to the pre-activation state $dh_t$ accumulates both the layer output gradient $dY_t$ and the recurrent gradient from timestep $t+1$:

$$dh_t = (dY_t + dh_{t+1} \cdot W_{hh}^T) \odot (1 - h_t^2)$$

Gradients for weight matrices are accumulated across all sequence timesteps:

$$dW_{xh} = \sum_{t=1}^T X_t^T \cdot dh_t$$
$$dW_{hh} = \sum_{t=1}^T h_{t-1}^T \cdot dh_t$$
$$db_h = \sum_{t=1}^T dh_t$$

---

## 5. Optimization and Gradient Management

### 5.1 Stochastic Gradient Descent (SGD) with Momentum
Updates parameters $\theta$ using velocity vectors $v$:

$$v_{t+1} = \gamma v_t + \eta \left( \nabla_\theta L + \lambda \theta \right)$$
$$\theta_{t+1} = \theta_t - v_{t+1}$$

Where:
- $\eta$ is the learning rate (`LEARNING_RATE`).
- $\gamma$ is momentum (`MOMENTUM`, default $0.9$).
- $\lambda$ is weight decay (`WEIGHT_DECAY`).

### 5.2 L2 Gradient Clipping
To stabilize recurrent training and prevent exploding gradients, `rnn_clip_gradients()` scales gradients if their combined L2 norm exceeds a maximum threshold $g_{max}$:

$$\text{Norm} = \sqrt{ \sum \|\nabla_\theta\|^2 }$$

$$\text{If } \text{Norm} > g_{max}, \quad \nabla_\theta \leftarrow \nabla_\theta \cdot \frac{g_{max}}{\text{Norm}}$$

---

## 6. Tokenization and Binary Dataset Streaming

### 6.1 Vocabulary Encoding
The tokenizer system (`src/tokenizer.c`) manages mapping between text strings and discrete integer IDs. It supports character-level tokenization and marker-bounded word-level tokenization.

### 6.2 Streamed Binary Dataset Loader
To accommodate large text corpora without filling system memory, `dataset_loader.c` converts input corpora into a compact, uint32-encoded binary format (`DatasetHeader` + token arrays). `dataset_next_batch()` streams sliding window input sequences and target sequences continuously with wraparound logic.

---

## 7. Inference and Logit Sampling Engine

Sampling algorithms in `src/sampler.c` handle logit post-processing prior to token selection:

1. Greedy Sampling: Selects the token with maximum logit probability ($\text{argmax}$).
2. Temperature Scaling: Rescales raw logits $z_i$ prior to Softmax:

$$\hat{z}_i = \frac{z_i}{T}$$

3. Top-K Sampling: Truncates candidate choices to the $K$ highest probability logits.
4. Top-P (Nucleus) Sampling: Restricts selection to the cumulative probability mass threshold $P \in (0, 1]$.

---

## 8. Terminal UI (TUI) Menuconfig Subsystem

### 8.1 Zero-Dependency ANSI Architecture
The configuration utility in `config/config_ui.c` renders an interactive terminal UI directly via ANSI escape codes and POSIX raw terminal modes (`termios`).

Key mechanisms:
- Terminal Raw Mode: Disables `ECHO` and canonical input buffering via `tcsetattr()`.
- System Hardware Probing: Reads `/proc/cpuinfo` and `/sys/class/drm/` to auto-detect CPU cores and GPU capabilities.
- Configuration Serialization: Writes parameter selections to `neuralc_config.h`.

---

## 9. Hardware Acceleration and Threading Model

### 9.1 OpenMP Parallel Execution
Tensor operations (such as matrix fills, activations, and one-hot encoding loops) feature OpenMP pragmas (`#pragma omp parallel for`). Multi-threading dynamically scales to the hardware core count detected during startup.

### 9.2 CUDA / OpenCL Abstraction Layers
For GPU hardware, backend implementations (`cuda_backend.cu` and `opencl_backend.c`) offload matrix multiplication operations to GPU VRAM buffers when enabled in `neuralc_config.h`.

---

## 10. Verification and Execution Protocols

### 10.1 Compiling and Running Menuconfig
```bash
make config
```

### 10.2 Training the Model
```bash
make train
./train assets/data.txt assets/vocab.txt model.bin
```

### 10.3 Interactive Standalone Chatbot CLI Execution
```bash
make neurix
./neurix
```

Or install `neurix` globally:

```bash
make install
neurix
```

Features auto-discovery of `model.bin` and `assets/vocab.txt`, live token streaming, and dynamic `/temp`, `/topk`, `/reset`, and `/exit` slash commands inside the REPL loop.

### 10.4 Model Sanity Verification
```bash
make sanity
./sanity model.bin assets/vocab.txt "The machine"
```
