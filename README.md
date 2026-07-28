# Neurix v1 (neuralc)

A native, zero-dependency Neural Network & Language Model Engine written from scratch in pure C.

---

## Overview

Neurix v1 is a lightweight, high-performance C engine designed for training and running Recurrent Neural Network (RNN) language models. Built as an advanced evolution and development of the neuralc C Neural Network library framework, Neurix v1 operates without external deep learning libraries (such as PyTorch or TensorFlow), implementing every layer, matrix operation, backpropagation algorithm, and terminal configuration utility from first principles.

### Key Highlights
- Zero External Dependencies: Core math, tensor routines, and interactive configuration TUI built strictly in standard C (C11/POSIX).
- Kernel-Style menuconfig TUI: Linux kernel-inspired terminal interface to configure hyper-parameters, hardware bindings, and compiler flags without writing code.
- Full Training & Inference Stack:
  - Custom N-dimensional Tensor engine with row-major memory views.
  - Dense (Linear) layers, Softmax, and Recurrent Neural Network (RNN) layers with Backpropagation Through Time (BPTT).
  - SGD Optimizer with Momentum, Weight Decay, and L2 Gradient Clipping.
  - Custom Tokenizer (Char/Word) & Streamed Binary Dataset Loader.
  - Advanced Samplers: Temperature, Top-K, Top-P (Nucleus), and Greedy sampling.
- Hardware Acceleration: Multi-threaded execution via OpenMP with support for CUDA and OpenCL backends.

---

## Repository Structure

```text
Neurixv1/
├── apps/                  # Executable applications (train, cli, pipeline)
│   ├── train.c            # Model training pipeline
│   ├── cli.c              # Interactive text generation CLI
│   └── pipeline.c         # Inference pipeline library
├── config/                # TUI Configuration engine (menuconfig)
│   ├── config_ui.c        # ANSI/termios terminal UI implementation
│   ├── config_ui.h        # Configuration structs & definitions
│   └── neuralc_config_main.c # Main entry point for menuconfig
├── include/               # Public headers (tensor, rnn, nn, optimizer, etc.)
├── src/                   # Core math, layers, samplers, and tokenizers
│   ├── tensor.c           # Matrix/Tensor memory management & math
│   ├── rnn.c              # Recurrent layer implementation & BPTT
│   ├── layer.c            # Dense layers & activations
│   ├── optimizer.c        # SGD, Momentum, and Gradient clipping
│   ├── tokenizer.c        # Vocabulary encoding & decoding
│   ├── sampler.c          # Logit sampling algorithms
│   └── dataset_loader.c   # Streaming binary batch loader
├── tools/                 # Diagnostic and sanity testing tools
│   └── sanity_test.c      # Model check & tensor validation tool
├── assets/                # Data corpora and vocabulary files
├── Makefile               # Build automation & auto-config auto-loader
└── neuralc_config.h       # Auto-generated parameter manifest (via menuconfig)
```

---

## Quickstart Guide

### 1. Prerequisites
- gcc compiler (with C11 support)
- POSIX Terminal (Linux/macOS)
- make build tool

### 2. Configure Settings (menuconfig)
Launch the interactive TUI configuration menu:

```bash
make config
```

Use Arrow Keys to navigate, Space/Enter to toggle or select values, and S to save settings to neuralc_config.h.

### 3. Train a Model
Train an RNN language model on your dataset:

```bash
make train
./train assets/data.txt assets/vocab.txt model.bin
```

During training, train streams mini-batches from assets/data.txt, tokenizes tokens on-the-fly, computes cross-entropy loss, clips gradients, updates weights via SGD, and writes model.bin.

### 4. Run Text Generation (Neurix CLI)
You can launch the interactive chatbot terminal directly without needing file paths:

```bash
make neurix
./neurix
```

Or install it globally to your user path:

```bash
make install
neurix
```

When run, neurix automatically discovers model.bin and assets/vocab.txt, renders a terminal prompt UI (User >), and streams token generation in real time. Inside the terminal shell, you can use slash commands:
- /help — Show help and active sampling settings
- /temp <val> — Adjust temperature dynamically (e.g. /temp 0.7)
- /topk <val> — Adjust Top-K sampling dynamically (e.g. /topk 40)
- /reset — Reset hidden state memory
- /exit — Exit shell

### 5. Run Model Sanity Diagnostic
Validate tensor integrity, logit distributions, and step-by-step decoding:

```bash
make sanity
./sanity model.bin assets/vocab.txt "The machine"
```

---

## Configuration Parameters (neuralc_config.h)

Parameters can be adjusted interactively via make config or passed via #define:

| Parameter | Macro Name | Description | Default |
|---|---|---|---|
| Hidden Size | HIDDEN_SIZE | Dimension of RNN hidden state vectors | 256 |
| Sequence Length | SEQ_LEN | Number of timesteps per training batch | 20 |
| Sampling Temp | TEMPERATURE | Temperature factor for output logit scaling | 0.8000 |
| Batch Size | BATCH_SIZE | Samples per gradient update step | 8 |
| Learning Rate | LEARNING_RATE | Step size for gradient updates | 0.0100 |
| Epoch Cycles | EPOCHS | Total passes over dataset | 20 |
| OpenMP Threads | NEURALC_OMP_THREADS | Number of CPU cores bound to OpenMP ops | Auto |

---

## Build Commands Summary

- make config — Build & auto-run interactive menuconfig TUI.
- make train — Build model trainer (./train).
- make cli — Build prompt inference shell (./cli).
- make sanity — Build diagnostic verification tool (./sanity).
- make all — Build all primary engine targets.
- make clean — Remove object files and compiled binaries.

---

## License

Distributed under the Apache License 2.0. See LICENSE for details.
