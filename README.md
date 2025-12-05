Student: Vancha Akhil Reddy
Student ID: 700772768


📌 Overview

This assignment contains two coding problems related to core components of the Transformer architecture:

Q1: Implementing Scaled Dot-Product Attention using NumPy

Q2: Implementing a simplified Transformer Encoder Block using PyTorch

Each question includes function implementations and optional test code to verify correctness.

✅ Q1 – Scaled Dot-Product Attention (NumPy)
Files Included

Q1.py

Description

In this task, I implemented the Scaled Dot-Product Attention mechanism as defined in the "Attention Is All You Need" paper.

The function performs:

Compute raw attention scores using matrix multiplication:

scores
=
𝑄
𝐾
⊤
scores=QK
⊤

Scale scores by 
𝑑
𝑘
d
k
	​

	​


Apply a numerically stable softmax

Compute the context vector:

context
=
softmax(scores)
⋅
𝑉
context=softmax(scores)⋅V
Functions Implemented
✔ softmax(x, axis=-1)

A numerically stable softmax that subtracts the max value to avoid overflow.

✔ scaled_dot_product_attention(Q, K, V)

Returns:

attn_weights: attention weights

context: the resulting context vectors

How to Run
python Q1.py


If the test block is enabled, the script will print:

Attention weights shape: (seq_len_q, seq_len_k)

Context shape: (seq_len_q, d_v)

✅ Q2 – Transformer Encoder Block (PyTorch)
Files Included

Q2.py

Description

This task implements a simplified version of the Transformer Encoder Block, including:

Multi-Head Self-Attention

Feed-Forward Network (FFN)

Residual Connections

Layer Normalization

Dropout (optional)

This follows the structure of the original Transformer model.

Class Implemented
✔ SimpleTransformerEncoderBlock(d_model, num_heads, d_ff, dropout)

The forward(x) method performs:

Multi-Head Self-Attention

Add & Norm

Feed-Forward Network

Add & Norm again

How to Run

Make sure PyTorch is installed in the environment.

python Q2.py


If the test block is included, it will print:

Input shape : torch.Size([32, 10, d_model])
Output shape: torch.Size([32, 10, d_model])


confirming the encoder block returns output in the same shape as the input.
