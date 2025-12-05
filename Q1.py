import numpy as np

def softmax(x, axis=-1):
    """
    Numerically stable softmax along a given axis.
    """
    # subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V):
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query matrix of shape (seq_len_q, d_k)
        K: Key matrix of shape (seq_len_k, d_k)
        V: Value matrix of shape (seq_len_k, d_v)

    Returns:
        attn_weights: Attention weights of shape (seq_len_q, seq_len_k)
        context: Context matrix of shape (seq_len_q, d_v)
    """
    # 1. Compute raw scores: (seq_len_q, d_k) @ (d_k, seq_len_k) -> (seq_len_q, seq_len_k)
    scores = np.matmul(Q, K.T)

    # 2. Scale by sqrt(d_k)
    d_k = K.shape[-1]
    scores = scores / np.sqrt(d_k)

    # 3. Softmax over keys dimension to get attention weights
    attn_weights = softmax(scores, axis=-1)

    # 4. Weighted sum of values: (seq_len_q, seq_len_k) @ (seq_len_k, d_v) -> (seq_len_q, d_v)
    context = np.matmul(attn_weights, V)

    return attn_weights, context
if __name__ == "__main__":
    # Simple test to verify the function works
    np.random.seed(0)

    # Example shapes: 3 query tokens, 4 key/value tokens, d_k = d_v = 5
    Q = np.random.rand(3, 5)
    K = np.random.rand(4, 5)
    V = np.random.rand(4, 5)

    attn_weights, context = scaled_dot_product_attention(Q, K, V)

    print("Attention weights shape:", attn_weights.shape)
    print(attn_weights)
    print("Context shape:", context.shape)
    print(context)
