# Flash Attention 2 Algorithm

## Standard Attention

Standard multi-head attention computes:

```
S = Q @ K^T / sqrt(d)      # [N, N] attention scores
P = softmax(S, dim=-1)     # [N, N] attention weights (with causal mask)
O = P @ V                  # [N, d] output
```

This requires O(N²) memory for the intermediate S and P matrices.

## Flash Attention 2 Forward Pass

Flash Attention avoids materializing the N×N matrices by fusing all three steps into a single kernel that processes tiles of Q, K, V using **online softmax**.

### Online Softmax

The key insight: softmax can be computed incrementally. For each new block of scores, we:
1. Update the running maximum: `m_new = max(m_old, block_max)`
2. Rescale the previous accumulator: `O *= exp(m_old - m_new)`
3. Add the new contribution: `O += exp(S_block - m_new) @ V_block`
4. Update the running sum: `l = l * exp(m_old - m_new) + block_sum`

After processing all blocks: `O = O / l`

### Tiling Strategy

- Q is split into blocks of size Br (rows)
- K, V are split into blocks of size Bc (columns)
- **Outer loop** (parallelized across GPU blocks): iterate over Q blocks
- **Inner loop** (sequential): iterate over K/V blocks
- Causal masking: skip K/V blocks where all keys are in the future

### Memory

- Shared memory: Qi[Br×d] + Kj[Bc×d] + Vj[Bc×d] + S[Br×Bc]
- Global memory: only O[N×d] and L[N] (logsumexp for backward)
- **No N×N matrix stored** — O(N) memory instead of O(N²)

## References

- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) (Dao et al., 2022)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (Dao, 2023)
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) (Milakov & Gimelshein, 2018)
