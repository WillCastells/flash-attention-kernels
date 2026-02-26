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

## Flash Attention 2 Backward Pass

The backward pass recomputes S and P on-the-fly using the stored logsumexp L, avoiding O(N²) memory for gradients too.

### Gradient Formulas

Given upstream gradient dO:

```
D[i] = sum_j(O[i,j] * dO[i,j])              # row-wise dot product
P[i,j] = exp(Q[i] @ K[j]^T * scale - L[i])  # recomputed from L
dS[i,j] = P[i,j] * (dO[i] @ V[j]^T - D[i])  # gradient of pre-softmax scores
dV += P^T @ dO                                # gradient w.r.t. values
dK += scale * dS^T @ Q                        # gradient w.r.t. keys
dQ += scale * dS @ K                          # gradient w.r.t. queries
```

### Tiling Strategy (Backward)

- **Outer loop** over K/V blocks (j): accumulates dKj, dVj in shared memory
- **Inner loop** over Q blocks (i): recomputes P, computes dS
- dQ uses atomicAdd to global memory (accumulated across j-blocks)
- dK, dV are written once per j-block (no atomics needed)

## Optimizations (V4)

1. **Register-tiled matmul**: Each thread computes multiple output elements using split-K dot products
2. **Warp shuffle reductions**: Use `__shfl_xor_sync` for rowmax/rowsum across collaborating threads
3. **float4 vectorized loads**: 128-bit memory transactions for global→register transfers
4. **Split backward**: Separate dK/dV and dQ kernels eliminate all atomicAdd operations
5. **Template parameters**: Compile-time tile sizes enable loop unrolling

## References

- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) (Dao et al., 2022)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (Dao, 2023)
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) (Milakov & Gimelshein, 2018)
