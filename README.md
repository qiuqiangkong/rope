# Rotary Positional Embeddings (RoPE)

A pytorch implementation of RoPE [1]. The code is modified from [2].

## Usage

```python
B = 4  # batch_size
T = 100  # time_steps
N = 16  # heads_num
H = 32  # head_dim
L = 1000  # rope_len

rope = build_rope(seq_len=L, head_dim=H)  # (l, h/2, 2)
x = torch.Tensor(B, T, N, H)  # (b, t, n, h)

# Apply RoPE
out = apply_rope(x, rope)  # (b, t, n, h)
```

## Visualization of rope:

![RoPE matrix](./assets/rope.png)

## References

[1] Su, J., Zhang, H., Li, X., Zhang, J. and Li, Y.R., August. Enhanced transformer with rotary position embedding. ACL-IJCNLP, 2021.

External Links

[2] https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py