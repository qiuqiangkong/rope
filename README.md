# Rotary Positional Embeddings (RoPE)

A pytorch implementation of RoPE. The code is modified from [1].

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

<img src="https://github.com/user-attachments/assets/99c28921-267d-477b-8785-de6a5afa3079" width="600">

## References

```bibtex
@misc{su2021roformer,
    title   = {RoFormer: Enhanced Transformer with Rotary Position Embedding}, 
    author  = {Jianlin Su and Yu Lu and Shengfeng Pan and Bo Wen and Yunfeng Liu},
    year    = {2021},
    eprint  = {2104.09864},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```

## External Links

[1] https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
