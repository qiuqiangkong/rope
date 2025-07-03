# Rotary Positional Embeddings (RoPE)

A pytorch implementation of RoPE. The code is modified from [1].

## Usage

Example 1: RoPE (1D)

```python
B = 4  # batch_size (b)
L = 100  # seq_len (l)
N = 8  # heads_num (n)
H = 24  # head_dim (h)

rope = RoPE(head_dim=H)
x = torch.rand((B, L, N, H))
out = rope(x)  # (b, l, n, h)
```

Example 2: RoPE (1D) with sparse positions

```python
rope = RoPE(head_dim=H)
x = torch.rand((B, 4, N, H))
pos = torch.LongTensor([[0], [3], [7], [8]])  # (l, 1)
out = rope.apply_nd(x, pos)  # (b, l, n, h)
```

Example 3: RoPE (2D image) with sparse positions

```python
data_dim = 2
rope = RoPE(head_dim=H // data_dim)
x = torch.rand((B, 4, N, H))
pos = torch.LongTensor([[0, 0], [0, 1], [1, 0], [1, 1]])  # (l, 2)
out = rope.apply_nd(x, pos)  # (b, l, n, h)
```

Example 4: RoPE (3D video) with sparse positions

```python
data_dim = 3
rope = RoPE(head_dim=H // data_dim)
x = torch.rand((B, 4, N, H))
pos = torch.LongTensor([[0, 0, 0], [1, 3, 4], [2, 2, 2], [5, 4, 3]])  # (l, 3)
out3 = rope.apply_nd(x, pos)  # (b, l, n, h)
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
