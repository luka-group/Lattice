"""
Test a 3*3 table (one token per cell) with one page title token, one section title token and one PAD token
"""
import torch

from model.structural_attention import rewrite_encoder_attention_mask


general_attention_mask = torch.Tensor([1,1,1,1,1,1,1,1,1,1,1,0]).reshape(1,-1)
type_ids = torch.Tensor([1,2,3,3,3,3,3,3,3,3,3,0]).reshape(1,-1)
row_ids = torch.Tensor([0,0,1,1,1,2,2,2,3,3,3,0]).reshape(1,-1)
col_ids = torch.Tensor([0,0,1,2,3,1,2,3,1,2,3,0]).reshape(1,-1)
type_edges = ((1, 1), (2, 2), (1, 2), (2, 1), (3, 1), (3, 2), (1, 3), (2, 3))

gold_attention = torch.BoolTensor([[
    [1,1,1,1,1,1,1,1,1,1,1,0],
    [1,1,1,1,1,1,1,1,1,1,1,0],
    [1,1,1,1,1,1,0,0,1,0,0,0],
    [1,1,1,1,1,0,1,0,0,1,0,0],
    [1,1,1,1,1,0,0,1,0,0,1,0],
    [1,1,1,0,0,1,1,1,1,0,0,0],
    [1,1,0,1,0,1,1,1,0,1,0,0],
    [1,1,0,0,1,1,1,1,0,0,1,0],
    [1,1,1,0,0,1,0,0,1,1,1,0],
    [1,1,0,1,0,0,1,0,1,1,1,0],
    [1,1,0,0,1,0,0,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0]
]])

attention_mask = rewrite_encoder_attention_mask(general_attention_mask, type_ids, row_ids, col_ids, type_edges)

assert torch.all(torch.eq(attention_mask,gold_attention)) == torch.BoolTensor([True])

print("Test Passed.")
