"""
Test a 3*3 table (one token per cell) with one page title token, one section title token and one PAD token
"""
import torch

from model.invariant_position import compute_invariant_position


type_ids = torch.Tensor([1,2,3,3,3,3,3,3,3,3,3,0]).reshape(1,-1)
row_ids = torch.Tensor([0,0,1,1,1,2,2,2,3,3,3,0]).reshape(1,-1)
col_ids = torch.Tensor([0,0,1,2,3,1,2,3,1,2,3,0]).reshape(1,-1)

gold_position = torch.Tensor([[
         [  0,   1, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512],
         [ -1,   0, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512],
         [512, 512,   0, 512, 512, 512, 512, 512, 512, 512, 512, 512],
         [512, 512, 512,   0, 512, 512, 512, 512, 512, 512, 512, 512],
         [512, 512, 512, 512,   0, 512, 512, 512, 512, 512, 512, 512],
         [512, 512, 512, 512, 512,   0, 512, 512, 512, 512, 512, 512],
         [512, 512, 512, 512, 512, 512,   0, 512, 512, 512, 512, 512],
         [512, 512, 512, 512, 512, 512, 512,   0, 512, 512, 512, 512],
         [512, 512, 512, 512, 512, 512, 512, 512,   0, 512, 512, 512],
         [512, 512, 512, 512, 512, 512, 512, 512, 512,   0, 512, 512],
         [512, 512, 512, 512, 512, 512, 512, 512, 512, 512,   0, 512],
         [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512]]])

relative_position = compute_invariant_position(12, 12, type_ids, row_ids, col_ids)

assert torch.all(torch.eq(relative_position,gold_position)) == torch.BoolTensor([True])

print("Test Passed.")

