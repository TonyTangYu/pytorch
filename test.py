import torch
torch.annotate_log("hello")
x = torch.Tensor([1]).checkpoint()
y = torch.Tensor([2]).checkpoint()
z = x + y
x.data = z
