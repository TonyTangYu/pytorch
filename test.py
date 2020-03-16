import torch
x = torch.Tensor([1]).checkpoint()
y = torch.Tensor([2]).checkpoint()
z = x + y
print(z)
print(z.decheckpoint())
print(z.is_checkpoint())
print(z.decheckpoint().is_checkpoint())
