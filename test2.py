import torch

a = [1.2, 1.4, 0.5e-2]
b = [1.2561, 1.4425, 0.5425e-2]

a1 = torch.tensor(a)
b1 = torch.tensor(b)
print(a1, b1)
