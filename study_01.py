from __future__ import print_function
import torch
import numpy as np

# Tensors
x=torch.Tensor(5, 3)  # --> undefined tensor
print(x)
print(x.size())

y=torch.rand(5, 3)  # --> randomly defined tensor
print(y)
print(y.size())



# Operations
print(x + y)
print(torch.add(x, y))

result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print(result)

y.add_(x)  # --> add x to y
print(y)

x.copy_(y)  # --> copy y to x
print(x)

print(x[:, 1])  # --> indexing as like NumPy

x = torch.randn(4,4)
y = x.view(16)     # --> change size or shape of tensor x
z = x.view(-1, 8)  # --> if use '-1', the shape can be inferred.
print(x.size(), y.size(), z.size())



# Bridge 1: from Torch Tensor to NumPy
a = torch.ones(5)
print(a)
b = a.numpy()  # --> Transfer from Torch Tensor to NumPy
print(b)
# Because Torch Tensor and NumPy arrays share storage space, changing one will change the other.
a.add_(1)
print(a)
print(b)



# Bridge 2: from NumPy to Torch Tensor
a = np.ones(5)
print(a)
b = torch.from_numpy(a)  # --> Transfer from NumPy to Torch Tensor
print(b)
# Because Torch Tensor and NumPy arrays share storage space, changing one will change the other.
np.add(a, 1, out=a)
print(a)
print(b)



# CUDA Tensors
if torch.cuda.is_available():
    x = torch.Tensor(5,3)
    y = torch.Tensor(5,3)
    x = x.cuda()  # --> Put the tensor x on the GPU.
    y = y.cuda()  # --> Put the tensor y on the GPU.
    print(x + y)