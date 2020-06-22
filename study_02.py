import torch
from torch.autograd import Variable

# Variable
x = Variable(torch.ones(2, 2), requires_grad=True)  # --> Generate a variable
print(x)
print(x.grad_fn)

y = x + 2
print(y)
print(y.grad_fn)

z = y*y*3
out = z.mean()
print(z, out)



# Gradient
out.backward()  # backpropagation
print(x.grad)