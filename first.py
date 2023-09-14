import torch

# x = torch.randn(3, requires_grad=True)
# print(x)

# x.requires_grad_(False)
# x.detach()
# with torch.no_grad():

weights = torch.ones(4, requires_grad=True)

for epoch in range(6):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()