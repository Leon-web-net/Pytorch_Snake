import torch
import numpy as np


device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
x = torch.ones(5, device=device)
y =torch.ones(5)
y = y.to(device)
z= x+y
print(z)
z = z.to("cpu")

print(z)
