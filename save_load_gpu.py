import torch
import torch.nn as nn

# Save on GPU, Load on CPU

device = torch.device("cuda")
module.to(device)
torch.save(model.state_dict(), PATH)

device = torch.device("cpu")
module =Model()
model.load_state_dict(torch.load(PATH, map_location=device))


# Save on GPU, Load on GPU

device = torch.device("cuda")
module.to(device)
torch.save(model.state_dict(), PATH)


module =Model()
model.load_state_dict(torch.load(PATH))
model.to(device)

# Save on CPU, Load on GPU
torch.save(model.state_dict(),PATH)

device = torch.device("cuda")
module =Model()
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))
model.to(device)