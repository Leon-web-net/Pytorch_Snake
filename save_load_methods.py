import torch
import torch.nn as nn

# This works fine for cpu but need changes for gpu check => save_load_gpu.py

# # option 1 (lazy)
# torch.save(model, PATH)
#
# #model class defined
# model = torch.load(PATH)
# model.eval()
#
# # option 2 recommended
# torch.save(model.state_dict(),PATH)
#
# # model created again with parameters
# model =Model(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self,x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model(n_input_features=6)

FILE = "Dataset/model_saves/model2.pth"
torch.save(model.state_dict(),FILE)

# option 1
# model = torch.load(FILE)
# model.eval()
#
# for param in model.parameters():
#     print(param)

# option 2
loaded_model = Model(n_input_features=6)
# loaded_model.load_state_dict(torch.load(FILE))
# loaded_model.eval()
#
# for param in loaded_model.parameters():
#     print(param)

# save a checkpoint
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(optimizer.state_dict())

# checkpoint = {
#     "epoch": 90,
#     "model_state": model.state_dict(),
#     "optim_state": optimizer.state_dict()
# }

# torch.save(checkpoint, "Dataset/model_saves/checkpoint.pth")

loaded_checkpoint = torch.load("Dataset/model_saves/checkpoint.pth")
epoch = loaded_checkpoint["epoch"]
model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0)

model.load_state_dict(loaded_checkpoint["model_state"])
optimizer.load_state_dict(loaded_checkpoint["optim_state"])

print(optimizer.state_dict())