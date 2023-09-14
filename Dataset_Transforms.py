import torchvision
import torch

dataset = torchvision.datasets.MNIST(
    root="./data", transform=torchvision.transforms.ToTensor()
)