#pytorch imports
import torch
import torchvision
import torchvision.transforms as transforms

#runtime measurement import (kept from original even though used elsewhere)
import time

#seed control imports
import random
import numpy as np

def get_data(seed=1):

    #standardize the seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    #prevents gpu from trying to pick a faster algorithm dynamically, need it use the same one each time for consistency
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #converts it to a tensor and on the range of [0,1]
    transform = transforms.Compose([transforms.ToTensor()])

    #loads CIFAR-10, loads train/test set based on the boolean (second argument) and applies the transform
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    #bratches dataset in randomized batches of 64
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    return trainloader, testloader