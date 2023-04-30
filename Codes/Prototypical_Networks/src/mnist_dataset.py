import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms




class MNISTDataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.train:
            self.dataset = datasets.MNIST(root=self.root, train=True, download=True, transform=self.transform, target_transform=self.target_transform)
        else:
            self.dataset = datasets.MNIST(root=self.root, train=False, download=True, transform=self.transform, target_transform=self.target_transform)

    def __getitem__(self, index):
        img, target = self.dataset[index]
        return img, target

    def __len__(self):
        return len(self.dataset)