import torch
from torchvision import transforms, datasets

def load_mnist(is_train=True):
    compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])
    out_dir = "data"
    return datasets.MNIST(root=out_dir, train=is_train,
                          transform=compose, download=True)

if __name__ == '__main__':
    data = load_mnist()
    print(data[0][0].shape)