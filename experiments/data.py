import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

def get_dataset(dataset):
    if dataset == "cifar10":
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        outdataset = torchvision.datasets.CIFAR10(root='./data', train=True,transform=transform,download=True)
    elif dataset == "mnist":
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))])
        outdataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    else:    
        raise ValueError("Unknown dataset")
    return outdataset


def split_dataset(data, train_num, test_num):
    train_data, test_data = train_test_split(data, train_size=train_num, test_size=test_num, random_state=42)
    return train_data, test_data
    