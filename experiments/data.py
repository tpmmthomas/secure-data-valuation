import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

def get_dataset(dataset):
    if dataset == "cifar10":
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        outdataset = torchvision.datasets.CIFAR10(root='./data', train=False,transform=transform,download=True)
    elif dataset == "cifar100":
        outdataset = torchvision.datasets.CIFAR100
    else:
        raise ValueError("Unknown dataset")
    return outdataset


def split_dataset(data, train_num, test_num):
    train_data, test_data = train_test_split(data, train_size=train_num, test_size=test_num, random_state=42)
    return train_data, test_data
    