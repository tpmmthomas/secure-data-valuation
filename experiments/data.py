import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
import random
import numpy as np
import math
from PIL import Image
from collections import defaultdict


def preprocess_features(X, preprocessor=None):
    # Define feature groups
    categorical_features = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country'
    ]
    numeric_features = [
        'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'
    ]
    # If no preprocessor is provided, create and fit one.
    if preprocessor is None:
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            # For scikit-learn < 1.2 use sparse=False; for >=1.2, you may use sparse_output=False.
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        X_preprocessed = preprocessor.fit_transform(X)
        return X_preprocessed, preprocessor
    else:
        X_preprocessed = preprocessor.transform(X)
        return X_preprocessed
    
def preprocess_targets(y):
    y_processed = np.where(y.to_numpy() == '>50K.', 1, 0)
    return y_processed.reshape(-1, )

def get_dataset(dataset):
    if dataset == "cifar10":
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        outdataset = torchvision.datasets.CIFAR10(root='./data', train=True,transform=transform,download=True)
    elif dataset == "mnist":
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))])
        outdataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    elif dataset == "adult":
        adult = fetch_ucirepo(id=2) 
        X_preprocessed, _ = preprocess_features(adult.data.features)
        y_preprocessed = preprocess_targets(adult.data.targets)
        outdataset = [(x,y) for x,y in zip(X_preprocessed, y_preprocessed)]
    else:    
        raise ValueError("Unknown dataset")
    return outdataset


def split_dataset(data, train_num, test_num):
    train_data, test_data = train_test_split(data, train_size=train_num, test_size=test_num, random_state=42)
    return train_data, test_data

def add_noise(data, var=None):
    if var is None:
        var = random.random() * 2 + 0.1
    def _add_noise(data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        noise = torch.randn_like(data) * (var ** 0.5)
        return data + noise
    return [(_add_noise(x), y) for x, y in data]



def generate_balanced_distribution(num_classes, per_batch):
    """
    Create a distribution where each of the `num_classes` gets ~ per_batch/num_classes items.
    """
    base = per_batch // num_classes
    dist = [base] * num_classes
    leftover = per_batch - sum(dist)
    i = 0
    while leftover > 0:
        dist[i % num_classes] += 1
        leftover -= 1
        i += 1
    return dist

def generate_skewed_distribution(num_classes, per_batch, ensure_extreme=False):
    """
    Create a skewed distribution:
      - If ensure_extreme=True, put all per_batch items in a single class (completely skewed).
      - Otherwise, generate a random distribution that is likely unbalanced.
    """
    if ensure_extreme:
        # Single-class batch (but not necessarily all identical images)
        single_class = random.randint(0, num_classes - 1)
        dist = [0] * num_classes
        dist[single_class] = per_batch
        return dist
    
    # Random approach that typically yields a skewed distribution
    raw_counts = [random.randint(0, 10) for _ in range(num_classes)]
    total_counts = sum(raw_counts)
    if total_counts == 0:
        # fallback: if all zero, pick uniform
        raw_counts = [1]*num_classes
        total_counts = num_classes
    
    scaled_counts = [math.floor(c / total_counts * per_batch) for c in raw_counts]
    shortfall = per_batch - sum(scaled_counts)
    while shortfall > 0:
        idx = random.randint(0, num_classes-1)
        scaled_counts[idx] += 1
        shortfall -= 1
    
    return scaled_counts

def sample_from_distribution(class_to_data, dist):
    """
    Given a distribution `dist` of length num_classes, draw the specified number of samples
    from each class's list in `class_to_data`. Remove them from the pool.
    Returns a list of (image, label).
    """
    batch = []
    for class_idx, needed in enumerate(dist):
        if needed <= 0:
            continue
        
        available = class_to_data[class_idx]
        actual_count = min(needed, len(available))
        
        chosen = available[:actual_count]
        class_to_data[class_idx] = available[actual_count:]  # remove used samples
        batch.extend(chosen)
    
    return batch

def replicate_images_in_one_class(batch):
    """
    Pick one class that appears in the batch.
    Then replace *all* images of that class with the same (identical) image
    chosen from among that class’s images in this batch.
    
    For example, if the batch has 5 images from class A and 5 from class B,
    and we choose class A, we pick 1 of those 5 images from A, and replicate it
    for all 5 A images. The 5 B images remain unchanged.
    """
    # Identify which classes appear in the batch
    label_to_indices = {}
    for i, (img, lbl) in enumerate(batch):
        label_to_indices.setdefault(lbl, []).append(i)
    
    # If there's no variety or no images at all, do nothing
    if not label_to_indices:
        return
    
    # Pick one random class from the ones present
    chosen_class = random.choice(list(label_to_indices.keys()))
    indices = label_to_indices[chosen_class]
    
    # If that class has at least 1 image, pick one of them to replicate
    if len(indices) >= 1:
        chosen_index = random.choice(indices)
        chosen_image, chosen_label = batch[chosen_index]
        
        # Now replicate that image for all indices in the chosen class
        for i in indices:
            batch[i] = (chosen_image, chosen_label)

def degrade_batch(batch, degrade_prob=0.5):
    """
    Degrade the entire batch or none of itransforms.
    If random.random() < degrade_prob, degrade every image in the batch;
    otherwise, leave them as-is.
    """
    degrade_entire_batch = (random.random() < degrade_prob)
    if not degrade_entire_batch:
        return  # do nothing

    # Degrade each image in the batch
    for i in range(len(batch)):
        img, label = batch[i]
        # Convert Tensor -> PIL if needed
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        
        # Apply degrade_image pipeline
        img = degrade_image(img)
        
        # Convert back to Tensor for consistency
        img = transforms.ToTensor()(img)
        batch[i] = (img, label)

def degrade_image(pil_img):
    """
    Example "quality degradation" transform pipeline:
      1. Possibly blur
      2. Possibly downscale & re-upscale
      3. Possibly color jitter
    """
    w, h = pil_img.size
    transforms_list = []
    
    # 1. Randomly blur
    if random.random() < 0.7:
        kernel_size = random.choice([3, 5])
        transforms_list.append(transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0)))
    
    # 2. Randomly downscale and upscale
    if random.random() < 0.7:
        new_size = random.randint(8, 24)  # e.g. for CIFAR (32x32)
        transforms_list.append(transforms.Resize((new_size, new_size), interpolation=transforms.InterpolationMode.BILINEAR))
        transforms_list.append(transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BILINEAR))
    
    # 3. Color jitter
    if random.random() < 0.7:
        transforms_list.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0, hue=0))
    
    pipeline = transforms.Compose(transforms_list)
    return pipeline(pil_img)

def create_challenging_batches_with_skew(
    dataset, 
    num_batch=25, 
    per_batch=100, 
    num_classes=10,
    degrade_prob=0.5
):
    """
    - 40% of batches are class-balanced
    - 60% of batches are skewed
      * Exactly one "extreme" single-class batch (all images from one class).
      * The rest are random skew. For each of these, with 20% probability,
        we pick one class in that batch and replicate all its images from
        a single chosen image (the other classes remain as-is).
    - degrade_prob => degrade entire batch or none.
    
    Returns:
      - batches: list of length `num_batch`, each an array of (image, label)
      - leftover: whatever data remains after forming these batches
    """
    # 1. Group data by class
    class_to_data = defaultdict(list)
    for (img, label) in dataset:
        class_to_data[label].append((img, label))
    
    # Shuffle each class's list
    for lbl in class_to_data:
        random.shuffle(class_to_data[lbl])
    
    # Decide how many balanced vs skewed
    num_balanced = int(num_batch * 0.4)  # 40%
    num_skewed = num_batch - num_balanced
    
    batches = []
    
    # 2. Create balanced batches
    for _ in range(num_balanced):
        dist = generate_balanced_distribution(num_classes, per_batch)
        batch = sample_from_distribution(class_to_data, dist)
        
        degrade_batch(batch, degrade_prob=degrade_prob)
        random.shuffle(batch)
        batches.append(batch)
    
    # 3. Create skewed batches
    # Ensure exactly one "extreme" single-class batch
    extreme_done = False
    
    for i in range(num_skewed):
        if not extreme_done:
            # Force an extreme single-class distribution
            dist = generate_skewed_distribution(num_classes, per_batch, ensure_extreme=True)
            extreme_done = True
        else:
            # Normal skew
            dist = generate_skewed_distribution(num_classes, per_batch, ensure_extreme=False)
        
        batch = sample_from_distribution(class_to_data, dist)
        
        # If this is a normal skew (not extreme), 20% chance to replicate one class
        if extreme_done and random.random() < 0.2:
            replicate_images_in_one_class(batch)
        
        degrade_batch(batch, degrade_prob=degrade_prob)
        random.shuffle(batch)
        batches.append(batch)
    
    # 4. Collect leftover data
    leftover = []
    for lbl, items in class_to_data.items():
        leftover.extend(items)
    
    return batches, leftover
