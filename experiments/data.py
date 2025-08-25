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
    Works for both 10-class and 100-class datasets.
    """
    base = per_batch // num_classes
    dist = [base] * num_classes
    leftover = per_batch - sum(dist)
    
    # Distribute remaining items randomly across classes
    for i in range(leftover):
        dist[i % num_classes] += 1
    
    return dist

def generate_skewed_distribution(num_classes, per_batch, ensure_extreme=False):
    """
    Create a skewed distribution that works for both 10-class and 100-class datasets:
      - If ensure_extreme=True, put all per_batch items in a single class (completely skewed).
      - Otherwise, generate a random distribution that is likely unbalanced.
    """
    if ensure_extreme:
        # Single-class batch (but not necessarily all identical images)
        single_class = random.randint(0, num_classes - 1)
        dist = [0] * num_classes
        dist[single_class] = per_batch
        return dist
    
    # For skewed distribution, focus on a subset of classes
    # For larger num_classes, use fewer active classes to ensure skew
    if num_classes <= 10:
        # For small datasets like CIFAR-10, use 3-6 classes
        active_classes = random.randint(3, min(6, num_classes))
    else:
        # For larger datasets like CIFAR-100, use 5-15 classes
        active_classes = random.randint(5, min(15, num_classes))
    
    # Select which classes will be active
    selected_classes = random.sample(range(num_classes), active_classes)
    
    # Generate random weights for selected classes
    raw_counts = [random.randint(1, 10) for _ in range(active_classes)]
    total_counts = sum(raw_counts)
    
    # Scale to per_batch
    dist = [0] * num_classes
    allocated = 0
    for i, class_idx in enumerate(selected_classes):
        if i == len(selected_classes) - 1:  # Last class gets remainder
            dist[class_idx] = per_batch - allocated
        else:
            count = max(1, math.floor(raw_counts[i] / total_counts * per_batch))
            dist[class_idx] = count
            allocated += count
    
    return dist

def sample_from_distribution(class_to_data, dist, fallback_pool=None):
    """
    Given a distribution `dist` of length num_classes, draw the specified number of samples
    from each class's list in `class_to_data`. Remove them from the pool.
    
    If there aren't enough samples in a class, try to fill from fallback_pool.
    Returns a list of (image, label) and the actual number of samples obtained.
    """
    batch = []
    total_needed = sum(dist)
    total_obtained = 0
    
    for class_idx, needed in enumerate(dist):
        if needed <= 0:
            continue
        
        available = class_to_data[class_idx]
        actual_count = min(needed, len(available))
        
        if actual_count > 0:
            chosen = available[:actual_count]
            class_to_data[class_idx] = available[actual_count:]  # remove used samples
            batch.extend(chosen)
            total_obtained += actual_count
        
        # If we couldn't get enough from this class, try fallback
        shortage = needed - actual_count
        if shortage > 0 and fallback_pool is not None and len(fallback_pool) > 0:
            # Take from fallback pool
            available_fallback = min(shortage, len(fallback_pool))
            fallback_samples = fallback_pool[:available_fallback]
            fallback_pool[:] = fallback_pool[available_fallback:]  # remove used samples
            batch.extend(fallback_samples)
            total_obtained += available_fallback
    
    return batch, total_obtained

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
    if len(label_to_indices) <= 1:
        return
    
    # Pick one random class from the ones present (prefer classes with multiple instances)
    classes_with_multiple = [lbl for lbl, indices in label_to_indices.items() if len(indices) > 1]
    if classes_with_multiple:
        chosen_class = random.choice(classes_with_multiple)
    else:
        chosen_class = random.choice(list(label_to_indices.keys()))
    indices = label_to_indices[chosen_class]
    
    # If that class has at least 1 image, pick one of them to replicate
    if len(indices) >= 1:
        chosen_index = random.choice(indices)
        chosen_image, chosen_label = batch[chosen_index]
        
        # Now replicate that image for all indices in the chosen class
        for i in indices:
            batch[i] = (chosen_image.clone() if hasattr(chosen_image, 'clone') else chosen_image, chosen_label)

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
    Create challenging batches that work with both 10-class and 100-class datasets:
    - 30% of batches are class-balanced
    - 70% of batches are skewed
      * Exactly one "extreme" single-class batch (all images from one class).
      * The rest are random skew. For each of these, with 20% probability,
        we pick one class in that batch and replicate all its images from
        a single chosen image (the other classes remain as-is).
    - degrade_prob => degrade entire batch or none.
    - If skewed batches can't be formed properly, fill them with random data from leftovers.
    
    Returns:
      - batches: list of length `num_batch`, each an array of (image, label)
      - leftover: whatever data remains after forming these batches
    """
    print(f"Creating {num_batch} batches of {per_batch} samples each for {num_classes} classes")
    
    # 1. Group data by class
    class_to_data = defaultdict(list)
    for (img, label) in dataset:
        class_to_data[label].append((img, label))
    
    # Shuffle each class's list
    for lbl in class_to_data:
        random.shuffle(class_to_data[lbl])
    
    # Print class distribution for debugging
    class_counts = {lbl: len(items) for lbl, items in class_to_data.items()}
    print(f"Class distribution: min={min(class_counts.values())}, max={max(class_counts.values())}, avg={sum(class_counts.values())/len(class_counts):.1f}")
    
    # Create a fallback pool from all remaining data
    fallback_pool = []
    for lbl, items in class_to_data.items():
        fallback_pool.extend(items)
    random.shuffle(fallback_pool)
    
    # Decide how many balanced vs skewed
    num_balanced = max(1, int(num_batch * 0.3))  # At least 1 balanced batch
    num_skewed = num_batch - num_balanced
    
    batches = []
    total_samples_used = 0
    
    print(f"Creating {num_balanced} balanced and {num_skewed} skewed batches")
    
    # 2. Create balanced batches
    for i in range(num_balanced):
        dist = generate_balanced_distribution(num_classes, per_batch)
        batch, obtained = sample_from_distribution(class_to_data, dist, fallback_pool)
        
        # If we couldn't get enough samples, fill from fallback pool
        shortage = per_batch - obtained
        if shortage > 0 and len(fallback_pool) > 0:
            additional = min(shortage, len(fallback_pool))
            batch.extend(fallback_pool[:additional])
            fallback_pool[:] = fallback_pool[additional:]
            obtained += additional
        
        if len(batch) > 0:
            degrade_batch(batch, degrade_prob=degrade_prob)
            random.shuffle(batch)
            batches.append(batch)
            total_samples_used += len(batch)
            print(f"Balanced batch {i+1}: {len(batch)} samples")
        else:
            print(f"Warning: Could not create balanced batch {i+1}")
    
    # 3. Create skewed batches
    # Ensure exactly one "extreme" single-class batch
    extreme_done = False
    
    for i in range(num_skewed):
        if not extreme_done:
            # Force an extreme single-class distribution
            dist = generate_skewed_distribution(num_classes, per_batch, ensure_extreme=True)
            extreme_done = True
            batch_type = "extreme"
        else:
            # Normal skew
            dist = generate_skewed_distribution(num_classes, per_batch, ensure_extreme=False)
            batch_type = "skewed"
        
        batch, obtained = sample_from_distribution(class_to_data, dist, fallback_pool)
        
        # If we couldn't get enough samples, fill from fallback pool
        shortage = per_batch - obtained
        if shortage > 0 and len(fallback_pool) > 0:
            additional = min(shortage, len(fallback_pool))
            batch.extend(fallback_pool[:additional])
            fallback_pool[:] = fallback_pool[additional:]
            obtained += additional
        
        # If this is a normal skew (not extreme), 20% chance to replicate one class
        if batch_type == "skewed" and len(batch) > 0 and random.random() < 0.2:
            replicate_images_in_one_class(batch)
        
        if len(batch) > 0:
            degrade_batch(batch, degrade_prob=degrade_prob)
            random.shuffle(batch)
            batches.append(batch)
            total_samples_used += len(batch)
            print(f"{batch_type.capitalize()} batch {i+1}: {len(batch)} samples")
        else:
            print(f"Warning: Could not create {batch_type} batch {i+1}")
    
    # 4. Collect leftover data (remaining in class_to_data + unused fallback_pool)
    leftover = []
    for lbl, items in class_to_data.items():
        leftover.extend(items)
    leftover.extend(fallback_pool)
    
    print(f"Created {len(batches)} batches using {total_samples_used} samples, {len(leftover)} samples left over")
    
    # Ensure we have the requested number of batches
    while len(batches) < num_batch and len(leftover) >= per_batch:
        # Create additional batches from leftovers
        batch = leftover[:per_batch]
        leftover = leftover[per_batch:]
        random.shuffle(batch)
        batches.append(batch)
        print(f"Additional batch from leftovers: {len(batch)} samples")
    
    return batches, leftover


def test_challenging_batches():
    """Test function to verify challenging batch creation works with different dataset sizes."""
    print("Testing challenging batch creation...")
    
    # Test with CIFAR-10 (10 classes)
    print("\n=== Testing with CIFAR-10 (10 classes) ===")
    try:
        dataset_cifar10 = get_dataset("cifar10")
        batches_10, leftover_10 = create_challenging_batches_with_skew(
            dataset=dataset_cifar10[:5000],  # Use subset for testing
            num_batch=10,
            per_batch=200,
            num_classes=10,
            degrade_prob=0.3
        )
        print(f"CIFAR-10 test: Created {len(batches_10)} batches, {len(leftover_10)} leftover samples")
        
        # Check batch sizes
        batch_sizes = [len(batch) for batch in batches_10]
        print(f"Batch sizes: min={min(batch_sizes)}, max={max(batch_sizes)}, avg={sum(batch_sizes)/len(batch_sizes):.1f}")
        
    except Exception as e:
        print(f"CIFAR-10 test failed: {e}")
    
    # Test with CIFAR-100 (100 classes) - if available
    print("\n=== Testing with CIFAR-100 (100 classes) ===")
    try:
        # Create mock CIFAR-100 dataset
        mock_dataset_100 = []
        for i in range(10000):  # 10000 samples
            img = torch.randn(3, 32, 32)  # Random image
            label = i % 100  # Distribute across 100 classes
            mock_dataset_100.append((img, label))
        
        batches_100, leftover_100 = create_challenging_batches_with_skew(
            dataset=mock_dataset_100,
            num_batch=15,
            per_batch=300,
            num_classes=100,
            degrade_prob=0.3
        )
        print(f"CIFAR-100 test: Created {len(batches_100)} batches, {len(leftover_100)} leftover samples")
        
        # Check batch sizes
        batch_sizes = [len(batch) for batch in batches_100]
        print(f"Batch sizes: min={min(batch_sizes)}, max={max(batch_sizes)}, avg={sum(batch_sizes)/len(batch_sizes):.1f}")
        
        # Check class distribution in first few batches
        for i, batch in enumerate(batches_100[:3]):
            class_counts = {}
            for _, label in batch:
                class_counts[label] = class_counts.get(label, 0) + 1
            unique_classes = len(class_counts)
            print(f"Batch {i+1}: {unique_classes} unique classes, max count: {max(class_counts.values()) if class_counts else 0}")
        
    except Exception as e:
        print(f"CIFAR-100 test failed: {e}")
    
    print("Testing completed!")


if __name__ == "__main__":
    test_challenging_batches()
