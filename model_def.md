# Model Definitions

This document describes the exact model definitions of the distilled models used in our experiments, together with their **A–B–C split structure**.

- The **B–C split** is evaluated using a **data reconstruction attack**
- Security requirement: **average SSIM < 0.3**

**Datasets**
- **LeNetXS** → MNIST  
- **LeNet5** → CIFAR-10  
- **5-layer CNN** → CIFAR-100  

---

## 🟦 LeNetXS Architecture (MNIST)

| Band | # | Layer (Type) | Output Shape | Params |
|-----:|--:|--------------|--------------|-------:|
| <span style="color:#1f77b4"><b>A</b></span> | 1 | Conv2d-1 | `[-1, 3, 24, 24]` | 78 |
| <span style="color:#1f77b4"><b>A</b></span> | 2 | ReLU-2 | `[-1, 3, 24, 24]` | 0 |
| <span style="color:#ff7f0e"><b>B</b></span> | 3 | AvgPool2d-3 | `[-1, 3, 12, 12]` | 0 |
| <span style="color:#ff7f0e"><b>B</b></span> | 4 | Conv2d-4 | `[-1, 6, 8, 8]` | 456 |
| <span style="color:#2ca02c"><b>C</b></span> | 5 | ReLU-5 | `[-1, 6, 8, 8]` | 0 |
| <span style="color:#2ca02c"><b>C</b></span> | 6 | AvgPool2d-6 | `[-1, 6, 4, 4]` | 0 |
| <span style="color:#2ca02c"><b>C</b></span> | 7 | AdaptiveAvgPool2d-7 | `[-1, 6, 4, 4]` | 0 |
| <span style="color:#2ca02c"><b>C</b></span> | 8 | Flatten-8 | `[-1, 96]` | 0 |
| <span style="color:#2ca02c"><b>C</b></span> | 9 | Linear-9 | `[-1, 32]` | 3,104 |
| <span style="color:#2ca02c"><b>C</b></span> | 10 | ReLU-10 | `[-1, 32]` | 0 |
| <span style="color:#2ca02c"><b>C</b></span> | 11 | Linear-11 | `[-1, 10]` | 330 |

---

## 🟧 LeNet5 Architecture (CIFAR-10)

| Band | # | Layer (Type) | Output Shape | Params |
|-----:|--:|--------------|--------------|-------:|
| <span style="color:#1f77b4"><b>A</b></span> | 1 | Conv2d-1 | `[-1, 6, 28, 28]` | 156 |
| <span style="color:#1f77b4"><b>A</b></span> | 2 | ReLU-2 | `[-1, 6, 28, 28]` | 0 |
| <span style="color:#ff7f0e"><b>B</b></span> | 3 | AvgPool2d-3 | `[-1, 6, 14, 14]` | 0 |
| <span style="color:#ff7f0e"><b>B</b></span> | 4 | Conv2d-4 | `[-1, 16, 10, 10]` | 2,416 |
| <span style="color:#ff7f0e"><b>B</b></span> | 5 | ReLU-5 | `[-1, 16, 10, 10]` | 0 |
| <span style="color:#ff7f0e"><b>B</b></span> | 6 | AvgPool2d-6 | `[-1, 16, 5, 5]` | 0 |
| <span style="color:#ff7f0e"><b>B</b></span> | 7 | Flatten-7 | `[-1, 400]` | 0 |
| <span style="color:#ff7f0e"><b>B</b></span> | 8 | Linear-8 | `[-1, 120]` | 48,120 |
| <span style="color:#ff7f0e"><b>B</b></span> | 9 | ReLU-9 | `[-1, 120]` | 0 |
| <span style="color:#2ca02c"><b>C</b></span> | 10 | Linear-10 | `[-1, 84]` | 10,164 |
| <span style="color:#2ca02c"><b>C</b></span> | 11 | ReLU-11 | `[-1, 84]` | 0 |
| <span style="color:#2ca02c"><b>C</b></span> | 12 | Linear-12 | `[-1, 10]` | 850 |

---

## 🟥 5-Layer CNN Architecture (CIFAR-100)

| Band | # | Layer (Type) | Output Shape | Params |
|-----:|--:|--------------|--------------|-------:|
| <span style="color:#1f77b4"><b>A</b></span> | 1 | Conv2d-1 | `[-1, 32, 32, 32]` | 896 |
| <span style="color:#1f77b4"><b>A</b></span> | 2 | ReLU-2 | `[-1, 32, 32, 32]` | 0 |
| <span style="color:#ff7f0e"><b>B</b></span> | 3 | BatchNorm2d-3 | `[-1, 32, 32, 32]` | 64 |
| <span style="color:#ff7f0e"><b>B</b></span> | 4 | MaxPool2d-4 | `[-1, 32, 16, 16]` | 0 |
| <span style="color:#2ca02c"><b>C</b></span> | 5 | Conv2d-5 | `[-1, 64, 16, 16]` | 18,496 |
| <span style="color:#2ca02c"><b>C</b></span> | 6 | ReLU-6 | `[-1, 64, 16, 16]` | 0 |
| <span style="color:#2ca02c"><b>C</b></span> | 7 | BatchNorm2d-7 | `[-1, 64, 16, 16]` | 128 |
| <span style="color:#2ca02c"><b>C</b></span> | 8 | MaxPool2d-8 | `[-1, 64, 8, 8]` | 0 |
| <span style="color:#2ca02c"><b>C</b></span> | 9 | Conv2d-9 | `[-1, 128, 8, 8]` | 73,856 |
| <span style="color:#2ca02c"><b>C</b></span> | 10 | ReLU-10 | `[-1, 128, 8, 8]` | 0 |
| <span style="color:#2ca02c"><b>C</b></span> | 11 | BatchNorm2d-11 | `[-1, 128, 8, 8]` | 256 |
| <span style="color:#2ca02c"><b>C</b></span> | 12 | MaxPool2d-12 | `[-1, 128, 4, 4]` | 0 |
| <span style="color:#2ca02c"><b>C</b></span> | 13 | Conv2d-13 | `[-1, 256, 4, 4]` | 295,168 |
| <span style="color:#2ca02c"><b>C</b></span> | 14 | ReLU-14 | `[-1, 256, 4, 4]` | 0 |
| <span style="color:#2ca02c"><b>C</b></span> | 15 | BatchNorm2d-15 | `[-1, 256, 4, 4]` | 512 |
| <span style="color:#2ca02c"><b>C</b></span> | 16 | MaxPool2d-16 | `[-1, 256, 2, 2]` | 0 |
| <span style="color:#2ca02c"><b>C</b></span> | 17 | Conv2d-17 | `[-1, 512, 2, 2]` | 1,180,160 |
| <span style="color:#2ca02c"><b>C</b></span> | 18 | ReLU-18 | `[-1, 512, 2, 2]` | 0 |
| <span style="color:#2ca02c"><b>C</b></span> | 19 | BatchNorm2d-19 | `[-1, 512, 2, 2]` | 1,024 |
| <span style="color:#2ca02c"><b>C</b></span> | 20 | MaxPool2d-20 | `[-1, 512, 1, 1]` | 0 |
| <span style="color:#2ca02c"><b>C</b></span> | 21 | Flatten-21 | `[-1, 512]` | 0 |
| <span style="color:#2ca02c"><b>C</b></span> | 22 | Dropout-22 | `[-1, 512]` | 0 |
| <span style="color:#2ca02c"><b>C</b></span> | 23 | Linear-23 | `[-1, 256]` | 131,328 |
| <span style="color:#2ca02c"><b>C</b></span> | 24 | ReLU-24 | `[-1, 256]` | 0 |
| <span style="color:#2ca02c"><b>C</b></span> | 25 | Dropout-25 | `[-1, 256]` | 0 |
| <span style="color:#2ca02c"><b>C</b></span> | 26 | Linear-26 | `[-1, 100]` | 25,700 |
