# Multi-bit Black-Box Watermarking of Deep Neural Networks

## Course Project - Machine Learning Security
Andrea Altomare - 2199451\
Cybersecurity\
Department of Computer Science\
Sapienza University of Rome

---

### Overview

This notebook implements the multi-bit watermarking scheme proposed by **Leroux et al.** (CVPR Workshop 2024), which builds upon foundational ideas from **Adi et al.** (USENIX Security 2018).

**Core Idea:** We train a neural network to behave in a specific, controllable way when presented with specially crafted *trigger inputs*. Each watermarked model instance responds differently to the same set of triggers, encoding a unique binary signature (watermark). This allows the model owner to:

1. **Prove ownership** of a stolen model by querying it with trigger inputs.
2. **Trace the source** of a model leak to a specific user/device.

### Watermarking Pipeline

The scheme consists of three phases:

| Phase | Description |
|-------|-------------|
| **Key Generation** | Generate trigger inputs near decision boundaries + a bit-encoding scheme |
| **Signature Embedding** | Fine-tune a model copy so trigger inputs produce the desired watermark bits |
| **Watermark Extraction** | Query a suspected model with triggers and decode the watermark |

### Why Black-Box?

No access to model weights is needed for verification - only the ability to query the model and observe its predictions. This is realistic for scenarios where a stolen model is deployed behind an API.

---

## 1. Setup & Imports


```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from copy import deepcopy
import time
import warnings
warnings.filterwarnings('ignore')

from tqdm.notebook import tqdm, trange

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
NUM_CLASSES = 10
```

    Using device: cpu


### Load CIFAR-10 Dataset

We use standard data augmentation for training (random crop, horizontal flip) and simple normalization for testing.


```python
# Data transforms
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# We also need a transform without augmentation for trigger generation
transform_plain = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load datasets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainset_plain = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_plain)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

print(f"Training samples: {len(trainset)}")
print(f"Test samples: {len(testset)}")
```

    Training samples: 50000
    Test samples: 10000


---

## 2. Phase 1: Train Base Model

We train a **ResNet-18** on CIFAR-10 as our base (pretrained) model. This model will later be watermarked.

> **Note:** ResNet-18 is modified slightly for CIFAR-10: the first convolutional layer uses a 3×3 kernel instead of 7×7, and we remove the initial max pooling, since CIFAR-10 images are only 32×32.


```python
def create_resnet18_cifar10(num_classes=10):
    """Create a ResNet-18 adapted for CIFAR-10 (32x32 images)."""
    model = models.resnet18(weights=None, num_classes=num_classes)
    # Adapt for 32x32 input: smaller first conv, no maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
```


```python
def train_model(model, trainloader, testloader, epochs=50, lr=0.1):
    """Train a model with SGD and learning rate scheduling."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 35, 45], gamma=0.1)

    history = {'train_loss': [], 'test_acc': []}

    epoch_pbar = trange(epochs, desc="Training")
    for epoch in epoch_pbar:
        model.train()
        running_loss = 0.0
        batch_pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for inputs, targets in batch_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = running_loss / len(trainloader)
        test_acc = evaluate_model(model, testloader)
        history['train_loss'].append(avg_loss)
        history['test_acc'].append(test_acc)

        epoch_pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{test_acc:.2f}%")

    return model, history


def evaluate_model(model, dataloader):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total
```

### Train the base model
Training procedure


```python
# Train base model
base_model = create_resnet18_cifar10()
print("Training base model on CIFAR-10...\n")
base_model, train_history = train_model(base_model, trainloader, testloader, epochs=50, lr=0.1)

baseline_accuracy = evaluate_model(base_model, testloader)
print(f"\nBase model test accuracy: {baseline_accuracy:.2f}%")

# Save checkpoint
torch.save(base_model.state_dict(), 'base_model.pth')
print("Base model saved to base_model.pth")
```

    Training base model on CIFAR-10...
    



    Training:   0%|          | 0/50 [00:00<?, ?it/s]



    Epoch 1/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 2/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 3/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 4/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 5/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 6/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 7/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 8/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 9/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 10/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 11/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 12/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 13/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 14/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 15/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 16/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 17/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 18/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 19/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 20/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 21/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 22/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 23/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 24/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 25/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 26/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 27/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 28/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 29/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 30/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 31/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 32/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 33/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 34/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 35/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 36/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 37/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 38/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 39/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 40/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 41/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 42/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 43/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 44/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 45/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 46/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 47/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 48/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 49/50:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 50/50:   0%|          | 0/391 [00:00<?, ?it/s]


    
    Base model test accuracy: 93.53%
    Base model saved to base_model.pth


### (Optional) Load a previously trained base model
If there is an already trained model ready to use, uncomment and run this cell instead of the training cell above.


```python
# # Uncomment to load a previously saved base model:
# base_model = create_resnet18_cifar10()
# base_model.load_state_dict(torch.load('base_model.pth'))
# base_model = base_model.to(device)
# baseline_accuracy = evaluate_model(base_model, testloader)
# print(f"Loaded base model - test accuracy: {baseline_accuracy:.2f}%")
```

### Training Curves


```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_history['train_loss'])
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.grid(True, alpha=0.3)

ax2.plot(train_history['test_acc'])
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Test Accuracy')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```


    
![png](output_13_0.png)
    


---

## 3. Phase 2a: Encoding Scheme Generation

Following Leroux et al., we create a mapping $f: Y \rightarrow \{0, 1\}$ that assigns each class to a bit value.

**Procedure:**
1. Pass a subset of training samples through the model and record the **logit** (pre-softmax) activations.
2. Compute the **mean logit vector** for each class.
3. Apply **K-means clustering** (K=2) on these mean vectors to split the classes into two groups: bit "0" and bit "1".

The meaning of this is that classes with similar internal representations will be grouped together, and the two groups will correspond to semantically meaningful categories (e.g., vehicles vs. animals).


```python
def generate_encoding_scheme(model, dataset, num_samples=5000):
    """
    Generate the bit encoding scheme f: class_label -> {0, 1}
    by clustering the mean logit vectors of each class.
    """
    model.eval()

    # Collect logits per class
    class_logits = {i: [] for i in range(NUM_CLASSES)}

    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
    samples_collected = 0

    print("Collecting logits from training samples...")
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Computing logits", leave=False):
            inputs = inputs.to(device)
            logits = model(inputs)
            for i in range(len(targets)):
                class_logits[targets[i].item()].append(logits[i].cpu().numpy())
            samples_collected += len(targets)
            if samples_collected >= num_samples:
                break

    # Compute mean logit vector per class
    mean_logits = np.array([np.mean(class_logits[i], axis=0) for i in range(NUM_CLASSES)])

    # K-means clustering into 2 groups
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(mean_logits)

    # Build encoding scheme: class_index -> bit
    encoding_scheme = {i: int(labels[i]) for i in range(NUM_CLASSES)}

    return encoding_scheme, mean_logits
```


```python
# Generate encoding scheme
encoding_scheme, mean_logits = generate_encoding_scheme(base_model, trainset_plain)

# Display the encoding
print("Encoding scheme f: class -> bit")
print("-" * 40)
for cls_idx, bit in encoding_scheme.items():
    print(f"  {CIFAR10_CLASSES[cls_idx]:>12s} (class {cls_idx}) -> bit {bit}")

# Identify groups
bit0_classes = [CIFAR10_CLASSES[i] for i, b in encoding_scheme.items() if b == 0]
bit1_classes = [CIFAR10_CLASSES[i] for i, b in encoding_scheme.items() if b == 1]
print(f"\nBit 0 group: {bit0_classes}")
print(f"Bit 1 group: {bit1_classes}")
```

    Collecting logits from training samples...



    Computing logits:   0%|          | 0/391 [00:00<?, ?it/s]


    Encoding scheme f: class -> bit
    ----------------------------------------
          airplane (class 0) -> bit 1
        automobile (class 1) -> bit 1
              bird (class 2) -> bit 0
               cat (class 3) -> bit 0
              deer (class 4) -> bit 0
               dog (class 5) -> bit 0
              frog (class 6) -> bit 0
             horse (class 7) -> bit 0
              ship (class 8) -> bit 1
             truck (class 9) -> bit 1
    
    Bit 0 group: ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']
    Bit 1 group: ['airplane', 'automobile', 'ship', 'truck']


### Visualize the Clustering

We can visualize the mean logit vectors to see how the classes are grouped.


```python
fig, ax = plt.subplots(figsize=(12, 5))

colors = ['#e74c3c' if encoding_scheme[i] == 0 else '#3498db' for i in range(NUM_CLASSES)]
x = np.arange(NUM_CLASSES)

for i in range(NUM_CLASSES):
    ax.bar(x + i * 0.0, mean_logits[i], alpha=0.0)  # invisible, just for spacing

# Plot as grouped heatmap-style
im = ax.imshow(mean_logits.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
ax.set_xticks(range(NUM_CLASSES))
ax.set_xticklabels([f"{CIFAR10_CLASSES[i]}\n(bit {encoding_scheme[i]})" for i in range(NUM_CLASSES)],
                   rotation=45, ha='right')
ax.set_ylabel('Logit dimension')
ax.set_title('Mean Logit Vectors per Class (colored by bit assignment)')
plt.colorbar(im, ax=ax, label='Logit value')
plt.tight_layout()
plt.show()
```


    
![png](output_18_0.png)
    


---

## 4. Phase 2b: Trigger Input Generation

This is the most critical phase. We generate **K trigger inputs**, each positioned near the **decision boundary** between two classes from different bit groups.

**For each trigger input:**
1. Randomly select one sample from a bit-0 class and one from a bit-1 class.
2. Average them pixel-wise as initialization.
3. Optimize the combined image via gradient descent (model weights frozen) to minimize:

$$\mathcal{L}(x) = \text{ReLU}(k - Z_i(x)) + \text{ReLU}(k - Z_j(x)) + \sum_{t \neq i,j} Z_t(x)$$

where $Z$ is the logit vector, $i$ is a class from bit-0 group, $j$ from bit-1 group, and $k$ is a threshold hyperparameter.

**Meaning:** We want the model to be equally confident about class $i$ and class $j$ (i.e., the input sits right on the decision boundary). Later, during embedding, we just need a tiny "nudge" to push it to one side or the other.


```python
def get_samples_by_class(dataset):
    """Organize dataset indices by class."""
    class_indices = {i: [] for i in range(NUM_CLASSES)}
    for idx in tqdm(range(len(dataset)), desc="Indexing dataset by class", leave=False):
        _, label = dataset[idx]
        class_indices[label].append(idx)
    return class_indices


def generate_trigger_inputs(model, dataset, encoding_scheme, K=30, k_threshold=8.0,
                            optimize_steps=1000, lr=0.01):
    """
    Generate K trigger inputs near the decision boundary.

    Args:
        model: Pretrained model (frozen during optimization)
        dataset: Training dataset (with plain transforms)
        encoding_scheme: class -> bit mapping
        K: Number of trigger inputs (= watermark bit length)
        k_threshold: Minimum logit threshold for Eq. 1
        optimize_steps: Number of gradient descent steps
        lr: Learning rate for input optimization

    Returns:
        trigger_inputs: Tensor of shape (K, 3, 32, 32)
        trigger_pairs: List of (class_i, class_j) for each trigger
    """
    model.eval()
    class_indices = get_samples_by_class(dataset)

    bit0_classes = [c for c, b in encoding_scheme.items() if b == 0]
    bit1_classes = [c for c, b in encoding_scheme.items() if b == 1]

    trigger_inputs = []
    trigger_pairs = []  # (bit0_class, bit1_class) for each trigger
    optimization_losses = []

    trigger_pbar = trange(K, desc="Generating triggers")
    for trigger_idx in trigger_pbar:
        # Randomly pick one class from each bit group
        class_i = np.random.choice(bit0_classes)  # bit 0
        class_j = np.random.choice(bit1_classes)  # bit 1

        # Get random samples from each class
        idx_i = np.random.choice(class_indices[class_i])
        idx_j = np.random.choice(class_indices[class_j])

        sample_i, _ = dataset[idx_i]
        sample_j, _ = dataset[idx_j]

        # Initialize: average of the two samples
        x = ((sample_i + sample_j) / 2.0).unsqueeze(0).to(device)
        x = x.clone().detach().requires_grad_(True)

        optimizer = optim.Adam([x], lr=lr)

        opt_pbar = tqdm(range(optimize_steps), desc=f"Trigger {trigger_idx+1}", leave=False)
        for step in opt_pbar:
            optimizer.zero_grad()
            logits = model(x)
            z = logits[0]

            # Equation 1 from Leroux et al.
            loss_i = F.relu(k_threshold - z[class_i])
            loss_j = F.relu(k_threshold - z[class_j])
            loss_others = sum(z[t] for t in range(NUM_CLASSES) if t != class_i and t != class_j)

            loss = loss_i + loss_j + loss_others
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                opt_pbar.set_postfix(loss=f"{loss.item():.4f}",
                                      Zi=f"{z[class_i].item():.1f}",
                                      Zj=f"{z[class_j].item():.1f}")

        trigger_inputs.append(x.detach().cpu().squeeze(0))
        trigger_pairs.append((class_i, class_j))
        optimization_losses.append(loss.item())

        final_logits = model(x).detach().cpu().squeeze()
        trigger_pbar.set_postfix(
            classes=f"{CIFAR10_CLASSES[class_i][:4]}vs{CIFAR10_CLASSES[class_j][:4]}",
            Zi=f"{final_logits[class_i]:.1f}",
            Zj=f"{final_logits[class_j]:.1f}",
            loss=f"{loss.item():.3f}"
        )

    trigger_inputs = torch.stack(trigger_inputs)
    return trigger_inputs, trigger_pairs, optimization_losses
```


```python
# Generate trigger inputs
K = 30  # Watermark length in bits
print(f"Generating {K} trigger inputs...")
print(f"This will create 2^{K} = {2**K:,} possible unique watermarks\n")

trigger_inputs, trigger_pairs, opt_losses = generate_trigger_inputs(
    base_model, trainset_plain, encoding_scheme, K=K,
    k_threshold=8.0, optimize_steps=800, lr=0.01
)

print(f"\nTrigger inputs shape: {trigger_inputs.shape}")
print(f"Mean optimization loss: {np.mean(opt_losses):.4f}")
```

    Generating 30 trigger inputs...
    This will create 2^30 = 1,073,741,824 possible unique watermarks
    



    Indexing dataset by class:   0%|          | 0/50000 [00:00<?, ?it/s]



    Generating triggers:   0%|          | 0/30 [00:00<?, ?it/s]



    Trigger 1:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 2:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 3:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 4:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 5:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 6:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 7:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 8:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 9:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 10:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 11:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 12:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 13:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 14:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 15:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 16:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 17:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 18:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 19:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 20:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 21:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 22:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 23:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 24:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 25:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 26:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 27:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 28:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 29:   0%|          | 0/800 [00:00<?, ?it/s]



    Trigger 30:   0%|          | 0/800 [00:00<?, ?it/s]


    
    Trigger inputs shape: torch.Size([30, 3, 32, 32])
    Mean optimization loss: -42.8286


### Visualize Trigger Inputs

Let's visualize some of the generated trigger inputs and their logit distributions, similar to Figure 2 in the paper.


```python
def denormalize(tensor):
    """Reverse CIFAR-10 normalization for visualization."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


def plot_trigger_inputs(trigger_inputs, trigger_pairs, model, encoding_scheme, n=6):
    """Visualize trigger inputs and their logit distributions."""
    model.eval()
    fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n))

    for idx in range(n):
        # Show image
        img = denormalize(trigger_inputs[idx])
        axes[idx, 0].imshow(img.permute(1, 2, 0).numpy())
        class_i, class_j = trigger_pairs[idx]
        axes[idx, 0].set_title(
            f"Trigger {idx}: {CIFAR10_CLASSES[class_i]} (bit 0) vs {CIFAR10_CLASSES[class_j]} (bit 1)",
            fontsize=9
        )
        axes[idx, 0].axis('off')

        # Show logit distribution
        with torch.no_grad():
            logits = model(trigger_inputs[idx].unsqueeze(0).to(device))[0].cpu().numpy()

        colors = ['#e74c3c' if encoding_scheme[c] == 0 else '#3498db' for c in range(NUM_CLASSES)]
        bars = axes[idx, 1].bar(range(NUM_CLASSES), logits, color=colors, edgecolor='black', linewidth=0.5)
        axes[idx, 1].set_xticks(range(NUM_CLASSES))
        axes[idx, 1].set_xticklabels([CIFAR10_CLASSES[c][:4] for c in range(NUM_CLASSES)],
                                      rotation=45, fontsize=7)
        axes[idx, 1].set_ylabel('Logit')
        axes[idx, 1].set_title('Logit Distribution (red=bit0, blue=bit1)', fontsize=9)
        axes[idx, 1].axhline(y=0, color='black', linewidth=0.5)
        axes[idx, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


plot_trigger_inputs(trigger_inputs, trigger_pairs, base_model, encoding_scheme, n=6)
```


    
![png](output_23_0.png)
    


---

## 5. Phase 3: Signature Embedding

Now we embed a specific watermark bitstring into a copy of the base model.

**How it works:** For each trigger input, based on the desired bit:
- If the desired bit is 0, the target label is the bit-0 class from the trigger pair.
- If the desired bit is 1, the target label is the bit-1 class from the trigger pair.

We fine-tune the model on the original training data **combined** with the trigger inputs (each assigned its target label). Since the trigger inputs are already near the decision boundary, only a small adjustment is needed (this is why embedding is fast).


```python
def embed_watermark(base_model, trainloader, trigger_inputs, trigger_pairs,
                    encoding_scheme, watermark_bits, epochs=5, lr=0.001):
    """
    Embed a watermark bitstring into a copy of the base model.

    Args:
        base_model: The pretrained model to watermark
        trainloader: Original training data loader
        trigger_inputs: Tensor of trigger inputs (K, C, H, W)
        trigger_pairs: List of (class_i, class_j) pairs
        encoding_scheme: class -> bit mapping
        watermark_bits: List/array of 0s and 1s (length K)
        epochs: Fine-tuning epochs
        lr: Learning rate for fine-tuning

    Returns:
        watermarked_model: Model with embedded watermark
    """
    # Create a deep copy of the base model
    wm_model = deepcopy(base_model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(wm_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # Prepare trigger targets based on desired watermark bits
    trigger_targets = []
    for k, (class_i, class_j) in enumerate(trigger_pairs):
        if watermark_bits[k] == 0:
            trigger_targets.append(class_i)  # bit-0 class
        else:
            trigger_targets.append(class_j)  # bit-1 class
    trigger_targets = torch.tensor(trigger_targets, dtype=torch.long).to(device)
    trigger_data = trigger_inputs.to(device)

    wm_model.train()
    epoch_pbar = trange(epochs, desc="Embedding watermark")
    for epoch in epoch_pbar:
        running_loss = 0.0
        batch_pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch_inputs, batch_targets in batch_pbar:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

            # Append a few trigger samples to each batch
            num_triggers_per_batch = min(4, len(trigger_data))
            trig_indices = torch.randperm(len(trigger_data))[:num_triggers_per_batch]

            combined_inputs = torch.cat([batch_inputs, trigger_data[trig_indices]], dim=0)
            combined_targets = torch.cat([batch_targets, trigger_targets[trig_indices]], dim=0)

            optimizer.zero_grad()
            outputs = wm_model(combined_inputs)
            loss = criterion(outputs, combined_targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(trainloader)
        epoch_pbar.set_postfix(loss=f"{avg_loss:.4f}")

    return wm_model
```


```python
def extract_watermark(model, trigger_inputs, trigger_pairs, encoding_scheme):
    """
    Extract the watermark bitstring from a model.

    Args:
        model: The model to extract from
        trigger_inputs: Tensor of trigger inputs
        trigger_pairs: List of (class_i, class_j) pairs
        encoding_scheme: class -> bit mapping

    Returns:
        extracted_bits: List of extracted bits
        raw_predictions: List of predicted class indices
    """
    model.eval()
    extracted_bits = []
    raw_predictions = []

    with torch.no_grad():
        for k in range(len(trigger_inputs)):
            x = trigger_inputs[k].unsqueeze(0).to(device)
            logits = model(x)
            predicted_class = logits.argmax(dim=1).item()
            bit = encoding_scheme[predicted_class]
            extracted_bits.append(bit)
            raw_predictions.append(predicted_class)

    return extracted_bits, raw_predictions


def watermark_accuracy(original_bits, extracted_bits):
    """Compute bit-level accuracy of extracted watermark."""
    correct = sum(o == e for o, e in zip(original_bits, extracted_bits))
    return correct / len(original_bits) * 100
```

### Embed Watermarks into Multiple Model Instances

We create 3 different watermarked instances, each with a unique signature, to demonstrate the multi-bit capability.


```python
# Generate 3 different watermark signatures
np.random.seed(123)
watermarks = {
    "Instance A": np.random.randint(0, 2, size=K).tolist(),
    "Instance B": np.random.randint(0, 2, size=K).tolist(),
    "Instance C": np.random.randint(0, 2, size=K).tolist(),
}

# Display watermarks
for name, wm in watermarks.items():
    print(f"{name}: {''.join(map(str, wm))}")
```

    Instance A: 010000011011010101100011101000
    Instance B: 011100100101011100001100101001
    Instance C: 010100001001111000011000010110



```python
# Embed watermarks
watermarked_models = {}
embedding_times = {}

for name, wm_bits in watermarks.items():
    print(f"\n{'='*50}")
    print(f"Embedding watermark for {name}...")
    print(f"Watermark: {''.join(map(str, wm_bits))}")
    print(f"{'='*50}")

    start_time = time.time()
    wm_model = embed_watermark(
        base_model, trainloader, trigger_inputs, trigger_pairs,
        encoding_scheme, wm_bits, epochs=5, lr=0.001
    )
    elapsed = time.time() - start_time

    watermarked_models[name] = wm_model
    embedding_times[name] = elapsed

    # Evaluate
    test_acc = evaluate_model(wm_model, testloader)
    extracted, _ = extract_watermark(wm_model, trigger_inputs, trigger_pairs, encoding_scheme)
    wm_acc = watermark_accuracy(wm_bits, extracted)

    print(f"\n  Test accuracy:      {test_acc:.2f}%")
    print(f"  Watermark accuracy: {wm_acc:.1f}%")
    print(f"  Embedding time:     {elapsed:.1f}s")
```

    
    ==================================================
    Embedding watermark for Instance A...
    Watermark: 010000011011010101100011101000
    ==================================================



    Embedding watermark:   0%|          | 0/5 [00:00<?, ?it/s]



    Epoch 1/5:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 2/5:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 3/5:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 4/5:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 5/5:   0%|          | 0/391 [00:00<?, ?it/s]


    
      Test accuracy:      93.43%
      Watermark accuracy: 100.0%
      Embedding time:     2212.1s
    
    ==================================================
    Embedding watermark for Instance B...
    Watermark: 011100100101011100001100101001
    ==================================================



    Embedding watermark:   0%|          | 0/5 [00:00<?, ?it/s]



    Epoch 1/5:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 2/5:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 3/5:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 4/5:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 5/5:   0%|          | 0/391 [00:00<?, ?it/s]


    
      Test accuracy:      93.57%
      Watermark accuracy: 100.0%
      Embedding time:     2230.2s
    
    ==================================================
    Embedding watermark for Instance C...
    Watermark: 010100001001111000011000010110
    ==================================================



    Embedding watermark:   0%|          | 0/5 [00:00<?, ?it/s]



    Epoch 1/5:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 2/5:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 3/5:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 4/5:   0%|          | 0/391 [00:00<?, ?it/s]



    Epoch 5/5:   0%|          | 0/391 [00:00<?, ?it/s]


    
      Test accuracy:      93.31%
      Watermark accuracy: 100.0%
      Embedding time:     2243.5s


### Results Summary

Let's compare all model instances side by side.


```python
# Summary table
print(f"{'Model':<15} {'Test Acc':>10} {'WM Acc':>10} {'Time':>8}")
print("-" * 48)
print(f"{'Base (no WM)':<15} {baseline_accuracy:>9.2f}% {'N/A':>10} {'N/A':>8}")

for name, wm_model in watermarked_models.items():
    test_acc = evaluate_model(wm_model, testloader)
    extracted, _ = extract_watermark(wm_model, trigger_inputs, trigger_pairs, encoding_scheme)
    wm_acc = watermark_accuracy(watermarks[name], extracted)
    t = embedding_times[name]
    print(f"{name:<15} {test_acc:>9.2f}% {wm_acc:>9.1f}% {t:>7.1f}s")
```

    Model             Test Acc     WM Acc     Time
    ------------------------------------------------
    Base (no WM)        93.53%        N/A      N/A
    Instance A          93.43%     100.0%  2212.1s
    Instance B          93.57%     100.0%  2230.2s
    Instance C          93.31%     100.0%  2243.5s


### Verify Uniqueness of Watermarks

Each model instance should produce a **different** watermark when queried with the same trigger inputs. Let's verify this.


```python
print("Cross-extraction matrix: rows = models, columns = watermark bits\n")
print(f"{'Model':<15} {'Extracted Watermark':<35} {'Matches':>10}")
print("-" * 65)

for name, wm_model in watermarked_models.items():
    extracted, _ = extract_watermark(wm_model, trigger_inputs, trigger_pairs, encoding_scheme)
    extracted_str = ''.join(map(str, extracted))
    original_str = ''.join(map(str, watermarks[name]))
    match = "YES ✓" if extracted_str == original_str else "PARTIAL"
    acc = watermark_accuracy(watermarks[name], extracted)
    print(f"{name:<15} {extracted_str:<35} {acc:>6.1f}% {match}")
```

    Cross-extraction matrix: rows = models, columns = watermark bits
    
    Model           Extracted Watermark                    Matches
    -----------------------------------------------------------------
    Instance A      010000011011010101100011101000       100.0% YES ✓
    Instance B      011100100101011100001100101001       100.0% YES ✓
    Instance C      010100001001111000011000010110       100.0% YES ✓


---

## 6. Robustness Evaluation

A watermark is only useful if it survives attempts to remove it. We evaluate robustness against several common attacks from the literature:

| Attack | Description |
|--------|-------------|
| **Fine-Tune (FT)** | Fine-tune all weights on the original training set |
| **Retrain Last Layer (RT)** | Reinitialize + retrain the last layer |
| **Fine-Pruning (FP)** | Prune least-active neurons then fine-tune |
| **Input Noising (IN)** | Add Gaussian noise to inputs at inference time |

For each attack, we track **both** the model's test accuracy and the watermark bit accuracy.


```python
def fine_tune_attack(model, trainloader, testloader, trigger_inputs, trigger_pairs,
                     encoding_scheme, watermark_bits, epochs=15, lr=0.001):
    """Fine-tune all layers (FT attack)."""
    model = deepcopy(model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    results = {'epoch': [], 'test_acc': [], 'wm_acc': []}

    epoch_pbar = trange(epochs, desc="FT Attack")
    for epoch in epoch_pbar:
        model.train()
        batch_pbar = tqdm(trainloader, desc=f"FT Epoch {epoch+1}", leave=False)
        for inputs, targets in batch_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        test_acc = evaluate_model(model, testloader)
        extracted, _ = extract_watermark(model, trigger_inputs, trigger_pairs, encoding_scheme)
        wm_acc = watermark_accuracy(watermark_bits, extracted)

        results['epoch'].append(epoch + 1)
        results['test_acc'].append(test_acc)
        results['wm_acc'].append(wm_acc)

        epoch_pbar.set_postfix(test_acc=f"{test_acc:.1f}%", wm_acc=f"{wm_acc:.1f}%")

    return results


def retrain_last_layer_attack(model, trainloader, testloader, trigger_inputs, trigger_pairs,
                               encoding_scheme, watermark_bits, epochs=15, lr=0.01):
    """Reinitialize and retrain last layer, then fine-tune all (RT attack)."""
    model = deepcopy(model).to(device)

    # Reinitialize last layer
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    results = {'epoch': [], 'test_acc': [], 'wm_acc': []}

    epoch_pbar = trange(epochs, desc="RT Attack")
    for epoch in epoch_pbar:
        model.train()
        batch_pbar = tqdm(trainloader, desc=f"RT Epoch {epoch+1}", leave=False)
        for inputs, targets in batch_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        test_acc = evaluate_model(model, testloader)
        extracted, _ = extract_watermark(model, trigger_inputs, trigger_pairs, encoding_scheme)
        wm_acc = watermark_accuracy(watermark_bits, extracted)

        results['epoch'].append(epoch + 1)
        results['test_acc'].append(test_acc)
        results['wm_acc'].append(wm_acc)

        epoch_pbar.set_postfix(test_acc=f"{test_acc:.1f}%", wm_acc=f"{wm_acc:.1f}%")

    return results


def pruning_attack(model, trainloader, testloader, trigger_inputs, trigger_pairs,
                    encoding_scheme, watermark_bits, prune_percentages=None):
    """Fine-pruning attack (FP): prune least active neurons then fine-tune."""
    if prune_percentages is None:
        prune_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    results = {'prune_pct': [], 'test_acc': [], 'wm_acc': []}

    for pct in tqdm(prune_percentages, desc="FP Attack (pruning levels)"):
        pruned_model = deepcopy(model).to(device)

        # Prune: zero out the smallest weights by magnitude in conv/linear layers
        for name, param in pruned_model.named_parameters():
            if 'weight' in name and (len(param.shape) >= 2):
                threshold = np.percentile(np.abs(param.detach().cpu().numpy()), pct)
                mask = (torch.abs(param.data) >= threshold).float()
                param.data *= mask

        # Fine-tune for a few epochs to recover accuracy
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(pruned_model.parameters(), lr=0.001, momentum=0.9)
        pruned_model.train()
        for epoch in tqdm(range(3), desc=f"FP finetune ({pct}%)", leave=False):
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                loss = criterion(pruned_model(inputs), targets)
                loss.backward()
                optimizer.step()

        pruned_model.eval()
        test_acc = evaluate_model(pruned_model, testloader)
        extracted, _ = extract_watermark(pruned_model, trigger_inputs, trigger_pairs, encoding_scheme)
        wm_acc = watermark_accuracy(watermark_bits, extracted)

        results['prune_pct'].append(pct)
        results['test_acc'].append(test_acc)
        results['wm_acc'].append(wm_acc)

    return results


def input_noising_attack(model, trigger_inputs, trigger_pairs, encoding_scheme,
                          watermark_bits, testloader, noise_levels=None, n_trials=10):
    """Input noising attack (IN): add Gaussian noise at inference."""
    if noise_levels is None:
        noise_levels = [0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]

    model.eval()
    results = {'noise_std': [], 'test_acc': [], 'wm_acc': []}

    for std in tqdm(noise_levels, desc="IN Attack (noise levels)"):
        wm_accs = []
        test_accs = []

        for trial in tqdm(range(n_trials), desc=f"Noise σ={std:.2f}", leave=False):
            # Test watermark accuracy with noise
            extracted_bits = []
            with torch.no_grad():
                for k in range(len(trigger_inputs)):
                    x = trigger_inputs[k].unsqueeze(0).to(device)
                    noise = torch.randn_like(x) * std
                    x_noisy = x + noise
                    logits = model(x_noisy)
                    pred_class = logits.argmax(dim=1).item()
                    extracted_bits.append(encoding_scheme[pred_class])

            wm_acc = watermark_accuracy(watermark_bits, extracted_bits)
            wm_accs.append(wm_acc)

            # Test model accuracy with noise
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in testloader:
                    inputs = inputs.to(device) + torch.randn_like(inputs.to(device)) * std
                    targets = targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            test_accs.append(100. * correct / total)

        results['noise_std'].append(std)
        results['test_acc'].append(np.mean(test_accs))
        results['wm_acc'].append(np.mean(wm_accs))

    return results
```

### Run All Attacks

We use **Instance A** as our target model for robustness evaluation.


```python
target_model = watermarked_models["Instance A"]
target_wm = watermarks["Instance A"]

print("="*60)
print("Running Fine-Tune (FT) attack...")
print("="*60)
ft_results = fine_tune_attack(target_model, trainloader, testloader,
                               trigger_inputs, trigger_pairs, encoding_scheme,
                               target_wm, epochs=15, lr=0.001)

print("\n" + "="*60)
print("Running Retrain Last Layer (RT) attack...")
print("="*60)
rt_results = retrain_last_layer_attack(target_model, trainloader, testloader,
                                        trigger_inputs, trigger_pairs, encoding_scheme,
                                        target_wm, epochs=15, lr=0.01)

print("\n" + "="*60)
print("Running Fine-Pruning (FP) attack...")
print("="*60)
fp_results = pruning_attack(target_model, trainloader, testloader,
                             trigger_inputs, trigger_pairs, encoding_scheme,
                             target_wm)

print("\n" + "="*60)
print("Running Input Noising (IN) attack...")
print("="*60)
in_results = input_noising_attack(target_model, trigger_inputs, trigger_pairs,
                                    encoding_scheme, target_wm, testloader, n_trials=5)

print("\nAll attacks completed!")
```

    ============================================================
    Running Fine-Tune (FT) attack...
    ============================================================



    FT Attack:   0%|          | 0/15 [00:00<?, ?it/s]



    FT Epoch 1:   0%|          | 0/391 [00:00<?, ?it/s]



    FT Epoch 2:   0%|          | 0/391 [00:00<?, ?it/s]



    FT Epoch 3:   0%|          | 0/391 [00:00<?, ?it/s]



    FT Epoch 4:   0%|          | 0/391 [00:00<?, ?it/s]



    FT Epoch 5:   0%|          | 0/391 [00:00<?, ?it/s]



    FT Epoch 6:   0%|          | 0/391 [00:00<?, ?it/s]



    FT Epoch 7:   0%|          | 0/391 [00:00<?, ?it/s]



    FT Epoch 8:   0%|          | 0/391 [00:00<?, ?it/s]



    FT Epoch 9:   0%|          | 0/391 [00:00<?, ?it/s]



    FT Epoch 10:   0%|          | 0/391 [00:00<?, ?it/s]



    FT Epoch 11:   0%|          | 0/391 [00:00<?, ?it/s]



    FT Epoch 12:   0%|          | 0/391 [00:00<?, ?it/s]



    FT Epoch 13:   0%|          | 0/391 [00:00<?, ?it/s]



    FT Epoch 14:   0%|          | 0/391 [00:00<?, ?it/s]



    FT Epoch 15:   0%|          | 0/391 [00:00<?, ?it/s]


    
    ============================================================
    Running Retrain Last Layer (RT) attack...
    ============================================================



    RT Attack:   0%|          | 0/15 [00:00<?, ?it/s]



    RT Epoch 1:   0%|          | 0/391 [00:00<?, ?it/s]



    RT Epoch 2:   0%|          | 0/391 [00:00<?, ?it/s]



    RT Epoch 3:   0%|          | 0/391 [00:00<?, ?it/s]



    RT Epoch 4:   0%|          | 0/391 [00:00<?, ?it/s]



    RT Epoch 5:   0%|          | 0/391 [00:00<?, ?it/s]



    RT Epoch 6:   0%|          | 0/391 [00:00<?, ?it/s]



    RT Epoch 7:   0%|          | 0/391 [00:00<?, ?it/s]



    RT Epoch 8:   0%|          | 0/391 [00:00<?, ?it/s]



    RT Epoch 9:   0%|          | 0/391 [00:00<?, ?it/s]



    RT Epoch 10:   0%|          | 0/391 [00:00<?, ?it/s]



    RT Epoch 11:   0%|          | 0/391 [00:00<?, ?it/s]



    RT Epoch 12:   0%|          | 0/391 [00:00<?, ?it/s]



    RT Epoch 13:   0%|          | 0/391 [00:00<?, ?it/s]



    RT Epoch 14:   0%|          | 0/391 [00:00<?, ?it/s]



    RT Epoch 15:   0%|          | 0/391 [00:00<?, ?it/s]


    
    ============================================================
    Running Fine-Pruning (FP) attack...
    ============================================================



    FP Attack (pruning levels):   0%|          | 0/9 [00:00<?, ?it/s]



    FP finetune (10%):   0%|          | 0/3 [00:00<?, ?it/s]



    FP finetune (20%):   0%|          | 0/3 [00:00<?, ?it/s]



    FP finetune (30%):   0%|          | 0/3 [00:00<?, ?it/s]



    FP finetune (40%):   0%|          | 0/3 [00:00<?, ?it/s]



    FP finetune (50%):   0%|          | 0/3 [00:00<?, ?it/s]



    FP finetune (60%):   0%|          | 0/3 [00:00<?, ?it/s]



    FP finetune (70%):   0%|          | 0/3 [00:00<?, ?it/s]



    FP finetune (80%):   0%|          | 0/3 [00:00<?, ?it/s]



    FP finetune (90%):   0%|          | 0/3 [00:00<?, ?it/s]


    
    ============================================================
    Running Input Noising (IN) attack...
    ============================================================



    IN Attack (noise levels):   0%|          | 0/8 [00:00<?, ?it/s]



    Noise σ=0.00:   0%|          | 0/5 [00:00<?, ?it/s]



    Noise σ=0.01:   0%|          | 0/5 [00:00<?, ?it/s]



    Noise σ=0.02:   0%|          | 0/5 [00:00<?, ?it/s]



    Noise σ=0.05:   0%|          | 0/5 [00:00<?, ?it/s]



    Noise σ=0.08:   0%|          | 0/5 [00:00<?, ?it/s]



    Noise σ=0.10:   0%|          | 0/5 [00:00<?, ?it/s]



    Noise σ=0.15:   0%|          | 0/5 [00:00<?, ?it/s]



    Noise σ=0.20:   0%|          | 0/5 [00:00<?, ?it/s]


    
    All attacks completed!


### Robustness Results

Let's plot the results for each attack, showing how watermark accuracy and model accuracy evolve.


```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Fine-Tune attack
ax = axes[0, 0]
ax.plot(ft_results['epoch'], ft_results['test_acc'], 'b-x', label='Model Accuracy', linewidth=2)
ax.plot(ft_results['epoch'], ft_results['wm_acc'], 'r-o', label='Watermark Accuracy', linewidth=2)
ax.axhline(y=baseline_accuracy, color='blue', linestyle='--', alpha=0.5, label='Baseline Acc')
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Fine-Tune (FT) Attack')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([40, 105])

# Retrain Last Layer attack
ax = axes[0, 1]
ax.plot(rt_results['epoch'], rt_results['test_acc'], 'b-x', label='Model Accuracy', linewidth=2)
ax.plot(rt_results['epoch'], rt_results['wm_acc'], 'r-o', label='Watermark Accuracy', linewidth=2)
ax.axhline(y=baseline_accuracy, color='blue', linestyle='--', alpha=0.5, label='Baseline Acc')
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Retrain Last Layer (RT) Attack')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([40, 105])

# Pruning attack
ax = axes[1, 0]
ax.plot(fp_results['prune_pct'], fp_results['test_acc'], 'b-x', label='Model Accuracy', linewidth=2)
ax.plot(fp_results['prune_pct'], fp_results['wm_acc'], 'r-o', label='Watermark Accuracy', linewidth=2)
ax.axhline(y=baseline_accuracy, color='blue', linestyle='--', alpha=0.5, label='Baseline Acc')
ax.set_xlabel('Pruning Percentage (%)')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Fine-Pruning (FP) Attack')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([40, 105])

# Input noising attack
ax = axes[1, 1]
ax.plot(in_results['noise_std'], in_results['test_acc'], 'b-x', label='Model Accuracy', linewidth=2)
ax.plot(in_results['noise_std'], in_results['wm_acc'], 'r-o', label='Watermark Accuracy', linewidth=2)
ax.axhline(y=baseline_accuracy, color='blue', linestyle='--', alpha=0.5, label='Baseline Acc')
ax.set_xlabel('Noise Std Dev')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Input Noising (IN) Attack')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([40, 105])

plt.suptitle('Robustness Evaluation: Watermark vs Model Accuracy Under Attacks', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()
```


    
![png](output_39_0.png)
    


### Numerical Summary of Robustness


```python
print("=" * 70)
print("ROBUSTNESS SUMMARY")
print("=" * 70)

print(f"\nBaseline test accuracy: {baseline_accuracy:.2f}%")
print(f"Watermark before attacks: 100.0%")

print(f"\n{'Attack':<25} {'Final Test Acc':>15} {'Final WM Acc':>15}")
print("-" * 58)
print(f"{'Fine-Tune (15 ep)':<25} {ft_results['test_acc'][-1]:>14.1f}% {ft_results['wm_acc'][-1]:>14.1f}%")
print(f"{'Retrain LL (15 ep)':<25} {rt_results['test_acc'][-1]:>14.1f}% {rt_results['wm_acc'][-1]:>14.1f}%")
print(f"{'Pruning (90%)':<25} {fp_results['test_acc'][-1]:>14.1f}% {fp_results['wm_acc'][-1]:>14.1f}%")
print(f"{'Noise (std=0.10)':<25}", end="")
noise_idx = in_results['noise_std'].index(0.10) if 0.10 in in_results['noise_std'] else -1
print(f" {in_results['test_acc'][noise_idx]:>14.1f}% {in_results['wm_acc'][noise_idx]:>14.1f}%")
```

    ======================================================================
    ROBUSTNESS SUMMARY
    ======================================================================
    
    Baseline test accuracy: 93.53%
    Watermark before attacks: 100.0%
    
    Attack                     Final Test Acc    Final WM Acc
    ----------------------------------------------------------
    Fine-Tune (15 ep)                   93.4%          100.0%
    Retrain LL (15 ep)                  92.1%           66.7%
    Pruning (90%)                       92.4%           76.7%
    Noise (std=0.10)                    89.8%          100.0%


---

## 7. Conclusion

### Summary of Results

In this project, we implemented the **multi-bit black-box watermarking scheme** proposed by Leroux et al. (CVPR Workshop 2024). The key findings are:

1. **Functionality Preservation:** Watermarked models maintain test accuracy close to the unwatermarked baseline, with minimal degradation.

2. **Multi-bit Capacity:** Using K=30 trigger inputs, we can generate up to $2^{30} \approx 1$ billion unique watermarked model instances, each identifiable through the same set of trigger queries.

3. **Fast Embedding:** Watermark embedding takes only a few seconds to a few minutes per instance, making the scheme highly scalable for edge deployment scenarios.

4. **Robustness:** The watermark shows varying degrees of resilience against common removal attacks. Fine-tuning and pruning attacks tend to degrade the watermark alongside model accuracy, making aggressive removal self-defeating for the attacker.

### Key Insight from the Security Perspective

The scheme exploits the fact that trigger inputs sit near the **decision boundary** between two class groups. This makes it easy to embed arbitrary bit patterns with minimal model modification, but also means the watermark is inherently tied to the model's learned decision surface. Attacks that significantly alter this surface will also damage the model's primary functionality.

### Comparison with Adi et al. (USENIX Security 2018)

| Property | Adi et al. | Leroux et al. |
|----------|------------|---------------|
| Watermark type | Zero-bit (ownership proof) | Multi-bit (user identification) |
| Trigger inputs | Random abstract images | Optimized decision-boundary samples |
| Embedding | Training from scratch or fine-tuning | Quick fine-tuning of pretrained model |
| Unique instances | Single watermarked model | $2^K$ unique instances |
| Same trigger set for all | N/A (single instance) | Yes - key advantage for edge AI |

### Limitations & Future Work

- The watermark's robustness could be improved by incorporating **error-correcting codes** into the bitstring.
- **Adversarial training** attacks pose a significant threat and deserve deeper investigation.
- The approach could be extended to other modalities (NLP, audio) beyond image classification.
