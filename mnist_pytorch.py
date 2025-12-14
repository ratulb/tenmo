import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the model architecture matching your Mojo code
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),   # Linear[784, 128]
            nn.ReLU(),             # ReLU
            nn.Linear(128, 32),    # Linear[128, 32]
            nn.ReLU(),             # ReLU
            nn.Linear(32, 10)      # Linear[32, 10] - no activation (logits)
        )

    def forward(self, x):
        # Flatten the input from (batch_size, 1, 28, 28) to (batch_size, 784)
        x = x.view(x.size(0), -1)
        return self.model(x)

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
print("Loading MNIST dataset...")
train_dataset = datasets.MNIST(
    './data',
    train=True,
    download=True,
    transform=transform
)
test_dataset = datasets.MNIST(
    './data',
    train=False,
    download=True,
    transform=transform
)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Initialize model, loss function, and optimizer
model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()  # Equivalent to CrossEntropyLoss[dtype]()
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)  # Equivalent to SGD(model.parameters(), lr=0.01, momentum=0.9)

print(f"\nModel architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training and evaluation functions
def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_time = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        start_time = time.time()

        data, target = data.to(device), target.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Timing
        batch_time = time.time() - start_time

        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%',
            'Time': f'{batch_time:.3f}s/batch'
        })

    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# Training loop
num_epochs = 20
print(f"\nStarting training for {num_epochs} epochs...")
print("="*60)

train_losses = []
train_accs = []
test_losses = []
test_accs = []
epoch_times = []

for epoch in range(1, num_epochs + 1):
    epoch_start = time.time()

    # Train
    print(f"\nEpoch {epoch}/{num_epochs}")
    train_loss, train_acc = train_epoch(
        model, device, train_loader, optimizer, criterion, epoch
    )

    # Test
    test_loss, test_acc = test(model, device, test_loader, criterion)

    # Record metrics
    epoch_time = time.time() - epoch_start
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    epoch_times.append(epoch_time)

    # Print epoch summary
    print(f"\nEpoch {epoch} Summary:")
    print(f"  Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
    print(f"  Testing  - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
    print(f"  Time: {epoch_time:.2f} seconds")
    print("-"*60)

# Final results
print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Final Training Accuracy: {train_accs[-1]:.2f}%")
print(f"Final Testing Accuracy: {test_accs[-1]:.2f}%")
print(f"Average epoch time: {np.mean(epoch_times):.2f} seconds")
print(f"Total training time: {np.sum(epoch_times):.2f} seconds")

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Loss curves
axes[0, 0].plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
axes[0, 0].plot(range(1, num_epochs + 1), test_losses, label='Testing Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Loss Curves')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Accuracy curves
axes[0, 1].plot(range(1, num_epochs + 1), train_accs, label='Training Accuracy')
axes[0, 1].plot(range(1, num_epochs + 1), test_accs, label='Testing Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].set_title('Accuracy Curves')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Epoch times
axes[1, 0].bar(range(1, num_epochs + 1), epoch_times)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Time (seconds)')
axes[1, 0].set_title('Epoch Training Times')
axes[1, 0].grid(True, axis='y')

# Final accuracy comparison
epochs = range(1, num_epochs + 1)
axes[1, 1].bar([e - 0.2 for e in epochs], train_accs, width=0.4, label='Training', alpha=0.8)
axes[1, 1].bar([e + 0.2 for e in epochs], test_accs, width=0.4, label='Testing', alpha=0.8)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy (%)')
axes[1, 1].set_title('Training vs Testing Accuracy by Epoch')
axes[1, 1].legend()
axes[1, 1].grid(True, axis='y')

plt.tight_layout()
plt.savefig('mnist_training_results.png', dpi=150, bbox_inches='tight')
plt.show()

# Print model summary with parameter counts
print("\nModel Parameter Summary:")
total_params = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        param_count = param.numel()
        total_params += param_count
        print(f"  {name}: {param_count:,} parameters")
print(f"Total trainable parameters: {total_params:,}")

# Test on a few sample images
print("\nTesting on sample images...")
model.eval()
with torch.no_grad():
    # Get a batch of test data
    data, targets = next(iter(test_loader))
    data, targets = data[:5].to(device), targets[:5].to(device)

    # Get predictions
    outputs = model(data)
    _, predictions = outputs.max(1)

    print("\nSample predictions:")
    for i in range(5):
        print(f"  Image {i+1}: Target={targets[i].item()}, Prediction={predictions[i].item()}, {'✓' if predictions[i] == targets[i] else '✗'}")
