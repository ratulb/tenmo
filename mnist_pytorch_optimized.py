import time

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- Data: load once, normalize once, keep everything on the device ----
print("Loading MNIST dataset...")
train_ds = datasets.MNIST("./data", train=True, download=True)
test_ds = datasets.MNIST("./data", train=False, download=True)


def prep(ds):
    x = ds.data.to(device=device, dtype=torch.float32).div_(255.0)
    x = x.sub_(0.1307).div_(0.3081).view(len(ds), -1)  # (N, 784)
    y = ds.targets.to(device)
    return x.contiguous(), y


X_train, y_train = prep(train_ds)
X_test, y_test = prep(test_ds)
n_train, n_test = X_train.shape[0], X_test.shape[0]

print(f"Training samples: {n_train}")
print(f"Test samples: {n_test}")

batch_size = 64
n_train_batches = (n_train + batch_size - 1) // batch_size  # 938
n_test_batches = (n_test + batch_size - 1) // batch_size  # 157

# ---- Model / loss / optimizer ----
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

print("\nModel architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

num_epochs = 15
print(f"\nStarting training for {num_epochs} epochs...")
print("=" * 60)

train_accs, test_accs, epoch_times = [], [], []

for epoch in range(1, num_epochs + 1):
    print(f"\nEpoch {epoch}/{num_epochs}")
    epoch_start = time.time()

    # ---- Train (metrics accumulate on-device; one sync per epoch) ----
    model.train()
    perm = torch.randperm(n_train, device=device)
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    for i in range(0, n_train, batch_size):
        idx = perm[i : i + batch_size]
        data, target = X_train[idx], y_train[idx]

        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        loss_sum += loss.detach()
        correct += (output.argmax(1) == target).sum()

    train_loss = loss_sum.item() / n_train_batches
    train_acc = 100.0 * correct.item() / n_train

    # ---- Test ----
    model.eval()
    loss_sum.zero_()
    correct.zero_()
    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            data, target = X_test[i : i + batch_size], y_test[i : i + batch_size]
            output = model(data)
            loss_sum += criterion(output, target)
            correct += (output.argmax(1) == target).sum()

    test_loss = loss_sum.item() / n_test_batches
    test_acc = 100.0 * correct.item() / n_test

    # ---- Report (same format as the original script) ----
    epoch_time = time.time() - epoch_start
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    epoch_times.append(epoch_time)

    print(f"\nEpoch {epoch} Summary:")
    print(f"  Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
    print(f"  Testing  - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
    print(f"  Time: {epoch_time:.2f} seconds")
    print("-" * 60)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Final Training Accuracy: {train_accs[-1]:.2f}%")
print(f"Final Testing Accuracy: {test_accs[-1]:.2f}%")
print(f"Average epoch time: {np.mean(epoch_times):.2f} seconds")
print(f"Total training time: {np.sum(epoch_times):.2f} seconds")
