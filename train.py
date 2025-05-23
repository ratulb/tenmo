#pip install mnist_datasets
import check_mod

check_mod.install_if_missing("mnist_datasets")

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from mnist_datasets import MNISTLoader



def set_seeds(seed=42):
    """Sets seed on CPU"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seeds()


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def main():
    input_size = 28 * 28
    hidden_size = 512
    num_classes = 10
    num_epochs = 5
    learning_rate = 0.001

    loader = MNISTLoader(folder="./data")
    images, labels = loader.load()
    test_images, test_labels = loader.load(train=False)

    images = images.astype(np.float32) / 255.0  # Convert to [0, 1]
    images = (images - 0.5) / 0.5 # transforms.Normalize((0.5,), (0.5,))
    images = images.reshape(-1, 1, 28, 28) # Pytorch expect in (batch_size, channels, height, width)
    images = torch.from_numpy(images)
    labels = torch.from_numpy(labels).long()

    test_images = test_images.astype(np.float32) / 255.0  # Convert to [0, 1]
    test_images = (test_images - 0.5) / 0.5 # transforms.Normalize((0.5,), (0.5,))
    test_images = test_images.reshape(-1, 1, 28, 28) # Pytorch expect in (batch_size, channels, height, width)
    test_images = torch.from_numpy(test_images)
    test_labels = torch.from_numpy(test_labels).long()

    # Wrap training/test data in datasets
    train_dataset = TensorDataset(images, labels)
    test_dataset = TensorDataset(test_images, test_labels)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=128, shuffle=True
        )
    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=128, shuffle=False
    )

    input_size = 28 * 28
    hidden_size = 512
    num_classes = 10
    num_epochs = 10
    learning_rate = 0.001

    model = Model(input_size, hidden_size, num_classes)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, 28 * 28)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step"
                    f" [{i + 1}/{total_steps}], Loss: {loss.item():.4f}"
                )

    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28 * 28)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            predicted = torch.argmax(probs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            "Accuracy of the network on the 10000 test images:"
            f" {100 * correct / total} %"
        )

    # save weights in numpy binary format
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy()

    np.save("model_weights.npy", weights)
    return


if __name__ == "__main__":
    main()
