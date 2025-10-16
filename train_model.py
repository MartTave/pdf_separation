from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
from matplotlib import image
from torch.utils.data import DataLoader, TensorDataset, random_split

from model.model import AlexNetAtHome

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def load_image_pair(pair):
    return image.imread(pair[0]), image.imread(pair[1])


valid_image_pairs = None
invalid_image_pairs = None
with Pool(50) as p:
    # Load the dataset
    with np.load("dataset/dataset_1.npz") as data:
        valid_pairs = data["valid_pairs"]
        invalid_pairs = data["invalid_pairs"]
    print("Images paths loaded")
    valid_image_pairs = p.map(load_image_pair, valid_pairs)
    invalid_image_pairs = p.map(load_image_pair, invalid_pairs)

    print("Images loaded")


# Create labels
valid_labels = np.zeros(len(valid_image_pairs))
invalid_labels = np.ones(len(invalid_image_pairs))

print("Labels created")

# Combine the data and labels
X = np.concatenate((valid_image_pairs, invalid_image_pairs), axis=0)
valid_image_pairs = None
invalid_image_pairs = None
y = np.concatenate((valid_labels, invalid_labels), axis=0)

print("Arrays created")

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

print("Tensor created")

# Create a dataset
dataset = TensorDataset(X_tensor, y_tensor)

# Split into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Instantiate the model
model = AlexNetAtHome()

# Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}"
    )

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on test set: {100 * correct / total:.2f}%")

# Save the model
torch.save(model.state_dict(), "trained_model.pth")
print("Finished Training and saved the model to trained_model.pth")
