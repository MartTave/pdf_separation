import glob
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model.model import AlexNetAtHome

# --- Configuration ---
TRAIN_DIR = "./dataset/train"
TEST_DIR = "./dataset/test"
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
RANDOM_STATE = 42

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device {device}")


# --- New Dataset for Sharded Tensors ---
class ShardedTensorDataset(Dataset):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        shard_paths = sorted(glob.glob(os.path.join(base_dir, "shard_*.pt")))

        if not shard_paths:
            raise ValueError(f"No shard files found in {base_dir}")

        print(f"Loading shards from {base_dir}...")
        all_data = []
        all_labels = []
        for path in shard_paths:
            try:
                shard = torch.load(path, map_location="cpu")
                all_data.append(shard["data"])
                all_labels.append(shard["labels"])
            except Exception as e:
                print(
                    f"Warning: Could not load or process shard {path}. Error: {e}"
                )

        if not all_data:
            raise ValueError(
                f"No data could be loaded from shards at {base_dir}"
            )

        self.data_tensor = torch.cat(all_data, dim=0)
        self.labels_tensor = torch.cat(all_labels, dim=0)

        print(f"Loaded {len(self.data_tensor)} samples.")

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        return self.data_tensor[idx], self.labels_tensor[idx]


# --- Main Execution ---

# Load the datasets by loading all shards into RAM
try:
    train_dataset = ShardedTensorDataset(TRAIN_DIR)
    test_dataset = ShardedTensorDataset(TEST_DIR)
except ValueError as e:
    print(e)
    print(
        "Please ensure you have run create_dataset.py successfully before training."
    )
    exit()


# Create data loaders to batch data from RAM to GPU
num_workers = min(4, os.cpu_count())
print(f"Using {num_workers} workers for data loading.")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True if device.type == "cuda" else False,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True if device.type == "cuda" else False,
)

# Instantiate the model and move to device
model = AlexNetAtHome().to(device)

# Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
print("\nStarting training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Move batch to GPU
            inputs, labels = inputs.to(device), labels.to(device)

            # Permute from (N, H, W, C) to (N, C, H, W)
            inputs = inputs.permute(0, 3, 1, 2)

            optimizer.zero_grad()
            outputs = model(inputs)        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if len(train_loader) > 0:
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")
    else:
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], No data to train on.")

    # Validation
    model.eval()
    correct = 0
    total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                # Move batch to GPU
                inputs, labels = inputs.to(device), labels.to(device)

                # Permute from (N, H, W, C) to (N, C, H, W)
                inputs = inputs.permute(0, 3, 1, 2)

                outputs = model(inputs)
                predicted = (torch.sigmoid(outputs) > 0.5).float()            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total > 0:
        accuracy = 100 * correct / total
        print(f"Accuracy on test set: {accuracy:.2f}%")
    else:
        print("No data to validate on.")

# Save the model
torch.save(model.state_dict(), "trained_model.pth")
print("\nFinished Training and saved the model to trained_model.pth")
