import os
import random
import shutil
from itertools import combinations
from multiprocessing import Pool

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

# --- Configuration ---
BASE_DIR = "./data/png/"
DATASET_DIR = "./dataset/"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
SHARD_SIZE = 1024
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42
TARGET_WIDTH = 416
TARGET_HEIGHT = 576
CPU_COUNT = 20

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


# --- Custom Dataset for paths ---
class PathPairDataset(Dataset):
    """A dataset to hold pairs of file paths and their labels."""

    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.labels[idx]


# --- Pair Generation Functions ---


def create_pairs(document):
    """Creates pairs of consecutive pages from a document directory."""
    pages = sorted(os.listdir(document))
    pairs = []
    if len(pages) > 1:
        for i in range(len(pages) - 1):
            pairs.append(
                (
                    os.path.join(document, pages[i]),
                    os.path.join(document, pages[i + 1]),
                )
            )
    return pairs


def create_invalid_pair(doc1, doc2):
    """Creates a pair of pages from two different documents."""
    try:
        page1 = sorted(os.listdir(doc1))[-1]
        page2 = sorted(os.listdir(doc2))[0]
        return (os.path.join(doc1, page1), os.path.join(doc2, page2))
    except (IndexError, FileNotFoundError):
        return None


# --- Image Preprocessing ---


def preprocess_image(image_path):
    """Loads an image, resizes it to a target size with padding,
    and returns it as a numpy array.
    """
    try:
        img = Image.open(image_path).convert("L")  # Convert to grayscale
    except FileNotFoundError:
        return None

    original_width, original_height = img.size
    width_ratio = TARGET_WIDTH / original_width
    height_ratio = TARGET_HEIGHT / original_height
    scaling_factor = min(width_ratio, height_ratio)

    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)

    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    padded_img = Image.new("L", (TARGET_WIDTH, TARGET_HEIGHT), 0)
    paste_x = (TARGET_WIDTH - new_width) // 2
    paste_y = (TARGET_HEIGHT - new_height) // 2
    padded_img.paste(img_resized, (paste_x, paste_y))

    return np.array(padded_img, dtype=np.float32) / 255.0


# --- Sharding Logic ---


def process_and_save_shards(dataset, output_dir, shard_size):
    """Processes pairs of images from a dataset and saves them in sharded tensor files."""
    os.makedirs(output_dir, exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=shard_size,
        shuffle=True,
        num_workers=CPU_COUNT,
        collate_fn=lambda x: x,
    )

    for shard_index, shard_batch in enumerate(loader):
        processed_data = []
        processed_labels = []

        for pair, label in shard_batch:
            try:
                img1 = preprocess_image(pair[0])
                img2 = preprocess_image(pair[1])

                if img1 is not None and img2 is not None:
                    # Stack images into a 2-channel tensor (H, W, C)
                    combined_image = np.stack((img1, img2), axis=-1)
                    processed_data.append(torch.from_numpy(combined_image))
                    processed_labels.append(label)
            except Exception as e:
                print(f"Warning: Skipping pair {pair} due to error: {e}")

        if not processed_data:
            continue

        data_tensor = torch.stack(processed_data)
        labels_tensor = torch.tensor(
            processed_labels, dtype=torch.float32
        ).view(-1, 1)

        shard_path = os.path.join(output_dir, f"shard_{shard_index}.pt")
        torch.save({"data": data_tensor, "labels": labels_tensor}, shard_path)

        print(
            f"Saved shard {shard_index} with {len(processed_data)} samples to {shard_path}"
        )


# --- Main Execution ---


def main():
    """Main function to generate and shard the dataset."""
    if os.path.exists(DATASET_DIR):
        old_npz = os.path.join(DATASET_DIR, "dataset_1.npz")
        if os.path.exists(old_npz):
            os.remove(old_npz)
            print(f"Removed old dataset file: {old_npz}")
        if os.path.exists(TRAIN_DIR):
            shutil.rmtree(TRAIN_DIR)
        if os.path.exists(TEST_DIR):
            shutil.rmtree(TEST_DIR)

    documents = [
        os.path.join(BASE_DIR, d)
        for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
    ]
    multipage = [d for d in documents if len(os.listdir(d)) > 1]
    print(f"{len(multipage)} multipage documents found.")

    print("Creating valid pairs...")
    with Pool(CPU_COUNT) as p:
        res = p.map(create_pairs, multipage)
    valid_pairs = [pair for sublist in res for pair in sublist]
    print(f"{len(valid_pairs)} valid pairs created.")

    print("Creating invalid pairs...")
    doc_combinations = list(combinations(documents, 2))
    random.shuffle(doc_combinations)

    num_invalid_to_create = len(valid_pairs)

    invalid_pairs = []
    with Pool(CPU_COUNT) as p:
        args = [combo for combo in doc_combinations[:num_invalid_to_create]]
        results = p.starmap(create_invalid_pair, args)

    invalid_pairs = [p for p in results if p is not None]
    print(f"{len(invalid_pairs)} invalid pairs created.")

    valid_labels = np.zeros(len(valid_pairs))
    invalid_labels = np.ones(len(invalid_pairs))

    generator = torch.Generator().manual_seed(RANDOM_STATE)

    valid_dataset = PathPairDataset(valid_pairs, valid_labels)
    invalid_dataset = PathPairDataset(invalid_pairs, invalid_labels)

    v_train_size = int(len(valid_dataset) * (1 - TEST_SPLIT_SIZE))
    v_test_size = len(valid_dataset) - v_train_size
    v_train_dataset, v_test_dataset = random_split(
        valid_dataset, [v_train_size, v_test_size], generator=generator
    )

    i_train_size = int(len(invalid_dataset) * (1 - TEST_SPLIT_SIZE))
    i_test_size = len(invalid_dataset) - i_train_size
    i_train_dataset, i_test_dataset = random_split(
        invalid_dataset, [i_train_size, i_test_size], generator=generator
    )

    train_dataset = ConcatDataset([v_train_dataset, i_train_dataset])
    test_dataset = ConcatDataset([v_test_dataset, i_test_dataset])

    print(f"Training set size: {len(train_dataset)} pairs")
    print(f"Test set size: {len(test_dataset)} pairs")

    print("Processing training data...")
    process_and_save_shards(train_dataset, TRAIN_DIR, SHARD_SIZE)

    print("Processing test data...")
    process_and_save_shards(test_dataset, TEST_DIR, SHARD_SIZE)

    print("Dataset creation and sharding complete.")


if __name__ == "__main__":
    main()
