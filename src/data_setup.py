import os
import random

from torch import cuda
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from .utils.seed_utils import set_seed

NUM_WORKERS = os.cpu_count()


def create_dataloader_from_folder(
    data_dir,
    batch_size,
    transform,
    subset_percentage=1.0,
    shuffle=True,
    seed=42,
    num_workers=NUM_WORKERS,
):
    """
    Create a DataLoader from an image folder.

    Args:
        data_dir (str): Path to the folder containing images organized by class.
        batch_size (int): Batch size for the DataLoader.
        transform (torchvision.transforms): Transformations to apply to images.
        subset_percentage (float, optional): Fraction of the dataset to use (0 < subset_percentage <= 1.0).
        shuffle (bool, optional): Whether to shuffle the data.
        seed (int, optional): Random seed for reproducibility.
        num_workers (int, optional): Number of worker processes for loading data.

    Returns:
        loader (DataLoader): PyTorch DataLoader for the dataset.
        class_names (list): List of class names in the dataset.
    """
    # Set random seeds for reproducibility using utils.set_seed
    set_seed(seed)

    # Create the dataset from the folder
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    # Optionally take a random subset of the dataset
    if 0 < subset_percentage < 1.0:
        indices = random.sample(
            range(len(dataset)), int(len(dataset) * subset_percentage)
        )
        dataset = Subset(dataset, indices)

    # Create the DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if cuda.is_available() else False,
    )

    # Get class names
    class_names = (
        dataset.dataset.classes if isinstance(dataset, Subset) else dataset.classes
    )

    return loader, class_names
