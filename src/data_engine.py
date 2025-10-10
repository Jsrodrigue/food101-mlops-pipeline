import json
import os
import random
import shutil
import zipfile
from pathlib import Path

from omegaconf import DictConfig
from torchvision.datasets import Food101
from tqdm import tqdm


def remove_readonly(func, path, excinfo):
    """
    Helper for Windows: clears readonly flag and retries removal.
    """
    os.chmod(path, 0o777)
    func(path)


def is_zipfile_valid(zip_path):
    """
    Checks if a zip file is valid (not corrupt).
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad_file = zf.testzip()
            return bad_file is None
    except Exception:
        return False


def create_data(
    cfg: DictConfig,
    keep_zip: bool = True,
    seed: int = None
):
    """
    Create a balanced subset of the Food101 dataset according to parameters defined
    in a Hydra configuration. Generates train/val/test splits, saves a class mapping file.

    Args:
        cfg (DictConfig): Hydra configuration with expected fields:
            - cfg.dataset.path (str): Path where dataset is stored or will be downloaded.
            - cfg.dataset.class_path (str): Path to save class mapping JSON.
            - cfg.dataset.creation.selected_classes (list[str], optional): Classes to include (if manual).
            - cfg.dataset.creation.samples_per_class (int): Number of images per class.
            - cfg.dataset.creation.train_ratio (float): Fraction for training split.
            - cfg.dataset.creation.val_ratio (float): Fraction for validation split.
            - cfg.dataset.creation.test_ratio (float): Fraction for test split.
            - cfg.dataset.creation.seed (int): Random seed for reproducibility.
            - cfg.dataset.creation.select_mode (str): "manual", "random" or "first"
            - cfg.dataset.creation.num_classes (int): Number of classes in random or first mode

        keep_zip (bool): If True, keeps the downloaded Food101 zip file. If False, deletes it after processing.
        select_mode (str): "manual" (use selected_classes from config), "random" (pick random classes), or "first" (pick first N alphabetically).
        num_classes (int): Number of classes to select if select_mode is "random" or "first".
        seed (int): Optional random seed to override config.

    Returns:
        tuple[Path, Path, Path, Path]:
            Paths to the train, validation, and test directories, and the class mapping JSON file.
    """
    # --- Step 1: Initialization ---
    if seed is not None:
        random.seed(seed)
    else:
        random.seed(cfg.dataset.creation.seed)

    samples_per_class = cfg.dataset.creation.samples_per_class
    train_ratio = cfg.dataset.creation.train_ratio
    val_ratio = cfg.dataset.creation.val_ratio
    test_ratio = cfg.dataset.creation.test_ratio
    select_mode = cfg.dataset.creation.select_mode
    num_classes = cfg.dataset.creation.num_classes


    # --- Step 2: Prepare dataset path ---
    subset_root = Path(cfg.dataset.path)
    if not subset_root.is_absolute():
        subset_root = Path.cwd() / subset_root

    # --- Check if dataset already exists ---
    train_dir = subset_root / "train"
    val_dir = subset_root / "val"
    test_dir = subset_root / "test"

    # Check if all splits exist and contain at least one class directory
    if all(d.exists() and any(d.iterdir()) for d in [train_dir, val_dir, test_dir]):
        print(
            "[INFO] Custom dataset already exists. Skipping download and preparation."
        )
        # Optionally, you could also check if the expected classes are present
        return train_dir, val_dir, test_dir, Path(cfg.dataset.class_path)

    images_dir = subset_root / "food-101" / "images"
    zip_file = subset_root / "food-101.zip"

    # --- Step 3: Download and extract if needed ---
    need_download = False
    if not images_dir.exists() or not any(images_dir.iterdir()):
        if zip_file.exists():
            print("[INFO] Found food-101.zip. Checking integrity...")
            if not is_zipfile_valid(zip_file):
                print("[WARNING] Zip file is corrupt. Removing and re-downloading...")
                zip_file.unlink()
                need_download = True
            else:
                print("[INFO] Zip file is valid. Extracting...")
                with zipfile.ZipFile(zip_file, "r") as zf:
                    zf.extractall(subset_root)
        else:
            need_download = True

        if need_download:
            print("[INFO] Downloading Food101 dataset...")
            Food101(root=subset_root, split="train", download=True)
            Food101(root=subset_root, split="test", download=True)
            print("[INFO] Download completed.")
    else:
        print("[INFO] Images folder exists. Verifying integrity...")
        print("[INFO] Dataset ready.")

    # --- Step 4: Select classes ---
    all_classes = sorted([d.name for d in images_dir.iterdir() if d.is_dir()])
    if select_mode == "manual":
        selected_classes = cfg.dataset.creation.selected_classes
        if not selected_classes:
            raise ValueError("selected_classes must be provided in manual mode.")
    elif select_mode == "random":
        if num_classes is None:
            raise ValueError("num_classes must be specified for random mode.")
        selected_classes = random.sample(all_classes, num_classes)
    elif select_mode == "first":
        if num_classes is None:
            raise ValueError("num_classes must be specified for first mode.")
        selected_classes = all_classes[:num_classes]
    else:
        raise ValueError(f"Unknown select_mode: {select_mode}")

    print(f"[INFO] Using classes: {selected_classes}")

    # --- Step 5: Create train/val/test directories ---
    for split in ["train", "val", "test"]:
        for cls in selected_classes:
            (subset_root / split / cls).mkdir(parents=True, exist_ok=True)

    # --- Step 6: Compute number of samples per split ---
    n_train = int(samples_per_class * train_ratio)
    n_val = int(samples_per_class * val_ratio)
    n_test = int(samples_per_class * test_ratio)

    # Adjust small rounding differences
    n_total = n_train + n_val + n_test
    if n_total < samples_per_class:
        n_train += samples_per_class - n_total
    elif n_total > samples_per_class:
        n_train -= n_total - samples_per_class

    print(f"[INFO] Split sizes per class → Train={n_train}, Val={n_val}, Test={n_test}")

    # --- Step 7: Copy images into splits ---
    for cls in tqdm(selected_classes, desc="[INFO] Processing classes"):
        class_path = images_dir / cls
        if not class_path.exists():
            print(f"[INFO] Warning: folder '{cls}' not found, skipping.")
            continue

        all_images = os.listdir(class_path)

        if len(all_images) < samples_per_class:
            selected_images = all_images
            total = len(selected_images)
            n_train_actual = int(total * train_ratio)
            n_val_actual = int(total * val_ratio)
            n_test_actual = total - n_train_actual - n_val_actual
        else:
            selected_images = random.sample(all_images, samples_per_class)
            n_train_actual, n_val_actual, n_test_actual = n_train, n_val, n_test

        train_imgs = selected_images[:n_train_actual]
        val_imgs = selected_images[n_train_actual : n_train_actual + n_val_actual]
        test_imgs = selected_images[
            n_train_actual
            + n_val_actual : n_train_actual
            + n_val_actual
            + n_test_actual
        ]

        for split_name, split_imgs in [
            ("train", train_imgs),
            ("val", val_imgs),
            ("test", test_imgs),
        ]:
            for img in split_imgs:
                shutil.copy2(class_path / img, subset_root / split_name / cls / img)

    print(f"[INFO] Subset created successfully at {subset_root}")

    # --- Step 8: Save alphabetically ordered class mapping ---
    class_map_path = Path(cfg.dataset.class_path)
    if not class_map_path.is_absolute():
        class_map_path = Path.cwd() / class_map_path
    class_map_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_classes = sorted(selected_classes)
    mapping = {
        "idx_to_class": {i: cls for i, cls in enumerate(sorted_classes)},
        "class_to_idx": {cls: i for i, cls in enumerate(sorted_classes)},
    }

    with open(class_map_path, "w") as f:
        json.dump(mapping, f, indent=4)

    print(f"[INFO] Class mapping saved at: {class_map_path}")
    print(f"[INFO] Alphabetical class order: {sorted_classes}")

    # --- Step 9: Remove unzipped folder and optionally the zip ---
    unzip_folder = subset_root / "food-101"
    if unzip_folder.exists():
        print(f"[INFO] Removing unzipped folder: {unzip_folder}")
        shutil.rmtree(unzip_folder, onerror=remove_readonly)

    if not keep_zip and zip_file.exists():
        print(f"[INFO] Removing zip file: {zip_file}")
        zip_file.unlink()

    return (
        subset_root / "train",
        subset_root / "val",
        subset_root / "test",
        class_map_path,
    )
