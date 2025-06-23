import os
from datasets import load_dataset
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial


def save_example(args):
    """Helper function to save a single example"""
    idx, example, save_path, img_col, label_col = args
    label = example[label_col]
    img = example[img_col]

    # Create class folder
    class_folder = os.path.join(save_path, str(label))
    os.makedirs(class_folder, exist_ok=True)

    # Save image
    img_filename = os.path.join(class_folder, f"{idx}.png")
    img.save(img_filename)

    return label


def save_hf_dataset_locally(
    hf_id: str,
    splits: list,
    out_dir: str,
    img_col: str,
    label_col: str,
    num_samples_per_class: int = None,
    num_workers: int = None,
    split_val_from_train: bool = False,
):
    """
    Downloads a Hugging Face dataset, organizes it by class, and saves it locally.
    Uses parallel processing to speed up the saving process.
    """
    if num_workers is None:
        num_workers = 4

    dataset = load_dataset(hf_id)
    os.makedirs(out_dir, exist_ok=True)

    save_func = partial(save_example)

    for split in splits:
        if split not in dataset:
            print(f"Split '{split}' not found in the dataset. Skipping.")
            continue

        split_ds = dataset[split]

        # Handle validation split if requested
        if split == "train" and split_val_from_train:
            # Create balanced validation split
            split_dataset = create_balanced_val_split(split_ds, label_col)

            # Save validation split
            val_path = os.path.join(out_dir, "val")
            print(f"Saving validation data to {val_path}...")

            val_args_list = [
                (idx, example, val_path, img_col, label_col)
                for idx, example in enumerate(split_dataset["test"])
            ]

            # Handle num_samples_per_class for validation
            if num_samples_per_class is not None:
                val_args_list = filter_samples_per_class(
                    val_args_list, label_col, num_samples_per_class
                )

            # Process validation split
            with Pool(num_workers) as pool:
                list(
                    tqdm(
                        pool.imap(save_func, val_args_list),
                        total=len(val_args_list),
                        desc="Processing validation split",
                    )
                )

            # Update train split
            split_ds = split_dataset["train"]

        save_path = os.path.join(out_dir, split)
        print(f"Saving {split} data to {save_path}...")

        # Prepare arguments for parallel processing
        args_list = [
            (idx, example, save_path, img_col, label_col)
            for idx, example in enumerate(split_ds)
        ]

        # If num_samples_per_class is set, filter examples
        if num_samples_per_class is not None:
            args_list = filter_samples_per_class(
                args_list, label_col, num_samples_per_class
            )

        # Process in parallel
        with Pool(num_workers) as pool:
            list(
                tqdm(
                    pool.imap(save_func, args_list),
                    total=len(args_list),
                    desc=f"Processing {split}",
                )
            )

        print(f"{split} data saved at {save_path}.")


def filter_samples_per_class(args_list, label_col, num_samples_per_class):
    """Helper function to filter samples per class"""
    class_counts = {}
    filtered_args = []
    for args in args_list:
        label = args[1][label_col]
        class_counts.setdefault(label, 0)
        if class_counts[label] < num_samples_per_class:
            filtered_args.append(args)
            class_counts[label] += 1
    return filtered_args


def create_balanced_val_split(dataset, label_col, val_ratio=0.1):
    """
    Create a balanced validation split with equal samples per class

    Args:
        dataset: The dataset to split
        label_col: Name of the label column
        val_ratio: Ratio of validation samples (default: 0.1)

    Returns:
        train_dataset, val_dataset
    """
    # Group examples by class
    class_examples = {}
    for idx, example in enumerate(dataset):
        label = example[label_col]
        if label not in class_examples:
            class_examples[label] = []
        class_examples[label].append(example)

    val_examples = []
    train_examples = []

    # For each class, take val_ratio of samples
    for label, examples in class_examples.items():
        n_val = max(1, int(len(examples) * val_ratio))
        val_examples.extend(examples[:n_val])
        train_examples.extend(examples[n_val:])

    return {"train": train_examples, "test": val_examples}


def check_if_already_exists(out_dir: str, splits: list):
    """
    Check if the dataset has already been downloaded.
    If Yes, skip the download.

    Args:
        out_dir (str): Local path to save the dataset.
        splits (list): Dataset splits (e.g., ['train', 'test']).

    Returns:
        bool: True if the dataset already exists, False otherwise.
    """

    for split in splits:
        if not os.path.exists(os.path.join(out_dir, split)):
            print(f"Dataset not found at {out_dir}/{split}. Proceeding with download.")
            return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument(
        "--hf_id",
        type=str,
        default="dpdl-benchmark/oxford_flowers102",
        help="Hugging Face dataset ID",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,test",
        help="Comma-separated list of dataset splits",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed/flowers102",
        help="Output directory",
    )
    parser.add_argument(
        "--img_col",
        type=str,
        default="image",
        help="Column name for images in the dataset",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="label",
        help="Column name for labels in the dataset",
    )
    parser.add_argument(
        "--num_samples_per_class",
        type=int,
        default=None,
        help="Max number of samples to save per class (per split). If not set, saves all.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes. Defaults to number of CPU cores.",
    )

    parser.add_argument(
        "--split_val_from_train",
        action="store_true",
        help="Split 10%% of training data as validation set, if there's no 'val' split in the dataset.",
    )

    args = parser.parse_args()
    splits = args.splits.split(",")

    # Validate split_val_from_train argument
    if args.split_val_from_train and "train" not in splits:
        raise ValueError(
            "--split_val_from_train can only be used when 'train' is included in --splits"
        )

    if check_if_already_exists(
        args.out_dir, splits + (["val"] if args.split_val_from_train else [])
    ):
        print("Dataset already exists. Skipping download.")
    else:
        save_hf_dataset_locally(
            hf_id=args.hf_id,
            splits=splits,
            out_dir=args.out_dir,
            img_col=args.img_col,
            label_col=args.label_col,
            num_samples_per_class=args.num_samples_per_class,
            num_workers=args.num_workers,
            split_val_from_train=args.split_val_from_train,
        )
