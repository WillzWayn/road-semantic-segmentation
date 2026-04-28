import argparse
import os
import shutil
import subprocess
import sys
import zipfile


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
DEFAULT_DATASET = "balraj98/deepglobe-road-extraction-dataset"
DEFAULT_ZIP_NAME = "deepglobe-road-extraction-dataset.zip"
EXPECTED_SPLITS = ("train", "valid", "test")


def dataset_already_exists(dataset_dir):
    for split in EXPECTED_SPLITS:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_dir):
            continue

        for file_name in os.listdir(split_dir):
            if file_name.endswith("_sat.jpg") or file_name.endswith("_mask.png"):
                return True

    return False


def ensure_kaggle_cli():
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "kaggle CLI not found. Install it first with `pip install kaggle` and configure your credentials."
        )


def download_zip(dataset, zip_path):
    cmd = [
        "kaggle",
        "datasets",
        "download",
        dataset,
        "--path",
        os.path.dirname(zip_path),
        "--force",
    ]
    subprocess.run(cmd, check=True)


def extract_zip(zip_path, dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)


def main():
    parser = argparse.ArgumentParser(description="Download and extract the DeepGlobe dataset from Kaggle")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Kaggle dataset slug",
    )
    parser.add_argument(
        "--dataset-dir",
        default=DATASET_DIR,
        help="Directory where the dataset will be extracted",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded zip after extraction",
    )
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    zip_path = os.path.join(PROJECT_ROOT, DEFAULT_ZIP_NAME)

    if dataset_already_exists(dataset_dir):
        print(f"Dataset already found in: {dataset_dir}")
        print("Aborting download to avoid overwriting existing files.")
        return

    try:
        ensure_kaggle_cli()
        print(f"Downloading {args.dataset} ...")
        download_zip(args.dataset, zip_path)

        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Expected zip not found after download: {zip_path}")

        print(f"Extracting to {dataset_dir} ...")
        extract_zip(zip_path, dataset_dir)

        if not args.keep_zip:
            os.remove(zip_path)
            print("Downloaded zip removed.")

        print("Dataset download and extraction completed.")
    except subprocess.CalledProcessError as exc:
        print(f"Kaggle download failed with exit code {exc.returncode}.", file=sys.stderr)
        sys.exit(exc.returncode)
    except Exception as exc:
        print(f"Failed to download dataset: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
