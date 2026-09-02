import os
from datasets import load_dataset

OUTPUT_DIR = "./data/raw/imdb"

def download_imdb():
    if os.path.isdir(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
        print(f"IMDb dataset already present at {OUTPUT_DIR}, skipping download.")
        return

    print("Downloading IMDb dataset from Hugging Face...")

    # Load the standard dataset with the explicit namespace
    dataset = load_dataset("stanfordnlp/imdb")

    # Ensure the raw data directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save the dataset to the local folder
    dataset.save_to_disk(OUTPUT_DIR)
    print(f"Dataset successfully saved locally to {OUTPUT_DIR}")

if __name__ == "__main__":
    download_imdb()
