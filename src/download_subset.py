"""
Download a custom subset of Open Images V7 using FiftyOne.
Allows downloading specific classes or limiting the number of samples
to save disk space and training time.
"""
import os
import argparse
import logging
from src.utils import setup_logging

logger = setup_logging()

def download_open_images_subset(output_dir, classes=None, max_samples=1000, seed=42):
    """
    Downloads a subset of Google Open Images V7.
    
    Args:
        output_dir (str): Directory to save the exported YOLOv8 dataset.
        classes (list): List of classes to download (e.g., ['Person', 'Car']). None for all.
        max_samples (int): Maximum number of samples to download per split.
        seed (int): Random seed for sampling.
    """
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        raise ImportError(
            "FiftyOne is required for downloading subsets. Install it using:\n"
            "  pip install fiftyone"
        )
    
    splits = ["train", "validation"]
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Downloading Open Images V7 subset...")
    logger.info(f"Classes: {classes if classes else 'All'}")
    logger.info(f"Max samples per split: {max_samples}")
    
    for split in splits:
        logger.info(f"Processing split: {split}")
        
        # Load dataset from FiftyOne Zoo (downloads only metadata + requested images)
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split=split,
            label_types=["detections"],
            classes=classes,
            max_samples=max_samples,
            seed=seed,
            shuffle=True,
            drop_existing_dataset=True
        )
        
        # Export to YOLOv8 format
        export_dir = os.path.join(output_dir, split)
        logger.info(f"Exporting {split} to YOLOv8 format at {export_dir}...")
        
        # FiftyOne maps classes automatically
        dataset.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,  # YOLOv5 format is compatible with YOLOv8
            label_field="ground_truth",
            split=split
        )
        
    logger.info(f"Subset download and export complete. Dataset saved to {output_dir}")
    print(f"\nSubset dataset ready. Path: {os.path.join(output_dir, 'dataset.yaml')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a custom subset of Open Images V7")
    parser.add_argument("--output", type=str, default="data/open_images_subset", help="Output directory")
    parser.add_argument("--classes", type=str, nargs="+", default=["Person", "Car", "Dog"], help="Classes to download")
    parser.add_argument("--samples", type=int, default=1000, help="Max samples per split")
    args = parser.parse_args()
    
    download_open_images_subset(
        output_dir=args.output,
        classes=args.classes,
        max_samples=args.samples
    )
