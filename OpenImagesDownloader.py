import fiftyone as fo
import fiftyone.zoo as foz
import os

OUTDIR = "./indoor-open-images-data"

CLASSES = [
    "Door", "Door handle", "Chair", "Table", "Bed", "Toilet", "Sink",
    "Bathtub", "Shower", "Stairs", "Couch", "Lamp", "Light switch",
    "Refrigerator", "Microwave oven", "Oven", "Gas stove", "Washing machine",
    "Computer keyboard", "Computer monitor", "Laptop", "Television", "Clock",
    "Waste container", "Plate", "Knife", "Fork", "Spoon", "Person", "Car", "Desk"
]

os.makedirs(OUTDIR, exist_ok=True)
print("Output directory:", OUTDIR)

fo.config.default_ml_backend = "torch"
fo.config.dataset_zoo_dir = OUTDIR

import fiftyone.utils.annotations as foua
if hasattr(foua, 'num_workers'):
    foua.num_workers = 32  

splits_config = [
    ("train", "train", 64000),
    ("valid", "validation", 8000),
    ("test", "test", 8000),
]

for split, fo_split, max_samples in splits_config:
    print(f"Downloading {split} split ({max_samples} samples)")
    try:
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split=fo_split,
            label_types=["detections"],
            classes=CLASSES,
            max_samples=max_samples,
            dataset_name=f"oi_{split}_{max_samples}",
            shuffle=True,
            seed=42,
        )
        
        print(f" Downloaded {len(dataset)} images")
        
    except Exception as e:
        print(f"Exception: {e}")
        continue

print("Downloading finished.")
