import os
import csv
import json
import shutil
from PIL import Image
from multiprocessing import Pool, cpu_count
from functools import partial
import random



BASE = "./indoor-images-data/open-images-v7"
OUTBASE = "./dataset_64k"
CLASSES = [
    "Door", "Door handle", "Chair", "Table", "Bed", "Toilet", "Sink",
    "Bathtub", "Shower", "Stairs", "Couch", "Lamp", "Light switch",
    "Refrigerator", "Microwave oven", "Oven", "Gas stove", "Washing machine",
    "Computer keyboard", "Computer monitor", "Laptop", "Television", "Clock",
    "Waste container", "Plate", "Knife", "Fork", "Spoon", "Person", "Car", "Desk"
]

IMAGE_LIMITS = {
    "train": 64000,
    "validation": 8000,
    "test": 8000
}

random.seed(42)


def find_file(subdir, keyname):
    """Search for the first file matching keyname in subdir."""
    for root, dirs, files in os.walk(subdir):
        for fname in files:
            if keyname.lower() in fname.lower():
                return os.path.join(root, fname)
    raise RuntimeError(f"Could not find file with '{keyname}' in {subdir}")


def process_single_image(args):
    """Process a single image - for parallel execution"""
    filename, images_dir, outdir = args
    image_base_id = filename.split('.')[0]
    file_path = os.path.join(images_dir, filename)
    
    try:
        img = Image.open(file_path)
        width, height = img.size
        
    
        shutil.copy(file_path, os.path.join(outdir, filename))
        
        return {
            "filename": filename,
            "image_base_id": image_base_id,
            "width": width,
            "height": height,
            "success": True
        }
    except Exception as e:
        return {
            "filename": filename,
            "image_base_id": image_base_id,
            "success": False,
            "error": str(e)
        }


def get_images_with_target_classes(labels_path, allowed_labelnames):
    """
    Find ALL images that contain at least one of the target classes.
    Returns a set of image IDs (no limit applied here).
    """
    images_with_classes = set()
    
    print("  Scanning entire labels file for target classes...")
    with open(labels_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["LabelName"] in allowed_labelnames:
                images_with_classes.add(row["ImageID"])
    
    return images_with_classes



for split in ["train", "validation", "test"]:
    print(f"\n{'='*60}")
    print(f"Processing {split} split...")
    print(f"{'='*60}")
    
    rawdir = os.path.join(BASE, split)
    outdir = os.path.join(OUTBASE, split)
    os.makedirs(outdir, exist_ok=True)

    images_dir = os.path.join(rawdir, "data")
    labels_path = find_file(os.path.join(rawdir, "labels"), "detections")
    classes_path = find_file(os.path.join(rawdir, "metadata"), "classes")
    
    print(f"Images directory: {images_dir}")
    print(f"Labels file: {labels_path}")
    print(f"Classes file: {classes_path}")

    oi_labelname_to_classname = {}
    with open(classes_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            oi_labelname_to_classname[row[0]] = row[1]
    
    allowed_labelnames = {}
    for labelname, classname in oi_labelname_to_classname.items():
        if classname in CLASSES:
            allowed_labelnames[labelname] = classname
    
    print(f"Allowed label codes: {len(allowed_labelnames)}")


    image_limit = IMAGE_LIMITS[split]
    print(f"\nFinding images with target classes (target: {image_limit})...")
    

    images_with_target_classes = get_images_with_target_classes(
        labels_path, allowed_labelnames
    )
    
    print(f"  Found {len(images_with_target_classes)} images with target classes")
    

    print(f"\nScanning available image files in {images_dir}...")
    
    all_available_files = []
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_id = filename.split('.')[0]

            if image_id in images_with_target_classes:
                all_available_files.append(filename)
    
    print(f"  Found {len(all_available_files)} available image files with target classes")


    random.shuffle(all_available_files)
    files_to_process = all_available_files[:image_limit]
    
    print(f"\nProcessing {len(files_to_process)} images with {cpu_count()} workers...")
    

    process_args = [(f, images_dir, outdir) for f in files_to_process]
    

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_single_image, process_args)
    

    coco_images = []
    img_id_map = {}
    processed_image_ids = set()
    
    for idx, result in enumerate(results, start=0):
        if result["success"]:
            coco_images.append({
                "id": idx,
                "file_name": result["filename"],
                "width": result["width"],
                "height": result["height"]
            })
            img_id_map[result["image_base_id"]] = idx
            processed_image_ids.add(result["image_base_id"])
        else:
            print(f"  Warning: Failed to process {result['filename']}: {result.get('error')}")
    
    print(f"  Successfully processed {len(coco_images)} images")


    categories = []
    classname_to_coco_id = {}
    for idx, classname in enumerate(CLASSES, start=0):
        classname_to_coco_id[classname] = idx
        categories.append({
            "id": idx,
            "name": classname,
            "supercategory": "object"
        })
    
    print(f"COCO categories: {len(categories)}")


    print(f"\nProcessing annotations for {len(processed_image_ids)} images...")
    coco_annotations = []
    annotation_id = 1
    skipped_no_image = 0
    skipped_wrong_class = 0
    
    with open(labels_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_base_id = row["ImageID"]
            

            if image_base_id not in processed_image_ids:
                skipped_no_image += 1
                continue
            
            labelname = row["LabelName"]
            

            if labelname not in allowed_labelnames:
                skipped_wrong_class += 1
                continue
            
            coco_image_id = img_id_map[image_base_id]
            classname = allowed_labelnames[labelname]
            coco_category_id = classname_to_coco_id[classname]
            
            img_info = coco_images[coco_image_id]
            width = img_info["width"]
            height = img_info["height"]
            
            x_min = float(row["XMin"]) * width
            x_max = float(row["XMax"]) * width
            y_min = float(row["YMin"]) * height
            y_max = float(row["YMax"]) * height
            
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            bbox_area = bbox_width * bbox_height
            
            coco_annotations.append({
                "id": annotation_id,
                "image_id": coco_image_id,
                "category_id": coco_category_id,
                "bbox": [x_min, y_min, bbox_width, bbox_height],
                "area": bbox_area,
                "iscrowd": int(row.get("IsGroupOf", 0))
            })
            annotation_id += 1
    
    print(f"  Created {len(coco_annotations)} annotations")
    print(f"  Skipped {skipped_wrong_class} annotations from unwanted classes")
    print(f"  Skipped {skipped_no_image} annotations from unprocessed images")


    coco_dict = {
        "info": {
            "description": f"Open Images {split} split - {', '.join(CLASSES)}",
            "version": "1.0",
            "year": 2025,
            "contributor": "Open Images Dataset",
            "date_created": "2025/10/05",
            "note": f"Target: {image_limit} images"
        },
        "licenses": [
            {
                "id": 1,
                "name": "CC BY 4.0",
                "url": "https://creativecommons.org/licenses/by/4.0/"
            }
        ],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories
    }
    
    output_json = os.path.join(outdir, "_annotations.coco.json")
    with open(output_json, "w") as f:
        json.dump(coco_dict, f, indent=2)
    
    print(f"\n✓ Saved COCO JSON: {output_json}")
    print(f"  - Images: {len(coco_images)}")
    print(f"  - Annotations: {len(coco_annotations)}")
    print(f"  - Categories: {len(categories)}")



print(f"\n{'='*60}")
print("COCO CONVERSION COMPLETE")
print(f"{'='*60}")
print(f"Output directory: {OUTBASE}")
print(f"Classes included: {', '.join(CLASSES)}")
print(f"\nDataset targets:")
print(f"  - Train: {IMAGE_LIMITS['train']} images")
print(f"  - Validation: {IMAGE_LIMITS['validation']} images")
print(f"  - Test: {IMAGE_LIMITS['test']} images")
print(f"\nReady for RF-DETR training!")
