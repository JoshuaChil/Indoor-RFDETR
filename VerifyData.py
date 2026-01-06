import os
import sys
import csv
import json
from PIL import Image
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial


BASE = "./indoor-images-data/open-images-v7"
OUTBASE = "./dataset_64k"
SPLITS = ["train", "validation", "test"]

print(f"Configuration loaded:", flush=True)
print(f"  BASE: {BASE}", flush=True)
print(f"  OUTBASE: {OUTBASE}", flush=True)
print(f"  SPLITS: {SPLITS}", flush=True)
sys.stdout.flush()

NUM_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count()))
print(f"Using {NUM_WORKERS} CPU cores for parallel processing", flush=True)
sys.stdout.flush()


def find_file(subdir, keyname):
    """Search for the first file matching keyname in subdir."""
    for root, dirs, files in os.walk(subdir):
        for fname in files:
            if keyname.lower() in fname.lower():
                return os.path.join(root, fname)
    raise RuntimeError(f"Could not find file with '{keyname}' in {subdir}")


def check_image_file(args):
    "Check a single image file - must be top-level for multiprocessing"
    img_info, outdir = args
    filename = img_info['file_name']
    img_path = os.path.join(outdir, filename)
    
    result = {
        'filename': filename,
        'exists': False,
        'dimensions_match': False,
        'error': None
    }
    
    if not os.path.exists(img_path):
        result['error'] = "File not found"
        return result
    
    result['exists'] = True
    
    try:
        with Image.open(img_path) as img:
            img.load() 
            actual_width, actual_height = img.size
            
            if img_info['width'] == actual_width and img_info['height'] == actual_height:
                result['dimensions_match'] = True
            else:
                result['error'] = f"Dimension mismatch: JSON=({img_info['width']}, {img_info['height']}), actual=({actual_width}, {actual_height})"
    except Exception as e:
        result['error'] = f"Cannot open: {str(e)}"
    
    return result


def verify_single_image(args):
    """
    Verify a single image's annotations match between original and COCO.
    This function is designed for parallel execution.
    """
    (coco_img, original_annotations, coco_annotations_by_image, 
     coco_category_id_to_name) = args
    
    coco_image_id = coco_img['id']
    filename = coco_img['file_name']
    original_image_id = filename.split('.')[0]
    
    result = {
        'filename': filename,
        'coco_image_id': coco_image_id,
        'verified': False,
        'errors': [],
        'warnings': [],
        'annotation_count': 0
    }
    
    if original_image_id not in original_annotations:
        result['errors'].append(f"Image {filename} not found in original annotations")
        return result
    

    original_anns = original_annotations[original_image_id]
    coco_anns = coco_annotations_by_image.get(coco_image_id, [])
    

    if len(original_anns) != len(coco_anns):
        result['warnings'].append(
            f"Annotation count mismatch: Original={len(original_anns)}, COCO={len(coco_anns)}"
        )
    

    img_width = coco_img['width']
    img_height = coco_img['height']
    
    matches = 0
    for coco_ann in coco_anns:
  
        x_min, y_min, width, height = coco_ann['bbox']
        x_max = x_min + width
        y_max = y_min + height
        
     
        norm_x_min = x_min / img_width
        norm_x_max = x_max / img_width
        norm_y_min = y_min / img_height
        norm_y_max = y_max / img_height
        
        category_name = coco_category_id_to_name[coco_ann['category_id']]
        

        found_match = False
        tolerance = 0.001
        
        for orig_ann in original_anns:
            if orig_ann['ClassName'] != category_name:
                continue
            

            if (abs(norm_x_min - orig_ann['XMin']) < tolerance and
                abs(norm_x_max - orig_ann['XMax']) < tolerance and
                abs(norm_y_min - orig_ann['YMin']) < tolerance and
                abs(norm_y_max - orig_ann['YMax']) < tolerance):
                found_match = True
                matches += 1
                break
        
        if not found_match:
            result['errors'].append(
                f"No match for {category_name} at bbox [{norm_x_min:.4f}, {norm_y_min:.4f}, {norm_x_max:.4f}, {norm_y_max:.4f}]"
            )
    
    result['annotation_count'] = matches
    result['verified'] = (len(result['errors']) == 0)
    
    return result


def verify_conversion(split):
    """
    Verifies that the COCO conversion correctly matched annotations to images
    by cross-referencing original Open Images data with converted COCO data.
    OPTIMIZED: Only loads annotations for converted images.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"Verifying {split} split conversion", flush=True)
    print(f"{'='*60}", flush=True)
    sys.stdout.flush()
    

    rawdir = os.path.join(BASE, split)
    outdir = os.path.join(OUTBASE, split)
    coco_json_path = os.path.join(outdir, "_annotations.coco.json")
    
    if not os.path.exists(coco_json_path):
        print(f"⚠ Skipping {split}: COCO JSON not found at {coco_json_path}", flush=True)
        sys.stdout.flush()
        return False, 0, 0
    
    print(f"\nSearching for required files...", flush=True)
    sys.stdout.flush()
    
    try:
        labels_path = find_file(os.path.join(rawdir, "labels"), "detections")
        classes_path = find_file(os.path.join(rawdir, "metadata"), "classes")
        print(f"  ✓ Found labels: {labels_path}", flush=True)
        print(f"  ✓ Found classes: {classes_path}", flush=True)
    except Exception as e:
        print(f"  ✗ ERROR finding files: {e}", flush=True)
        sys.stdout.flush()
        return False, 0, 0
    
    sys.stdout.flush()

    print(f"\nLoading class mappings...", flush=True)
    oi_labelname_to_classname = {}
    with open(classes_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            oi_labelname_to_classname[row[0]] = row[1]
    
    print(f"  ✓ Loaded {len(oi_labelname_to_classname)} class mappings", flush=True)
    sys.stdout.flush()
    

    print(f"\nLoading COCO converted data...", flush=True)
    sys.stdout.flush()
    
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)
    
    print(f"  ✓ Images: {len(coco_data['images'])}", flush=True)
    print(f"  ✓ Annotations: {len(coco_data['annotations'])}", flush=True)
    print(f"  ✓ Categories: {len(coco_data['categories'])}", flush=True)
    sys.stdout.flush()
    
  
    print(f"\nBuilding image ID filter...", flush=True)
    coco_image_base_ids = set()
    for img in coco_data['images']:
        image_base_id = img['file_name'].split('.')[0]
        coco_image_base_ids.add(image_base_id)
    
    print(f"  → Need annotations for {len(coco_image_base_ids)} images only", flush=True)
    sys.stdout.flush()
    
  
    print(f"\nLoading original annotations (filtered)...", flush=True)
    original_annotations = defaultdict(list)
    
    loaded_count = 0
    skipped_count = 0
    
    with open(labels_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["ImageID"]
            
   
            if image_id not in coco_image_base_ids:
                skipped_count += 1
                continue
            
            original_annotations[image_id].append({
                "LabelName": row["LabelName"],
                "ClassName": oi_labelname_to_classname.get(row["LabelName"], "Unknown"),
                "XMin": float(row["XMin"]),
                "XMax": float(row["XMax"]),
                "YMin": float(row["YMin"]),
                "YMax": float(row["YMax"]),
                "IsGroupOf": row.get("IsGroupOf", "0")
            })
            loaded_count += 1
    
    print(f"  ✓ Loaded {loaded_count} annotations for {len(original_annotations)} images", flush=True)
    print(f"  ✓ Skipped {skipped_count} annotations not in COCO dataset", flush=True)
    sys.stdout.flush()
    
  
    coco_category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    
    coco_annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        coco_annotations_by_image[ann['image_id']].append(ann)
    

    print(f"\nCross-referencing annotations using {NUM_WORKERS} workers...", flush=True)
    sys.stdout.flush()
    

    verify_args = [
        (img, original_annotations, coco_annotations_by_image, coco_category_id_to_name)
        for img in coco_data['images']
    ]
    
    print(f"  → Processing {len(verify_args)} images in parallel...", flush=True)
    sys.stdout.flush()
    

    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(verify_single_image, verify_args)
    
    print(f"  ✓ Parallel processing complete", flush=True)
    sys.stdout.flush()
    

    print(f"\nAggregating results...", flush=True)
    sys.stdout.flush()
    
    verified_images = sum(1 for r in results if r['verified'])
    verified_annotations = sum(r['annotation_count'] for r in results)
    
    all_errors = []
    all_warnings = []
    
    for result in results:
        if result['errors']:
            for error in result['errors']:
                all_errors.append(f"{result['filename']}: {error}")
        if result['warnings']:
            for warning in result['warnings']:
                all_warnings.append(f"{result['filename']}: {warning}")
    

    print(f"\n{'='*60}", flush=True)
    print("VERIFICATION RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    
    print(f"\n✓ Verified {verified_images}/{len(coco_data['images'])} images", flush=True)
    print(f"✓ Verified {verified_annotations} annotations", flush=True)
    sys.stdout.flush()
    
    all_passed = True
    
    if all_errors:
        all_passed = False
        print(f"\n❌ {len(all_errors)} annotation errors found:", flush=True)
        for msg in all_errors[:20]:
            print(f"  - {msg}", flush=True)
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more", flush=True)
        sys.stdout.flush()
    
    if all_warnings:
        print(f"\n⚠ {len(all_warnings)} warnings:", flush=True)
        for msg in all_warnings[:20]:
            print(f"  - {msg}", flush=True)
        if len(all_warnings) > 20:
            print(f"  ... and {len(all_warnings) - 20} more", flush=True)
        sys.stdout.flush()
    
    if all_passed:
        print(f"\n✅ ALL CHECKS PASSED!", flush=True)
        print(f"✅ Annotations are correctly matched to images!", flush=True)
        sys.stdout.flush()
    
    return all_passed, verified_images, verified_annotations


def verify_image_files_parallel(split):
    """
    Verify all image files exist and dimensions match using parallel processing.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"Verifying image files for {split} split", flush=True)
    print(f"{'='*60}", flush=True)
    sys.stdout.flush()
    
    outdir = os.path.join(OUTBASE, split)
    coco_json_path = os.path.join(outdir, "_annotations.coco.json")
    
    if not os.path.exists(coco_json_path):
        print(f"⚠ Skipping {split}: COCO JSON not found", flush=True)
        sys.stdout.flush()
        return True
    
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)
    
    print(f"\nVerifying {len(coco_data['images'])} image files using {NUM_WORKERS} workers...", flush=True)
    sys.stdout.flush()
    

    check_args = [(img, outdir) for img in coco_data['images']]
    
    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(check_image_file, check_args)
    

    missing_files = [r for r in results if not r['exists']]
    dimension_errors = [r for r in results if r['exists'] and not r['dimensions_match']]
    corrupted_images = [r for r in results if r['error'] and 'truncated' in r['error'].lower()]
    
    if missing_files:
        print(f"\n {len(missing_files)} missing image files (showing first 10):", flush=True)
        for r in missing_files[:10]:
            print(f"  - {r['filename']}: {r['error']}", flush=True)
        sys.stdout.flush()
        return False
    
    if dimension_errors:
        print(f"\n {len(dimension_errors)} dimension mismatches (showing first 10):", flush=True)
        for r in dimension_errors[:10]:
            print(f"  - {r['filename']}: {r['error']}", flush=True)
        sys.stdout.flush()
        return False
    
    if corrupted_images:
        print(f"\n⚠ {len(corrupted_images)} corrupted/truncated images found (showing first 10):", flush=True)
        for r in corrupted_images[:10]:
            print(f"  - {r['filename']}: {r['error']}", flush=True)
        print(f"\n  → Run the corrupted image cleaner script to remove these", flush=True)
        sys.stdout.flush()
        return False
    
    print(f"\n All {len(results)} image files verified successfully", flush=True)
    sys.stdout.flush()
    return True


def generate_statistics(split):
    """Generate dataset statistics"""
    print(f"\n{'='*60}", flush=True)
    print(f"Generating statistics for {split} split", flush=True)
    print(f"{'='*60}", flush=True)
    sys.stdout.flush()
    
    outdir = os.path.join(OUTBASE, split)
    coco_json_path = os.path.join(outdir, "_annotations.coco.json")
    
    if not os.path.exists(coco_json_path):
        print(f"⚠ Skipping {split}: COCO JSON not found", flush=True)
        return
    
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)
    

    category_counts = defaultdict(int)
    for ann in coco_data['annotations']:
        category_counts[ann['category_id']] += 1
    
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    print(f"\nDataset Statistics:", flush=True)
    print(f"  Total images: {len(coco_data['images'])}", flush=True)
    print(f"  Total annotations: {len(coco_data['annotations'])}", flush=True)
    print(f"  Average annotations per image: {len(coco_data['annotations']) / len(coco_data['images']):.2f}", flush=True)
    
    print(f"\nAnnotations per category:", flush=True)
    for cat_id, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        cat_name = category_id_to_name.get(cat_id, f"Unknown ({cat_id})")
        print(f"  {cat_name:25s}: {count:6d}", flush=True)
    
    sys.stdout.flush()



if __name__ == "__main__":
    sys.stdout.flush()
    
    all_splits_passed = True
    
    for split in SPLITS:
        try:
            print(f"\n\n{'#'*60}", flush=True)
            print(f"# Processing {split.upper()} split", flush=True)
            print(f"{'#'*60}", flush=True)
            sys.stdout.flush()
            

            files_ok = verify_image_files_parallel(split)
            
    
            passed, num_images, num_annotations = verify_conversion(split)
            

            generate_statistics(split)
            
            all_splits_passed = all_splits_passed and passed and files_ok
            
        except Exception as e:
            print(f"\n❌ Error verifying {split}: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            all_splits_passed = False
    
    print("Verification complete!", flush=True)
    sys.stdout.flush()
