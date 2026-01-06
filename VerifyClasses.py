import csv

labels_path = "/work/user/custom_openimages/open-images-v7/train/labels/detections.csv"
classes_path = "/work/user/custom_openimages/open-images-v7/train/metadata/classes.csv"

label_map = {}
with open(classes_path, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        label_map[row[0]] = row[1]

unique_labels = set()
with open(labels_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        unique_labels.add(row["LabelName"])
        
print("Classes present in detections.csv:")
for label_code in unique_labels:
    class_name = label_map.get(label_code, "Unknown")
    print(f"  {label_code} -> {class_name}")
