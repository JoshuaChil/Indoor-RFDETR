
import os
import torch
import gc
from rfdetr import RFDETRNano

torch.cuda.empty_cache()
gc.collect()


DATASET_DIR = "./dataset_64k"
OUTPUT_DIR = "/"
NUM_CLASSES = 31


os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"Dataset: {DATASET_DIR}")
print(f"Output: {OUTPUT_DIR}")


print("\nModel Initialization")
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'  

from rfdetr import RFDETRMedium
model = RFDETRMedium(NUM_CLASSES=NUM_CLASSES)
model.train(
    dataset_dir=DATASET_DIR,
    epochs=15,
    batch_size=16,  
    grad_accum_steps=1,  
    lr=1e-5, 
    output_dir="./rfdetr_model",
)

print("Training Complete!")

