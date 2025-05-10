import os
import glob

assert os.path.exists("data.yaml"), "Dataset configuration file (data.yaml) is missing!"

train_label_dir = "datasets/train/labels"
val_label_dir = "datasets/val/labels"

print("Training labels found:", glob.glob("datasets/train/labels/*.txt"))

# List label files
train_labels = [f for f in os.listdir(train_label_dir) if f.endswith(".txt")]
val_labels = [f for f in os.listdir(val_label_dir) if f.endswith(".txt")]

# Count empty files
empty_train_labels = [f for f in train_labels if os.path.getsize(os.path.join(train_label_dir, f)) == 0]
empty_val_labels = [f for f in val_labels if os.path.getsize(os.path.join(val_label_dir, f)) == 0]

print(f"✅ Found {len(train_labels)} label files in train/labels")
print(f"✅ Found {len(val_labels)} label files in val/labels")

if empty_train_labels:
    print(f"⚠️ Warning: {len(empty_train_labels)} train labels are empty!")
if empty_val_labels:
    print(f"⚠️ Warning: {len(empty_val_labels)} validation labels are empty!")

train_img_dir = "datasets/train"
train_label_dir = "datasets/train/labels"

image_files = [f.replace(".png", ".txt") for f in os.listdir(train_img_dir) if f.endswith(".png")]
missing_labels = [f for f in image_files if not os.path.exists(os.path.join(train_label_dir, f))]

if missing_labels:
    print("❌ Missing labels for the following images:")
    for img in missing_labels:
        print(img)
else:
    print("✅ All images have corresponding labels.")

