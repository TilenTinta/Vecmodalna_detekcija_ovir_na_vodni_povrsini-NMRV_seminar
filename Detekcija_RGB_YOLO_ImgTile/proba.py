import os
import glob
import yaml

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


######## IŠČI DUPLIKATE

# Load paths from data.yaml
with open("data.yaml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

train_dir = data['train']
val_dir = data['val']
test_dir = data.get('test', None)  # test is optional

def get_image_basenames(directory):
    if not os.path.exists(directory):
        return set()
    images = glob.glob(os.path.join(directory, "*.png")) + glob.glob(os.path.join(directory, "*.jpg"))
    return set(os.path.basename(img) for img in images)

train_images = get_image_basenames(train_dir)
val_images = get_image_basenames(val_dir)
test_images = get_image_basenames(test_dir) if test_dir else set()

# Find duplicates across splits
duplicate_train_val = train_images & val_images
duplicate_train_test = train_images & test_images
duplicate_val_test = val_images & test_images

if duplicate_train_val or duplicate_train_test or duplicate_val_test:
    print("⚠️ Similar images found in multiple datasets:")
    if duplicate_train_val:
        print(f"  🟡 Train ↔ Val ({len(duplicate_train_val)} images):")
        for img in duplicate_train_val:
            print("   -", img)
    if duplicate_train_test:
        print(f"  🟡 Train ↔ Test ({len(duplicate_train_test)} images):")
        for img in duplicate_train_test:
            print("   -", img)
    if duplicate_val_test:
        print(f"  🟡 Val ↔ Test ({len(duplicate_val_test)} images):")
        for img in duplicate_val_test:
            print("   -", img)
else:
    print("✅ No duplicate images found between datasets.")

# Your existing label verification code continues here...

