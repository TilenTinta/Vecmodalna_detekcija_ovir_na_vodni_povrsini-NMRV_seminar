import os
import yaml
import glob

# Load data.yaml
with open("data.yaml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

splits = ["train", "val"] # "test" - ne brisat ker rabiš vse slike
removed_count = 0

for split in splits:
    dir_path = data.get(split)
    if not dir_path or not os.path.exists(dir_path):
        continue

    image_dir = os.path.join(dir_path, "images")
    label_dir = os.path.join(dir_path, "labels")

    if not os.path.exists(label_dir):
        continue

    image_files = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))

    for img_path in image_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, f"{basename}.txt")

        # If label file is missing or empty, remove the image and (if it exists) the label
        if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
            os.remove(img_path)
            if os.path.exists(label_path):
                os.remove(label_path)
            removed_count += 1
            print(f"🗑️ Removed: {img_path} and {label_path if os.path.exists(label_path) else '(label not found)'}")

print(f"\n✅ Done. Total images removed: {removed_count}")
