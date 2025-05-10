import os
import json
import cv2
import argparse
import random
import shutil

# YOLO razredi (samo en razred -> ovire)
YOLO_CLASS = 0  # 1 je 0 ker...?

def copyImg():
    image_dir = "..\\NMRV_seminar\\RGB_images"
    train_txt_path = "..\\NMRV_seminar\\train.txt"
    test_txt_path = "..\\NMRV_seminar\\test.txt"

    # Define dataset directories
    dataset_dir = "datasets"
    train_img_dir = os.path.join(dataset_dir, "train")
    train_label_dir = os.path.join(train_img_dir, "labels")
    train_img_fol = os.path.join(train_img_dir, "images")
    val_img_dir = os.path.join(dataset_dir, "val")
    val_label_dir = os.path.join(val_img_dir, "labels")
    val_img_fol = os.path.join(val_img_dir, "images")
    test_img_dir = os.path.join(dataset_dir, "test")
    test_label_dir = os.path.join(test_img_dir, "labels")
    test_img_fol = os.path.join(test_img_dir, "images")

    # Ensure all dataset directories exist
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(train_img_fol, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    os.makedirs(val_img_fol, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)
    os.makedirs(test_img_fol, exist_ok=True)

    # Define label directory
    label_dir = "yolo_labels"  # Directory containing YOLO `.txt` labels

    # Read train and test image paths
    with open(train_txt_path, "r") as f:
        train_images = [os.path.join(image_dir, line.strip() + ".png") for line in f.readlines()]  # Prepend full image path
    with open(test_txt_path, "r") as f:
        test_images = [os.path.join(image_dir, line.strip() + ".png") for line in f.readlines()]  # Prepend full image path

    # Split train images into train (80%) and validation (20%)
    random.shuffle(train_images)
    split_idx = int(0.8 * len(train_images))
    train_split, val_split = train_images[:split_idx], train_images[split_idx:]

    def resize_image(src_path, dst_path, size=(640, 360)):
        if os.path.exists(src_path):
            img = cv2.imread(src_path)
            resized_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)  # Resize with interpolation
            cv2.imwrite(dst_path, resized_img)  # Save resized image
        else:
            print(f"Warning: Image not found - {src_path}")

    # Function to copy images and labels
    def copy_files(file_list, dst_img_dir, dst_label_dir):
        for img_path in file_list:
            img_name = os.path.basename(img_path)  # Extract filename
            label_name = os.path.splitext(img_name)[0] + ".txt"

            img_dst_path = os.path.join(dst_img_dir, img_name)
            label_src_path = os.path.join(label_dir, label_name)
            label_dst_path = os.path.join(dst_label_dir, label_name)

            resize_image(img_path, img_dst_path)

            # Copy label if it exists
            if os.path.exists(label_src_path):
                shutil.copy(label_src_path, label_dst_path)
            else:
                print(f"Warning: Label not found for {img_name}, {label_dst_path}")

    # Copy images & labels to respective folders
    copy_files(train_split, train_img_fol, train_label_dir)
    copy_files(val_split, val_img_fol, val_label_dir)
    copy_files(test_images, test_img_fol, test_label_dir)

    print(f"Train: {len(train_split)}, Validation: {len(val_split)}, Test: {len(test_images)}")


def convert_annotations(json_file, image_folder, output_folder, resized_size=(640, 360)): # (512, 288)

    os.makedirs(output_folder, exist_ok=True) # preveri če output folder obstaja čene jo naredi

    # Naloži anotacije
    with open(json_file, 'r') as f:
        annotations = json.load(f)

    for image_name, bboxes in annotations.items():
        #img_path = os.path.join(image_folder, image_name + ".jpg") 
        #label_path = os.path.join(output_folder, image_name + ".txt")
        img_path = os.path.normpath(os.path.join(image_folder, image_name + ".png"))
        label_path = os.path.normpath(os.path.join(output_folder, image_name + ".txt"))

        # Da se ne sesuje če slike ni
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found, skipping...")
            continue

        # Preberi sliko
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read {img_path}, skipping...")
            continue
        height, width, _ = img.shape

        orig_height, orig_width = img.shape[:2]
        resized_width, resized_height = resized_size

        scale_x = resized_width / orig_width
        scale_y = resized_height / orig_height

        # Pretvorba bounding boxov
        with open(label_path, 'w') as label_file:
            for bbox in bboxes:
                x_min, y_min, x_max, y_max = bbox

                # Scale coordinates to resized image
                x_min_scaled = x_min * scale_x
                y_min_scaled = y_min * scale_y
                x_max_scaled = x_max * scale_x
                y_max_scaled = y_max * scale_y

                # YOLO format (x_center, y_center, width, height)
                x_center = (x_min_scaled + x_max_scaled) / 2 / resized_width
                y_center = (y_min_scaled + y_max_scaled) / 2 / resized_height
                bbox_width = (x_max_scaled - x_min_scaled) / resized_width
                bbox_height = (y_max_scaled - y_min_scaled) / resized_height

                label_file.write(f"{YOLO_CLASS} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

        # if os.path.exists(label_path) and os.path.getsize(label_path) == 0:
        #     os.remove(label_path)
        #     print(f"Removed empty label: {label_path}")

        # else:
        print(f"Saved: {label_path}")

if __name__ == "__main__":

    # REFORMATIRANJE ZA YOLO #
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="..\\NMRV_seminar\\RGB_annotations.json", help="Path to JSON annotation file")
    parser.add_argument("--images", type=str, default="..\\NMRV_seminar\\RGB_images", help="Folder containing images")
    parser.add_argument("--output", type=str, default="yolo_labels", help="Output folder for YOLO labels")
    args = parser.parse_args()

    convert_annotations(args.json, args.images, args.output, resized_size=(640, 360))

    # SPLITANJE V TRAIN IN VALIDACIJO #
    copyImg() # Kopiraj slike v datasets in jih loči po mapah