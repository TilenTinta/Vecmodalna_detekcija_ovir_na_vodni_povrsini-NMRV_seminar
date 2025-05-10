import os
import json
import cv2
import argparse
import random
import shutil

YOLO_CLASS = 0  # Only one class: obstacles

def tile_and_convert_annotations(json_file, image_folder, output_image_folder, output_label_folder, tile_size=(736, 414), overlap=0):
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)

    with open(json_file, 'r') as f:
        annotations = json.load(f)

    for image_name, bboxes in annotations.items():
        img_path = os.path.normpath(os.path.join(image_folder, image_name + ".png"))

        if not os.path.exists(img_path):
            print(f"Image {img_path} not found, skipping.")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read {img_path}, skipping.")
            continue

        img_height, img_width = img.shape[:2]
        tile_w, tile_h = tile_size
        stride_x = tile_w - overlap
        stride_y = tile_h - overlap

        tile_id = 0
        for y in range(0, img_height, stride_y):
            for x in range(0, img_width, stride_x):
                x_end = min(x + tile_w, img_width)
                y_end = min(y + tile_h, img_height)

                if x_end - x < tile_w or y_end - y < tile_h:
                    continue

                tile = img[y:y_end, x:x_end]
                tile_img_name = f"{image_name}_tile{tile_id}"
                tile_img_path = os.path.join(output_image_folder, tile_img_name + ".png")
                tile_label_path = os.path.join(output_label_folder, tile_img_name + ".txt")

                cv2.imwrite(tile_img_path, tile)

                with open(tile_label_path, 'w') as label_file:
                    for bbox in bboxes:
                        x_min, y_min, x_max, y_max = bbox

                        if x_max < x or x_min > x_end or y_max < y or y_min > y_end:
                            continue

                        x_min_tile = max(0, x_min - x)
                        y_min_tile = max(0, y_min - y)
                        x_max_tile = min(tile_w, x_max - x)
                        y_max_tile = min(tile_h, y_max - y)

                        x_center = (x_min_tile + x_max_tile) / 2 / tile_w
                        y_center = (y_min_tile + y_max_tile) / 2 / tile_h
                        width = (x_max_tile - x_min_tile) / tile_w
                        height = (y_max_tile - y_min_tile) / tile_h

                        if width <= 0 or height <= 0:
                            continue

                        label_file.write(f"{YOLO_CLASS} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                tile_id += 1

def copyImg(tile_image_dir, tile_label_dir):
    train_txt_path = "..\\NMRV_seminar\\train.txt"
    test_txt_path = "..\\NMRV_seminar\\test.txt"

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

    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(train_img_fol, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    os.makedirs(val_img_fol, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)
    os.makedirs(test_img_fol, exist_ok=True)

    with open(train_txt_path, "r") as f:
        train_images = [line.strip() for line in f.readlines()]
    with open(test_txt_path, "r") as f:
        test_images = [line.strip() for line in f.readlines()]

    all_tiles = [fname for fname in os.listdir(tile_image_dir) if fname.endswith(".png")]
    train_tiles = [t for t in all_tiles if any(t.startswith(im + "_") for im in train_images)]
    test_tiles = [t for t in all_tiles if any(t.startswith(im + "_") for im in test_images)]

    random.shuffle(train_tiles)
    split_idx = int(0.8 * len(train_tiles))
    train_split, val_split = train_tiles[:split_idx], train_tiles[split_idx:]

    def copy_tiles(file_list, dst_img_dir, dst_label_dir):
        for img_file in file_list:
            label_file = os.path.splitext(img_file)[0] + ".txt"
            shutil.copy(os.path.join(tile_image_dir, img_file), os.path.join(dst_img_dir, img_file))
            shutil.copy(os.path.join(tile_label_dir, label_file), os.path.join(dst_label_dir, label_file))

    copy_tiles(train_split, train_img_fol, train_label_dir)
    copy_tiles(val_split, val_img_fol, val_label_dir)
    copy_tiles(test_tiles, test_img_fol, test_label_dir)

    print(f"Train tiles: {len(train_split)}, Val tiles: {len(val_split)}, Test tiles: {len(test_tiles)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="..\\NMRV_seminar\\RGB_annotations.json")
    parser.add_argument("--images", type=str, default="..\\NMRV_seminar\\RGB_images")
    parser.add_argument("--out_images", type=str, default="images")
    parser.add_argument("--out_labels", type=str, default="labels")
    args = parser.parse_args()

    tile_and_convert_annotations(args.json, args.images, args.out_images, args.out_labels, tile_size=(736, 414))
    copyImg(args.out_images, args.out_labels)
