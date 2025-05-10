import os
from glob import glob
import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict


def run_prediction_on_test():
    model_path = "yolo_ladica/OvireNaVodi/weights/best.pt"
    model = YOLO(model_path)

    tile_w, tile_h = 736, 414
    tiles_per_row = 3
    tiles_per_image = 9

    image_dir = "datasets/test/images"
    all_tile_paths = sorted(glob(os.path.join(image_dir, "*.png")) + glob(os.path.join(image_dir, "*.jpg")))

    merged_dir = "merged_detections"
    os.makedirs(merged_dir, exist_ok=True)

    # Group tiles by original image name
    tiles_by_image = defaultdict(list)
    for path in all_tile_paths:
        base = os.path.basename(path)
        if "_tile" in base:
            orig_name = base.split("_tile")[0]
            tiles_by_image[orig_name].append(path)

    for orig_name, tile_paths in tiles_by_image.items():
        merged_boxes = []

        # Predict tile-by-tile
        for tile_path in tile_paths:
            result = model.predict(
                source=tile_path,
                imgsz=736,
                conf=0.5,
                save=False,
                save_txt=False,
                verbose=False,
                batch=1,
                amp=True
            )[0]

            tile_name = os.path.basename(tile_path)
            tile_id = int(tile_name.split("_tile")[1].split(".")[0])
            row = tile_id // tiles_per_row
            col = tile_id % tiles_per_row

            if result.boxes is None:
                continue

            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = box.tolist()
                # Adjust for tile position
                x1 += col * tile_w
                x2 += col * tile_w
                y1 += row * tile_h
                y2 += row * tile_h
                merged_boxes.append([int(x1), int(y1), int(x2), int(y2)])

        # Merge tiles into a full image canvas
        canvas = np.zeros((tile_h * 3, tile_w * 3, 3), dtype=np.uint8)
        for i in range(tiles_per_image):
            tile_path = os.path.join(image_dir, f"{orig_name}_tile{i}.png")
            if not os.path.exists(tile_path):
                continue
            tile = cv2.imread(tile_path)
            if tile is None:
                continue
            row = i // tiles_per_row
            col = i % tiles_per_row
            y1, y2 = row * tile_h, (row + 1) * tile_h
            x1, x2 = col * tile_w, (col + 1) * tile_w
            canvas[y1:y2, x1:x2] = tile

        # Draw merged boxes
        for box in merged_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Save final stitched image
        cv2.imwrite(os.path.join(merged_dir, f"{orig_name}.png"), canvas)
        print(f"✅ Saved: {orig_name}.png")

    print(f"\n📁 All merged images saved to: {merged_dir}")

    # Optional: Run YOLO evaluation if needed
    metrics = model.val(
        data="data.yaml",
        split="test",
        imgsz=(736, 414),
        save_json=True,
        name="test_eval",
        save=True,
        batch=1,
        amp=True
    )

    print(f"\n📊 Evaluation:")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall: {metrics.box.mr:.3f}")
    f1 = (2 * metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-6)
    print(f"F1 Score: {f1:.3f}")


if __name__ == "__main__":
    run_prediction_on_test()
