import os
from glob import glob
from ultralytics import YOLO

def run_prediction_on_test():
    model_path = "yolo_ladica/OvireNaVodi/weights/best.pt" # POPRAVI NA USTREZEN MODEL !!!
    model = YOLO(model_path)

    # Pot do test slik iz YAML 
    image_dir = "datasets/test/images"
    image_paths = glob(os.path.join(image_dir, "*.jpg")) + glob(os.path.join(image_dir, "*.png"))

    # Mape za izhod
    image_save_dir = "runs/detect/test_images"
    label_save_dir = "runs/detect/test_labels"

    # Izvedi detekcijo
    results = model.predict(
        source=image_paths,
        conf=0.5,
        save=True,
        save_txt=True,
        #project="runs/detect",
        name="test_images",
        line_width=1
    )

    # Premakni .txt datoteke iz privzete mape v ločeno mapo
    pred_dir = os.path.join("runs/detect/test_images", "labels")
    os.makedirs(label_save_dir, exist_ok=True)

    for txt_file in glob(os.path.join(pred_dir, "*.txt")):
        base = os.path.basename(txt_file)
        new_path = os.path.join(label_save_dir, base)
        os.replace(txt_file, new_path)

    print("Detekcije in oznake uspešno shranjene.")

    metrics = model.val(
        data="data.yaml",
        split="test", 
        save_json=True,
        name="test_eval",
        save=True
    )

    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall: {metrics.box.mr:.3f}")
    f1 = (2 * metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-6)
    print(f"F1 Score: {f1:.3f}")


################## MAIN ################## 
if __name__ == "__main__":
    run_prediction_on_test()
