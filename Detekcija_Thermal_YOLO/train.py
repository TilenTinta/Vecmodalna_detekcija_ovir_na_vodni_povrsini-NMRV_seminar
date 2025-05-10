from ultralytics import YOLO
import os
import glob
import numpy as np
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

def main():

    for label_file in glob.glob("datasets/train/labels/*.txt"):
        data = np.loadtxt(label_file)
        if np.isnan(data).any():
            print(f"NaN found in {label_file}")

    # Prednaučen YOLO11 model, verzije:
    # - yolov11n.pt (nano, smallest and fastest)
    # - yolov11s.pt (small)
    # - yolov11m.pt (medium)
    # - yolov11l.pt (large)
    # - yolov11x.pt (xlarge, largest and most accurate)
    
    model = YOLO("yolo11s.pt")

    # Train the model on the custom dataset
    model.train(
        amp=False,
        data="data.yaml",           # File za dataset
        imgsz=736,                  # max. velikost ki jo zahtevamo
        epochs=300,                 # Št. epoch
        batch=12,                   # Velikost batcha
        lr0 = 0.001,                # Initial learning rate
        lrf=0.01,                   # Learning Rate Factor
        cos_lr = False,             # linearno ali kosinusno zmanjšanje
        patience=15,                # Early stop trigger
        device="cuda",              # Rabi GPU
        project="yolo_ladica",      # Directory za training loge
        name="OvireNaVodi",         # Sub-folder name
        workers=8,                  # Št. corov za loadat podatke
        show=True,                  # Prikaži rezultate
        save=True,                  # Shrani naučene uteži

        # Augmentacija
        hsv_h=0.2,                  # Hue
        hsv_s=0.2,                  # Saturation
        hsv_v=0.2,                  # Brightnes
        degrees=15,                 # Rotacija
        translate=0.1,              # Translacija
        scale=0.1,                  # Skaliranje
        fliplr=0.5,                 # Flipanje horizontalno
        erasing=0.1,                # Random prekrivanje slike
        gaussian=0.1                # Gaussian noise
    )

if __name__ == "__main__":
    main()
