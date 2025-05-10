import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
import cv2
import numpy as np
from dataloder import get_dataloaders
from network import UNet
from tqdm import tqdm

def main():
    # Parametri
    data_root = '.\\'
    dataset_path = '..\\NMRV_seminar'
    output_folder = os.path.join(data_root, 'test_results')
    model_weights = os.path.join(data_root, 'unet_thermal_best.pth')
    image_size = (512, 288)
    #image_size = (2208, 1242)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Code will run on:", device)

    # Nardi output mapo če je ni
    os.makedirs(output_folder, exist_ok=True)

    # Naloži model
    model = UNet().to(device) # Init SegFormerja
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.eval()

    # Dataloader
    _, _, test_loader = get_dataloaders(dataset_path, batch_size=1, size=image_size)

    # Test slik
    intersection, union = 0, 0
    iou_scores = []

    # Testing loop
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images, masks, file_names = batch
            images = images.to(device).float()
            masks = masks.to(device).long()  # Ensure integer masks
            if images is None or masks is None:
                print(f"Skipping missing file(s) in batch: {file_names}")
                continue  # Skip missing files

            # Get model predictions
            outputs = torch.softmax(model(images), dim=1)  # UPDATED: softmax for multi-class
            predictions = torch.argmax(outputs, dim=1)  # UPDATED: Get class with highest probability

            # Ignore class labels 6 and 7 in evaluation
            valid_mask = (masks != 255)

            # Compute IoU per class
            iou_per_class = []
            for class_id in range(4):  # Only for valid classes (0-3)
                class_intersection = ((predictions == class_id) & (masks == class_id) & valid_mask).sum().item()
                class_union = ((predictions == class_id) | (masks == class_id) & valid_mask).sum().item()
                if class_union > 0:
                    iou_per_class.append(class_intersection / class_union)

            # Compute mean IoU for this image
            batch_iou = np.mean(iou_per_class) if iou_per_class else 0
            iou_scores.append(batch_iou)

            # Save overlayed prediction
            for i in range(images.shape[0]):
                image = images[i].cpu().permute(1, 2, 0).numpy() * 255
                image = image.astype(np.uint8)

                # mask = predictions[i].cpu().numpy() * (255 // 3)  # Normalize to 0-255
                # mask = mask.astype(np.uint8)

                # # Overlay z 50% transparency
                # overlay = cv2.addWeighted(image, 0.5, cv2.applyColorMap(mask, cv2.COLORMAP_TURBO), 0.5, 0)

                class_colors = {
                    0: (255, 255, 0),  # Nebo
                    1: (0, 0, 255),    # Voda
                    2: (0, 255, 0),    # Premikajoče ovire
                    3: (255, 0, 0)     # Breg
                }

                # Create RGB mask
                mask = predictions[i].cpu().numpy()
                color_mask = np.zeros_like(image)

                for class_id, color in class_colors.items():
                    color_mask[mask == class_id] = color

                # Blend original image with color mask
                overlay = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)

                # Save image
                output_path = os.path.join(output_folder, file_names[i])
                cv2.imwrite(output_path, overlay)

    # IoU score
    final_iou = np.mean(iou_scores)
    print(f"Final IoU Score: {final_iou:.6f}")

if __name__ == '__main__':
    main()
