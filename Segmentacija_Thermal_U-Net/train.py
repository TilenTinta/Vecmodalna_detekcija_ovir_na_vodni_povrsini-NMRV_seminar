import argparse
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from network import UNet
import matplotlib.pyplot as plt
import numpy as np

from dataloder import get_dataloaders

# Inicializacija uteži - dodana zaradi čudnega loss-a in sem mislil da je to krivo
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)


if __name__ == "__main__":

    # Root pot
    root_path = '..\\NMRV_seminar'

    # Weights save
    weights_path = ".\\"

    # Argument parser
    options = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    options.add_argument('--dataroot', default=root_path, help='root directory of the dataset')
    options.add_argument('--batchsize', type=int, default=4, help='input batch size')
    # options.add_argument('--imagesize', type=int, default=(2208, 1242), help='size of the image (height, width)') # polna resolucija
    options.add_argument('--imagesize', type=int, default=(512, 288), help='size of the image (height, width)') # scaled down resolucija (512 sm določim in delil osnovno resolucijo, ohranil aspect)
    options.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    options.add_argument('--lr', type=float, default=0.0001, help='learning rate') 
    opt = options.parse_args()

    # Dataloaders - vrne: train, validacijo in test ki je ne rabim zdej
    train_loader, val_loader, _ = get_dataloaders(opt.dataroot, opt.batchsize, opt.imagesize)

    # Preveri če ma PC GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Code will run on:", device)

    model = UNet(in_channels=3, out_channels=4).to(device) # Init mreže
    initialize_weights(model)
    criterion = nn.CrossEntropyLoss(ignore_index=255) # Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr) # Optimizer (za backprop)

    # Training loop
    train_losses = []
    iou_scores = []
    best_iou = 0
    patience_limit = 10
    patience_counter = 0

    for epoch in range(opt.epochs):
        print(f"Training epoch: {epoch + 1}/{opt.epochs}")
        model.train() # Postaviš model v učenje
        running_loss = 0

        for batch in tqdm(train_loader):
            images, masks, _ = batch
            images = images.to(device).float()
            masks = masks.to(device).long()

            optimizer.zero_grad() # stare gradiente optimizatorja ki so poračunani daš na 0
            outputs = model(images) # forward

            #print(f"Output shape: {outputs.shape}, Mask shape: {masks.shape}")
            # Output shape: torch.Size([4, 4, 512, 288]), Mask shape: torch.Size([4, 512, 288]): [batch, class, sirina, visina]

            loss = criterion(outputs, masks) # gleda napoved z realnimi vrednostmi (CrossEntropyLoss), že ima softmax
            running_loss += loss.item() # loss
            loss.backward() # backpropagation (računa gradiente)
            optimizer.step() # posodabljanje uteži

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Average Training Loss: {avg_loss:.6f}")

        # Test modela
        print("Testing the model...")
        model.eval() # Postaviš model v evaluacijo
        intersection, union = 0, 0

        with torch.no_grad():
            for batch in tqdm(val_loader):
                images, masks, _ = batch
                images = images.to(device).float()
                masks = masks.to(device).float()

                outputs = torch.softmax(model(images), dim=1) # softmax nad izhodom
                predictions = torch.argmax(outputs, dim=1)  # logits v oznake razredov
                
                valid_mask = (masks != 255)  # ignoriraj razrede Water(hole), Ignore in Recording Boat
                intersection += ((predictions == masks) & valid_mask).sum().float().item()
                union += valid_mask.sum().float().item()

        iou_per_class = []
        for class_id in range(4):  # Razredi: 0-4
            intersection = ((predictions == class_id) & (masks == class_id)).sum().item()
            union = ((predictions == class_id) | (masks == class_id)).sum().item()
            if union > 0:
                iou_per_class.append(intersection / union)

        iou = np.mean(iou_per_class) if iou_per_class else 0

        iou_scores.append(iou)
        print(f"mIoU: {iou:.6f}")

        # Shrani najboljše uteži
        if iou > best_iou:
            best_iou = iou
            torch.save(model.state_dict(), os.path.join(weights_path, "unet_thermal_best.pth"))
            print("Model saved with best IoU!")
            patience_counter = 0 
        else:
            # Če se določeno število zaporednih epoh nič ne spremeni končaj - early stop
            patience_counter += 1
            if patience_counter >= patience_limit:
                print("Early stopping triggered.")
                break

    # Graf loss-a in IoU-ja
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, iou_scores, label='IoU Score')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Training Loss and IoU Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(opt.dataroot, 'metrics_plot.png'))
    plt.show()

    print("Training Complete. Best IoU:", best_iou)

