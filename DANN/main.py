import os
import cv2
from PIL import Image
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim import AdamW
from tqdm.auto import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from monai.losses import GeneralizedDiceFocalLoss

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Device:', device) 
print('Count of using GPUs:', torch.cuda.device_count())
print('Current cuda device:', torch.cuda.current_device())  

# Your Path
DATA_PATH = "/mnt/homes/6210seok/Samsung-Camera-Invariant-Domain-Adaptation"

from utils_ import *

CFG = {
    'IMG_SIZE':800,
    'EPOCHS':5, #Your Epochs,
    'LR':1e-4, #Your Learning Rate,
    'BATCH_SIZE':2, #Your Batch Size,
    'SEED':41
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정


ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255.
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255.

transform = A.Compose([
    A.Resize(width = CFG['IMG_SIZE'], height = CFG['IMG_SIZE']),
    A.augmentations.geometric.transforms.HorizontalFlip(p=0.5),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD)
])

train_data = pd.read_csv(DATA_PATH + '/train_source.csv')
train_target_data = pd.read_csv(DATA_PATH + '/train_target.csv')[:2194]

train_target_data['domain_label'] = [0]*len(train_target_data)
train_data['domain_label'] = [1]*len(train_data)

val_data = pd.read_csv(DATA_PATH + '/val_source.csv')

train_dataset = train_CustomDataset(csv_file=train_data, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=4, collate_fn=train_collate_fn)

train_target_dataset = target_CustomDataset(csv_file=train_target_data, transform=transform)
train_target_dataloader = DataLoader(train_target_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=4, collate_fn=target_collate_fn)

val_dataset = val_CustomDataset(csv_file=val_data, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=4, collate_fn=collate_fn)

fisheye_val_dataset = fisheye_CustomDataset(csv_file=val_data, transform=transform)
fisheye_val_dataloader = DataLoader(fisheye_val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=4, collate_fn=collate_fn)


    
# model 초기화
model = Dinov2ForSemanticSegmentation.from_pretrained("facebook/dinov2-giant", num_labels=13)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model, output_device=0)

model.to(device)
print('done.')

for name, param in model.named_parameters():
    if name.startswith("dinov2"):
        if 'dinov2.encoder.layer.39' not in name:
            param.requires_grad = False
            
def val(model, val_dataloader):
    model.eval()
    val_loss = 0
    miou = []
    miou_inter = torch.zeros(12).to(device)
    miou_union = torch.zeros(12).to(device)
    with torch.no_grad():
        loop = tqdm(enumerate(val_dataloader), total=len(val_data) // CFG['BATCH_SIZE'])
        for idx, batch in loop:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            # forward pass
            outputs = model(pixel_values, labels=labels)
            loss = outputs.loss
            
            preds = torch.softmax(outputs.logits, dim=1).cpu()
            preds = torch.argmax(preds, dim=1).numpy()
            P = []
            for pred in preds:
                pred = pred.astype(np.uint8)
                pred = Image.fromarray(pred) # 이미지로 변환
                pred = pred.resize((2048, 1024), Image.NEAREST)
                P.append(np.array(pred))
                # print(np.array(pred).shape)
            # print(batch['original_segmentation_maps'])
            # mIOU metric
            Pred = torch.stack([torch.from_numpy(i) for i in P], dim=0)
            
            batch_miou, batch_inter, batch_union = iou_pytorch(Pred.to(device), batch['original_segmentation_maps'].to(device), num_class=12)
            miou.append(batch_miou.mean())
            miou_inter += batch_inter
            miou_union += batch_union
            val_loss += loss.item()
            loop.set_description(f"Validation")
            loop.set_postfix(mIOU=batch_miou.mean().item())
            
        print(f"Val loss: {val_loss / len(val_dataloader):.4f}\n")

        total_iou = (miou_inter + 1) / (miou_union + 1)
        print(f"Batch Mean iou: {torch.tensor(miou).mean().item()}\n")
        print(f"Mean iou: {total_iou.mean().item()}")
        print(f"Mean iou(w/o background): {total_iou[:(13-1)].mean().item()}\n")
        for i in range(12):
            print(f'class {i}\'s iou: {total_iou[i].item():.5f}')
            
def train(model, optimizer, train_dataloader, val_dataloader=None):
    for epoch in range(2):
        model.train()
        total_loss = 0
        alpha = 1
        
        loop = tqdm(range(len(train_dataloader)))
        for step in loop:
            batch =[]
            batch.append(iter(train_target_dataloader).next())
            batch.append(iter(train_dataloader).next())

            for i in range(2):
                if i==1:
                    pixel_values = batch[i]["pixel_values"].to(device)
                    labels = batch[i]["labels"].to(device)
                    domain_labels = batch[i]["domain_labels"].to(device)
                    outputs = model(pixel_values, labels=labels, domain_labels=domain_labels)
                else:
                    pixel_values = batch[i]["pixel_values"].to(device)
                    domain_labels = batch[i]["domain_labels"].to(device)
                    outputs = model(pixel_values, domain_labels=domain_labels)
                    
                loss = outputs.loss
                # print(loss)
                loss.backward()
                optimizer.step()
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                loop.set_description(f"Train")
                if outputs.class_loss != None:
                    loop.set_postfix(loss=outputs.class_loss.item())
                    total_loss += outputs.class_loss.item()
                else:
                    loop.set_postfix(loss=loss.item())
        print(f"Epoch {epoch + 1} finished with average loss: {total_loss / len(train_dataloader):.4f}")
        if val_dataloader != None:
            val(model, val_dataloader)



   
optimizer = AdamW(model.parameters(), lr=CFG['LR'], weight_decay=1e-5)
train(model, optimizer, train_dataloader, val_dataloader)

optimizer = AdamW(model.parameters(), lr=CFG['LR']/10, weight_decay=1e-5)
train(model, optimizer, train_dataloader, val_dataloader)

torch.save(model.state_dict(), 'dinov2-dann.pt')