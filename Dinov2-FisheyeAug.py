## Import
import os
import cv2
from PIL import Image
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
# import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "2"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Device:', device) 
print('Count of using GPUs:', torch.cuda.device_count())
print('Current cuda device:', torch.cuda.current_device())  

# Your Path
DATA_PATH = "/mnt/homes/6210seok/Samsung-Camera-Invariant-Domain-Adaptation"
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
## Utils
# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
## Custom Dataset
# class CustomDataset(Dataset):
#     def __init__(self, csv_file, transform=None, infer=False):
#         self.data = csv_file
#         self.transform = transform
#         self.infer = infer

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img_path = DATA_PATH + self.data.iloc[idx, 1][1:]
#         original_image = cv2.imread(img_path)
#         original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
#         mask_path = DATA_PATH + self.data.iloc[idx, 2][1:]
#         original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
#         original_mask[original_mask == 255] = 12
        
#         if self.transform:
#             augmented = self.transform(image=original_image, mask=original_mask)
#             image, mask = torch.tensor(augmented['image']), torch.LongTensor(augmented['mask'])

#         image = image.permute(2,0,1)

#         return image, mask, original_mask
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = csv_file
        self.transform = transform
        self.infer = infer
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        a, b, c, d, dist = 0, 1, random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3), random.uniform(-0.1, 0.1)
        img_path = DATA_PATH + self.data.iloc[idx, 1][1:]
        original_image = cv2.imread(img_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        mask_path = DATA_PATH + self.data.iloc[idx, 2][1:]
        original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        original_mask[original_mask == 255] = 12
        
        mask = original_mask.copy()
        mask[mask==0]=13
        
        height, width = original_image.shape[:2]

        # 카메라 매트릭스 생성
        center_x = width / 2
        center_y = height / 2
        camera_matrix = np.array([[height/2, 0, center_x],
                                [0, width/2, center_y],
                                [0, dist, 1]], dtype=np.float32)

        # 왜곡 계수 생성
        dist_coeffs = np.array([a, b, c, d], dtype=np.float32)

        # 왜곡 보정
        undistorted_image = cv2.undistort(original_image, camera_matrix, dist_coeffs)
        undistorted_mask = cv2.undistort(mask, camera_matrix, dist_coeffs)
        
        undistorted_mask[undistorted_mask==0]=12
        undistorted_mask[undistorted_mask==13]=0
        
        fisheye = A.augmentations.crops.transforms.CenterCrop (1024, 1024, p=1.0)(image = undistorted_image, mask = undistorted_mask)
        undistorted_image, undistorted_mask = fisheye['image'], fisheye['mask']
        
        if self.transform:
            augmented = self.transform(image=undistorted_image, mask=undistorted_mask)
            image, mask = torch.tensor(augmented['image']), torch.LongTensor(augmented['mask'])

        image = image.permute(2,0,1)

        return image, mask, undistorted_mask
## Data Loader
ADE_MEAN = np.array([0.485,0.456,0.406])
ADE_STD = np.array([0.229,0.224,0.225])

train_transform = A.Compose([
    # A.Resize(width = CFG['IMG_SIZE'], height = CFG['IMG_SIZE']),
    A.Resize(width = 800, height = 800),
    A.augmentations.geometric.transforms.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.augmentations.transforms.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.2),
    A.augmentations.geometric.rotate.Rotate(limit=15, p=0.2),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD)
])
transform = A.Compose([
    # A.Resize(width = CFG['IMG_SIZE'], height = CFG['IMG_SIZE']),
    A.Resize(width = 800, height = 800),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD)
])

def collate_fn(inputs):
    batch = dict()
    batch["pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
    batch["labels"] = torch.stack([i[1] for i in inputs], dim=0)
    batch["original_segmentation_maps"] = torch.stack([torch.from_numpy(i[2]) for i in inputs], dim=0)

    return batch

train_data = pd.read_csv(DATA_PATH + '/train_source.csv')
val_data = pd.read_csv(DATA_PATH + '/val_source.csv')
train_data = pd.concat([train_data, val_data], axis=0).reset_index(drop=True)
## Data Loader
train_dataset = CustomDataset(csv_file=train_data, transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=4, collate_fn=collate_fn)

# val_dataset = val_CustomDataset(csv_file=val_data, transform=transform)
# val_dataloader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=4, collate_fn=collate_fn)
## Define Model
from monai.losses import GeneralizedDiceFocalLoss
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        bs = targets.size()[0]
        one_hot_gt = torch.zeros((bs, 13, 800, 800)).to(device)

        for class_idx in range(13):
            for i in range(bs):
                # class_mask = (targets == class_idx)  # 클래스 레이블과 일치하는 픽셀을 True로 하는 마스크
                one_hot_gt[i][class_idx][targets[i]==class_idx] = 1
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.softmax(inputs, dim=1)
        #flatten label and prediction tensors
        # inputs = inputs.squeeze()
        # # targets = targets.view(-1)
        # one_hot_gt = one_hot_gt.squeeze()

        intersection = (inputs * one_hot_gt).sum()
        total = (inputs + one_hot_gt).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
        
        return 1 - IoU 
import torch
from transformers import Dinov2Model, Dinov2PreTrainedModel
# from transformers import AutoModel
from transformers.modeling_outputs import SemanticSegmenterOutput

class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=7):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        # self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1,1))
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, 1024, (1,1)),
            nn.ReLU(),
            nn.Conv2d(1024, 512, (1,1)),
            nn.ReLU(),
            nn.Conv2d(512, num_labels, (1,1))
        )
        

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0,3,1,2)

        return self.classifier(embeddings)


class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
  def __init__(self, config):
    super().__init__(config)

    self.dinov2 = Dinov2Model(config)
    # print(config.hidden_size)
    # self.classifier = LinearClassifier(config.hidden_size, CFG['IMG_SIZE']//14, CFG['IMG_SIZE']//14, config.num_labels)
    self.classifier = LinearClassifier(config.hidden_size, 800//14, 800//14, config.num_labels)
    
  def forward(self, pixel_values, output_hidden_states=False, output_attentions=False, labels=None):
    # use frozen features
    outputs = self.dinov2(pixel_values,
                            output_hidden_states=output_hidden_states,
                            output_attentions=output_attentions)
    
    # get the patch embeddings - so we exclude the CLS token
    patch_embeddings = outputs.last_hidden_state[:,1:,:]
    # convert to logits and upsample to the size of the pixel values
    
    
    
    logits = self.classifier(patch_embeddings)
    logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)
    # print(torch.softmax(logits, dim=1).squeeze().size(),  labels.squeeze().size())
    
    loss = None
    if labels is not None:
      # important: we're going to use 0 here as ignore index instead of the default -100
      # as we don't want the model to learn to predict background
      # loss_fct = FocalLoss()
      # loss_fct = torch.nn.CrossEntropyLoss()
      loss_fct1 = GeneralizedDiceFocalLoss(sigmoid=True, to_onehot_y=True)
      bs = labels.size()[0]
      loss1 = loss_fct1(logits.squeeze(), torch.reshape(labels, (bs,1,800,800)))
      
      loss_fct2 = IoULoss()
      loss2 = loss_fct2(logits.squeeze(), labels.squeeze())
      
      loss_fct3 = torch.nn.CrossEntropyLoss()
      loss3 = loss_fct3(logits.squeeze(), labels.squeeze())
      
      loss = (loss1 + loss2 + 2 * loss3) / 4
      
    return SemanticSegmenterOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
import transformers
transformers.__version__
## Model Load
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
## Train
SMOOTH = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, num_class=13):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    bs = outputs.shape[0]
    intersection = torch.zeros((num_class)).to(device)
    union = torch.zeros((num_class)).to(device)

    for idx in range(num_class):
        intersection[idx] += ((outputs == idx) & (labels == idx)).float().sum()
        union[idx] += ((outputs == idx) | (labels == idx)).float().sum()
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded, intersection, union  # Or thresholded.mean() if you are interested in average across the batch
def val(model, val_dataloader):
    model.eval()
    val_loss = 0
    miou = []
    miou_inter = torch.zeros(13).to(device)
    miou_union = torch.zeros(13).to(device)
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
                pred = pred.resize((1024, 1024), Image.NEAREST)
                P.append(np.array(pred))
                # print(np.array(pred).shape)
            # print(batch['original_segmentation_maps'])
            # mIOU metric
            Pred = torch.stack([torch.from_numpy(i) for i in P], dim=0)
            
            batch_miou, batch_inter, batch_union = iou_pytorch(Pred.to(device), batch['original_segmentation_maps'].to(device))
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
        for i in range(13):
            print(f'class {i}\'s iou: {total_iou[i].item():.5f}')
def train(model, optimizer, train_dataloader, val_dataloader=None):
    for epoch in range(2):
        model.train()
        total_loss = 0
        
        loop = tqdm(enumerate(train_dataloader), total=len(train_data) // CFG['BATCH_SIZE'])
        for idx, batch in loop:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            # forward pass
            outputs = model(pixel_values, labels=labels)
            loss = outputs.loss
            # print(loss)
            loss.backward()
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()
            
            loop.set_description(f"Train")
            loop.set_postfix(loss=loss.item())
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch + 1} finished with average loss: {total_loss / len(train_dataloader):.4f}")
        if val_dataloader != None:
            val(model, val_dataloader)
from torch.optim import AdamW
from tqdm.auto import tqdm
optimizer = AdamW(model.parameters(), lr=CFG['LR'], weight_decay=1e-5)
train(model, optimizer, train_dataloader)
optimizer = AdamW(model.parameters(), lr=CFG['LR']/10, weight_decay=1e-5)
train(model, optimizer, train_dataloader)
torch.save(model.state_dict(), 'dinov2-giant-all.pt')
class CustomDataset_val(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = DATA_PATH + self.data.iloc[idx, 1][1:]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
                image = image.transpose(2,0,1)
            return image

test_dataset = CustomDataset_val(csv_file=DATA_PATH+'/test.csv', transform=transform, infer=True)
test_dataloader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=4)
width = 960
height = 540
img = np.ones((540,960)) * 255
# 원형 마스크 생성하기 (원의 중심은 이미지의 중심, 반지름은 이미지 너비의 절반으로 설정)
mask = np.zeros((height,width), np.uint8)
cv2.circle(mask,(width//2,height//2),radius=int(width//2.2),color=(255,255,255),thickness=-1)
# 원형 마스크 적용하기 (bitwise 연산을 사용)
masked_data = cv2.bitwise_and(img, img, mask=mask)
# 결과 보여주기
plt.imshow(masked_data, cmap='gray', vmin=0, vmax=255)
width = 960
height = 540
img = np.ones((540,960)) * 255
# 원형 마스크 생성하기 (원의 중심은 이미지의 중심, 반지름은 이미지 너비의 절반으로 설정)
mask = np.zeros((height,width), np.uint8)
cv2.circle(mask,(width//2,0),radius=int(width//1.75),color=(255,255,255),thickness=-1)

# 원형 마스크 적용하기 (bitwise 연산을 사용)
masked_data2 = cv2.bitwise_and(img, img, mask=mask)

# 결과 보여주기
plt.imshow(masked_data2, cmap='gray', vmin=0, vmax=255)

with torch.no_grad():
    model.eval()
    result = []
    c=0
    for images in tqdm(test_dataloader):
        plt.imshow(images[1].permute(1,2,0))
        plt.show()
        images = images.float().to(device)
        outputs = model(images).logits
        outputs = torch.softmax(outputs, dim=1).cpu()
        outputs = torch.argmax(outputs, dim=1).numpy()
        # batch에 존재하는 각 이미지에 대해서 반복
        for pred in outputs:
            pred = pred.astype(np.uint8)
            pred = Image.fromarray(pred) # 이미지로 변환
            pred = pred.resize((960, 540), Image.NEAREST) # 960 x 540 사이즈로 변환
            
            pred = np.array(pred) # 다시 수치로 변환
            
            pred = torch.from_numpy(pred).to(device)
            
            pred[masked_data==0] = 12
            pred[masked_data2==0] = 12
            pred = np.array(pred.cpu())
        plt.imshow(pred)
        plt.show()
        if c==10:
            break
        c+=1
            
            # # class 0 ~ 11에 해당하는 경우에 마스크 형성 / 12(배경)는 제외하고 진행
            # for class_id in range(12):
            #     class_mask = (pred == class_id).astype(np.uint8)
            #     if np.sum(class_mask) > 0: # 마스크가 존재하는 경우 encode
            #         mask_rle = rle_encode(class_mask)
            #         result.append(mask_rle)
            #     else: # 마스크가 존재하지 않는 경우 -1
            #         result.append(-1)

with torch.no_grad():
    model.eval()
    result = []
    c=0
    for images in tqdm(test_dataloader):
        images = images.float().to(device)
        
        outputs = model(images).logits
        outputs = torch.softmax(outputs, dim=1).cpu()
        outputs = torch.argmax(outputs, dim=1).numpy()
        # batch에 존재하는 각 이미지에 대해서 반복
        for pred in outputs:
            pred = pred.astype(np.uint8)
            pred = Image.fromarray(pred) # 이미지로 변환
            pred = pred.resize((960, 540), Image.NEAREST) # 960 x 540 사이즈로 변환
            
            pred = np.array(pred) # 다시 수치로 변환
            
            pred = torch.from_numpy(pred).to(device)
            
            pred[masked_data==0] = 12
            pred[masked_data2==0] = 12
            pred = np.array(pred.cpu())
            
            # class 0 ~ 11에 해당하는 경우에 마스크 형성 / 12(배경)는 제외하고 진행
            for class_id in range(12):
                class_mask = (pred == class_id).astype(np.uint8)
                if np.sum(class_mask) > 0: # 마스크가 존재하는 경우 encode
                    mask_rle = rle_encode(class_mask)
                    result.append(mask_rle)
                else: # 마스크가 존재하지 않는 경우 -1
                    result.append(-1)
## Submission
submit = pd.read_csv(DATA_PATH + '/sample_submission.csv')
submit['mask_rle'] = result
submit.to_csv('./dino-all.csv', index=False)
submit
print('done.')