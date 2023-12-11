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
from monai.losses import GeneralizedDiceFocalLoss

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CFG = {
    'IMG_SIZE':800,
    'EPOCHS':5, #Your Epochs,
    'LR':1e-4, #Your Learning Rate,
    'BATCH_SIZE':2, #Your Batch Size,
    'SEED':41
}

# your path
DATA_PATH = "/mnt/homes/6210seok/Samsung-Camera-Invariant-Domain-Adaptation"

def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

class train_CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = csv_file
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        domain_label = self.data.iloc[idx, 3]
        img_path = DATA_PATH + self.data.iloc[idx, 1][1:]
        original_image = cv2.imread(img_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        mask_path = DATA_PATH + self.data.iloc[idx, 2][1:]
        original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        original_mask[original_mask == 255] = 12
        
        if self.transform:
            augmented = self.transform(image=original_image, mask=original_mask)
            image, mask = torch.tensor(augmented['image']), torch.LongTensor(augmented['mask'])

        image = image.permute(2,0,1)
        return image, mask, original_mask, np.array(domain_label)
    
class target_CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = csv_file
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        domain_label = self.data.iloc[idx, 2]
        img_path = DATA_PATH + self.data.iloc[idx, 1][1:]
        original_image = cv2.imread(img_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=original_image)
            image= torch.tensor(augmented['image'])

        image = image.permute(2,0,1)
        
        return image, np.array(domain_label)
    
class val_CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = csv_file
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = DATA_PATH + self.data.iloc[idx, 1][1:]
        original_image = cv2.imread(img_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        mask_path = DATA_PATH + self.data.iloc[idx, 2][1:]
        original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        original_mask[original_mask == 255] = 12
        
        if self.transform:
            augmented = self.transform(image=original_image, mask=original_mask)
            image, mask = torch.tensor(augmented['image']), torch.LongTensor(augmented['mask'])

        image = image.permute(2,0,1)

        return image, mask, original_mask
    
class fisheye_CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = csv_file
        self.transform = transform
        self.infer = infer
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
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
                                [0, 0, 1]], dtype=np.float32)

        # 왜곡 계수 생성
        dist_coeffs = np.array([0, 1, 0, 0], dtype=np.float32)

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


def collate_fn(inputs):
    batch = dict()
    batch["pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
    batch["labels"] = torch.stack([i[1] for i in inputs], dim=0)
    batch["original_segmentation_maps"] = torch.stack([torch.from_numpy(i[2]) for i in inputs], dim=0)
    return batch

def train_collate_fn(inputs):
    batch = dict()
    batch["pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
    batch["labels"] = torch.stack([i[1] for i in inputs], dim=0)
    batch["original_segmentation_maps"] = torch.stack([torch.from_numpy(i[2]) for i in inputs], dim=0)
    batch["domain_labels"] = torch.stack([torch.from_numpy(i[3]).float() for i in inputs], dim=0)
    return batch

def target_collate_fn(inputs):
    batch = dict()
    batch["pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
    batch["domain_labels"] = torch.stack([torch.from_numpy(i[1]).float() for i in inputs], dim=0)
    return batch


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, one_hot_gt, smooth=1):
        inputs = torch.softmax(inputs, dim=1)
        intersection = (inputs * one_hot_gt).sum()
        total = (inputs + one_hot_gt).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
        
        return 1 - IoU 

def MyLoss(logits, labels, label_smooth=False):
    if label_smooth:
        bs = labels.size()[0]
        one_hot_gt = torch.zeros((bs, 13, 640, 640)).to(device)*(0.1/12)

        for class_idx in range(13):
            for i in range(bs):
                one_hot_gt[i][class_idx][labels[i]==class_idx] = 0.9
    else:
        bs = labels.size()[0]
        one_hot_gt = torch.zeros((bs, 13, 640, 640)).to(device)

        for class_idx in range(13):
            for i in range(bs):
                one_hot_gt[i][class_idx][labels[i]==class_idx] = 1
    
    
    loss_fct1 = GeneralizedDiceFocalLoss(sigmoid=True)
    bs = labels.size()[0]
    loss1 = loss_fct1(logits.squeeze(), one_hot_gt.squeeze())
    
    loss_fct2 = IoULoss()
    loss2 = loss_fct2(logits.squeeze(), one_hot_gt.squeeze())
    
    loss_fct3 = torch.nn.CrossEntropyLoss()
    loss3 = loss_fct3(logits.squeeze(), one_hot_gt.squeeze())
    
    return (2 * loss1 + loss2 + loss3) / 4

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

class GradReverse(torch.autograd.Function):
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):  # 역전파 시에 gradient에 음수를 취함
        return (grad_output * -1)


class domain_classifier(nn.Module):  # classifier 수정 필요
    def __init__(self, feature_dim):
        super(domain_classifier, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 1024)
        self.fc2 = nn.Linear(1024, 1)  # source = 0, fisheye = 1 회귀 가정

    def forward(self, x):
        x = GradReverse.apply(x)  # gradient reverse
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
  def __init__(self, config):
    super().__init__(config)

    self.dinov2 = Dinov2Model(config)
    # print(config.hidden_size)
    # self.classifier = LinearClassifier(config.hidden_size, CFG['IMG_SIZE']//14, CFG['IMG_SIZE']//14, config.num_labels)
    self.classifier = LinearClassifier(config.hidden_size, CFG['IMG_SIZE']//14, CFG['IMG_SIZE']//14, config.num_labels)
    self.domain_classifier = domain_classifier(config.hidden_size)
    self.Pool = nn.AdaptiveAvgPool2d((1,1536)) 
    
  def forward(self, pixel_values, output_hidden_states=False, output_attentions=False, labels=None, domain_labels=None, alpha=1):
    # use frozen features
    outputs = self.dinov2(pixel_values,
                            output_hidden_states=output_hidden_states,
                            output_attentions=output_attentions)
    logits=None
    if domain_labels[0] == 0: # train_target
        patch_embeddings = outputs.last_hidden_state[:,1:,:]
        # convert to logits and upsample to the size of the pixel values
        reshaped_features = self.Pool(patch_embeddings).squeeze()
        domain_logits = self.domain_classifier(reshaped_features)
        
    else: # train_source
        # get the patch embeddings - so we exclude the CLS token
        patch_embeddings = outputs.last_hidden_state[:,1:,:]
        # convert to logits and upsample to the size of the pixel values
        
        logits = self.classifier(patch_embeddings)
        logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)
        
        reshaped_features = self.Pool(patch_embeddings).squeeze()
        
        domain_logits = self.domain_classifier(reshaped_features)

    loss = None
    domain_loss = None
    class_loss = None
    loss_fc = nn.BCELoss()
    if domain_labels[0] == 1:
        class_loss = MyLoss(logits, labels)
        domain_loss = loss_fc(domain_logits.squeeze(), domain_labels.squeeze())
        loss = class_loss + alpha * domain_loss
        
    if domain_labels[0] == 0:
        loss = loss_fc(domain_logits.squeeze(), domain_labels.squeeze())

    return SemanticSegmenterOutput(
            loss=loss,
            class_loss=class_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
    )