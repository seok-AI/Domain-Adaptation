import torch
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import random
import warnings
import torch.nn.functional as F
import timm

warnings.filterwarnings(action='ignore') 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device) 
print('Count of using GPUs:', torch.cuda.device_count())
print('Current cuda device:', torch.cuda.current_device())  
## Hyperparameter Settings
CFG = {
    'IMG_SIZE':384,
    'EPOCHS':4, #Your Epochs,
    'LR':1e-4, #Your Learning Rate,
    'BATCH_SIZE':4, #Your Batch Size,
    'SEED':41
}

## Fixed Random-Seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정

## Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['img_path']
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # mos column 존재 여부에 따라 값을 설정
        mos = float(self.dataframe.iloc[idx]['mos']) if 'mos' in self.dataframe.columns else 0.0
        comment = self.dataframe.iloc[idx]['comments'] if 'comments' in self.dataframe.columns else ""
        
        return img, mos
    
## Define Model
class Swinv2(nn.Module):
    def __init__(self):
        super(Swinv2, self).__init__()

        self.backbone = timm.create_model("swinv2_large_window12to24_192to384", pretrained=True)
        self.backbone.head.fc = nn.Identity()
        
        # Image quality assessment head
        self.regression_head = nn.Linear(1536, 1)


    def forward(self, images, captions=None):
        # Swin V2
        features_vit = self.backbone(images)
        
        # Image quality regression
        mos = self.regression_head(features_vit)
        
        return mos
    
path = '/home/6210seok/data/Samsung-Image-Quality/'

# 데이터 로드
train_data = pd.read_csv(path + 'train.csv')
test_data = pd.read_csv(path + 'test.csv')

def rewrite(df):
    df['img_path'] = path + df['img_path']
    return df

train_data = rewrite(train_data).drop_duplicates(['img_name']).reset_index(drop=True)
data = train_data.copy()
test_data = rewrite(test_data)
val_data = train_data[50000:].reset_index(drop=True)
train_data = train_data[:50000]

## Train
def val(model, criterion1, val_loader, epoch):
    
    model.eval()
    val_loss = 0
    
    val_loop = tqdm(val_loader, leave=True)
    with torch.no_grad():
        for imgs, mos in val_loop:
            imgs, mos = imgs.float().to(device), mos.float().to(device)
            # Forward & Loss
            predicted_mos = model(imgs)
            loss = criterion1(predicted_mos.squeeze(1), mos)
            
            val_loss += loss.item()
            val_loop.set_description(f"Validation")
            val_loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1} val average loss: {val_loss / len(val_loader):.4f}")
    
def train(model, criterion1, optimizer, train_loader, val_loader):
    # 학습
    for epoch in range(2):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, leave=True)
        for imgs, mos in loop:
            imgs, mos = imgs.float().to(device), mos.float().to(device)
            
            # Forward & Loss
            predicted_mos = model(imgs)
            # loss0 = criterion0(predicted_mos, mos)
            loss = criterion1(predicted_mos.squeeze(1), mos)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            total_loss += loss.item()
            loop.set_description(f"Train")
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1} finished with average loss: {total_loss / len(train_loader):.4f}")
        
        val(model, criterion1, val_loader, epoch)
        
# 데이터셋 및 DataLoader 생성
transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])), 
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# 모델, 손실함수, 옵티마이저
model = Swinv2()
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model, output_device=0)

model.to(device)

criterion1 = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)

train_dataset = CustomDataset(train_data, transform)
train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=1)

val_dataset = CustomDataset(val_data, transform)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=1)

train(model, criterion1, optimizer, train_loader, val_loader)
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['LR']/100, weight_decay=1e-5)
train(model, criterion1, optimizer, train_loader, val_loader)

# torch.save(model.state_dict(), 'Only-SWIN-HUGE-41.pt')

model.eval()
predicted_mos_list = []
predicted_comments_list = []


def greedy_decode(model, image, max_length=50):
    image = image.unsqueeze(0).to(device)
    mos = model(image)
        
    return mos.item()
    
# 추론 과정
with torch.no_grad():
    for imgs, _ in tqdm(val_loader):
        for img in imgs:
            img = img.float().to(device)
            mos= greedy_decode(model, img)
            predicted_mos_list.append(mos)
            
# 결과 저장
result_df = pd.DataFrame({
    'img_name': val_data['img_name'],
    'mos': predicted_mos_list
})

result_df.to_csv('val-huge-41.csv', index=False)

import matplotlib.pyplot as plt

x = np.linspace(2.5,8)
plt.plot(result_df['mos'], val_data['mos'], 'g*')
plt.plot(x,x)
plt.show()
x_simple = result_df['mos']
y_simple = val_data['mos']
my_rho = np.corrcoef(x_simple, y_simple)[0][1]

from scipy import stats
res = stats.spearmanr(x_simple, y_simple).statistic

print(f"PLCC: {my_rho:.5f}, SRCC: {res:.5f}\nScore: {(my_rho + res) / 2:.5f}")

## Inference & Submit
test_dataset = CustomDataset(test_data, transform)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=1)

model.eval()
predicted_mos_list = []

def greedy_decode(model, image, max_length=50):
    image = image.unsqueeze(0).to(device)
    mos = model(image)
        
    return mos.item()
    
# 추론 과정
with torch.no_grad():
    for imgs, _ in tqdm(test_loader):
        for img in imgs:
            img = img.float().to(device)
            mos = greedy_decode(model, img)
            predicted_mos_list.append(mos)

# 결과 저장
result_df = pd.DataFrame({
    'img_name': test_data['img_name'],
    'mos': predicted_mos_list
})

result_df['comments'] = 'Nice Image.'

result_df.to_csv('SWIN-V2-HUGE-41.csv', index=False)

print("Inference completed and results saved to submit.csv.")