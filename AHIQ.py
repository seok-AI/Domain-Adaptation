import torch
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.ops.deform_conv import DeformConv2d

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
import torchvision
from timm.models.vision_transformer import Block
from timm.models.resnet import BasicBlock,Bottleneck

print(f"PyTorch 버전: {torch.__version__}")
print(f"torchvision 버전: {torchvision.__version__}")

CFG = {
    'IMG_SIZE':224,
    'EPOCHS':4, #Your Epochs,
    'LR':1e-4, #Your Learning Rate,
    'BATCH_SIZE':64, #Your Batch Size,
    'SEED':1919
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
    def __init__(self, dataframe, transform=None, transform_dis=None):
        self.dataframe = dataframe
        self.transform = transform
        self.transform_dis = transform_dis
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['img_path']
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            ref = self.transform(img)
            dis = self.transform_dis(img)
        
        # mos column 존재 여부에 따라 값을 설정
        mos = float(self.dataframe.iloc[idx]['mos']) if 'mos' in self.dataframe.columns else 0.0
        
        return ref, dis, mos
    
## Define Model

## deformarable fusion block
class deform_fusion(nn.Module):
    def __init__(self, patch_size=8, in_channels=768*5, cnn_channels=256*3, out_channels=256*3):
        super().__init__()
        #in_channels, out_channels, kernel_size, stride, padding
        self.d_hidn = 512
        if patch_size == 8:
            stride = 1
        else:
            stride = 2
        self.conv_offset = nn.Conv2d(in_channels, 2*3*3, 3, 1, 1)
        self.deform = DeformConv2d(cnn_channels, out_channels, 3, 1, 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=self.d_hidn, kernel_size=3,padding=1,stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.d_hidn, out_channels=out_channels, kernel_size=3, padding=1,stride=stride)
        )

    def forward(self, cnn_feat, vit_feat):
        vit_feat = F.interpolate(vit_feat, size=cnn_feat.shape[-2:], mode="nearest")
        offset = self.conv_offset(vit_feat)
        deform_feat = self.deform(cnn_feat, offset)
        deform_feat = self.conv1(deform_feat)
        
        return deform_feat

# Pixel Prediction
class Pixel_Prediction(nn.Module):
    def __init__(self, inchannels=768*5+256*3, outchannels=256, d_hidn=1024):
        super().__init__()
        self.d_hidn = d_hidn
        self.down_channel = nn.Conv2d(inchannels, outchannels, kernel_size=1)
        self.feat_smoothing = nn.Sequential(
            nn.Conv2d(in_channels=256*3, out_channels=self.d_hidn, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.d_hidn, out_channels=512, kernel_size=3, padding=1)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3,padding=1), 
            nn.ReLU()
        )
        self.conv_attent =  nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
        )
    
    def forward(self,f_dis, f_ref, cnn_dis, cnn_ref):
        f_dis = torch.cat((f_dis,cnn_dis),1)
        f_ref = torch.cat((f_ref,cnn_ref),1)
        f_dis = self.down_channel(f_dis)
        f_ref = self.down_channel(f_ref)

        f_cat = torch.cat((f_dis - f_ref, f_dis, f_ref), 1)

        feat_fused = self.feat_smoothing(f_cat)
        feat = self.conv1(feat_fused)
        f = self.conv(feat)
        w = self.conv_attent(feat)
        pred = (f*w).sum(dim=2).sum(dim=2)/w.sum(dim=2).sum(dim=2)

        return pred
    
class SaveOutput:
    def __init__(self):
        self.outputs = []
    
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
    
    def clear(self):
        self.outputs = []

# define module
class CNN_ViT(nn.Module):
    def __init__(self):
        super(CNN_ViT, self).__init__()
        
        self.patch_size = 8
        self.batch_size = CFG['BATCH_SIZE']
        self.saveoutput = SaveOutput()
        
        self.resnet50 = timm.create_model("resnet50", pretrained=True).eval()
        
        self.vit = timm.create_model("vit_base_patch8_224", pretrained=True).eval()
        
        self.deform_net = deform_fusion(patch_size = self.patch_size)
        self.regressor = Pixel_Prediction()
        

        hook_handles = []
        for layer in self.resnet50.modules():
            if isinstance(layer, Bottleneck):
                handle = layer.register_forward_hook(self.saveoutput)
                hook_handles.append(handle)
                    
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.saveoutput)
                hook_handles.append(handle)
        

        
    def forward(self, ref, dis):
        vit_ref = self.vit.blocks(self.vit.patch_embed(ref))
        vit_ref = torch.cat(
            (
                self.saveoutput.outputs[0][:,:,:],
                self.saveoutput.outputs[1][:,:,:],
                self.saveoutput.outputs[2][:,:,:],
                self.saveoutput.outputs[3][:,:,:],
                self.saveoutput.outputs[4][:,:,:],
            ),
            dim=2
        )
        self.saveoutput.outputs.clear()
        
        vit_dis = self.vit.blocks(self.vit.patch_embed(dis))
        vit_dis = torch.cat(
            (
                self.saveoutput.outputs[0][:,:,:],
                self.saveoutput.outputs[1][:,:,:],
                self.saveoutput.outputs[2][:,:,:],
                self.saveoutput.outputs[3][:,:,:],
                self.saveoutput.outputs[4][:,:,:],
            ),
            dim=2
        )
        self.saveoutput.outputs.clear()
        
        if self.patch_size == 8:
            vit_ref = vit_ref.transpose(1, 2).view(self.batch_size, 768*5, 28, 28)
            vit_dis = vit_dis.transpose(1, 2).view(self.batch_size, 768*5, 28, 28)
        else:
            vit_ref = vit_ref.transpose(1, 2).view(self.batch_size, 768*5, 14, 14)
            vit_dis = vit_dis.transpose(1, 2).view(self.batch_size, 768*5, 14, 14)


        cnn_dis = self.resnet50(dis)
        cnn_dis = torch.cat(   
            (
                self.saveoutput.outputs[0],
                self.saveoutput.outputs[1],
                self.saveoutput.outputs[2]
            ),
            dim=1
        )
        self.saveoutput.outputs.clear()

        cnn_dis = self.deform_net(cnn_dis, vit_ref)
        
        cnn_ref = self.resnet50(ref)
        cnn_ref = torch.cat(   
            (
                self.saveoutput.outputs[0],
                self.saveoutput.outputs[1],
                self.saveoutput.outputs[2]
            ),
            dim=1
        )
        self.saveoutput.outputs.clear()
        cnn_ref = self.deform_net(cnn_ref,vit_ref)
        pred = self.regressor(vit_dis, vit_ref, cnn_dis, cnn_ref)
        
        return pred
    
path = '/home/6210seok/data/Samsung-Image-Quality/'

# 데이터 로드
train_data = pd.read_csv(path + 'train.csv')
test_data = pd.read_csv(path + 'test.csv')

def rewrite(df):
    df['img_path'] = path + df['img_path']
    return df

train_data = rewrite(train_data).drop_duplicates(['img_name']).reset_index(drop=True)
test_data = rewrite(test_data)
val_data = train_data[50000:].reset_index(drop=True)
train_data = train_data[:50000]
## Train
def val(model, criterion1, val_loader, epoch):
    
    model.eval()
    predicted_mos_list = []
    val_loss = 0
    
    val_loop = tqdm(val_loader, leave=True)
    with torch.no_grad():
        for imgs, dis, mos in val_loop:
            imgs, dis, mos = imgs.float().cuda(), dis.float().cuda(), mos.float().cuda()
            
            # Forward & Loss
            predicted_mos = model(imgs, dis)
            loss = criterion1(predicted_mos.squeeze(1), mos)
            
            val_loss += loss.item()
            val_loop.set_description(f"Validation")
            val_loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1} val average loss: {val_loss / len(val_loader):.4f}")
def train(model, criterion1, optimizer, train_loader, val_loader=None):
    # 학습
    for epoch in range(2):
        model.deform_net.train()
        model.regressor.train()
        total_loss = 0
        loop = tqdm(train_loader, leave=True)
        for imgs, dis, mos in loop:
            imgs, dis, mos = imgs.float().cuda(), dis.float().cuda(), mos.float().cuda()
            
            # Forward & Loss
            predicted_mos = model(imgs, dis)
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
        if val_loader != None:
            val(model, criterion1, val_loader, epoch)
            
# 데이터셋 및 DataLoader 생성
def gauss_noise_tensor(img):
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)
    
    sigma = 25.0
    
    out = img + sigma * torch.randn_like(img)
    
    if out.dtype != dtype:
        out = out.to(dtype)
        
    return out

transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

transform_dis = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])), 
    transforms.ToTensor(),
    gauss_noise_tensor,
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# 모델, 손실함수, 옵티마이저
model = CNN_ViT()
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)

for name, param in model.named_parameters():
  if name.startswith("resnet50"):
    param.requires_grad = False

for name, param in model.named_parameters():
  if name.startswith("vit"):
    param.requires_grad = False

model.to(device)

criterion1 = nn.MSELoss()
# optimizer = torch.optim.Adam([
#         {'params': model.regressor.parameters(), 'lr': CFG['LR'],'weight_decay':1e-5}, 
#         {'params': model.deform_net.parameters(),'lr': CFG['LR'],'weight_decay':1e-5}
#         ])
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)

train_dataset = CustomDataset(train_data, transform, transform_dis)
train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=2, drop_last=True)

val_dataset = CustomDataset(val_data, transform, transform_dis)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=2, drop_last=True)

train(model, criterion1, optimizer, train_loader, val_loader)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-5)
train(model, criterion1, optimizer, train_loader, val_loader)

# torch.save(model.state_dict(), 'Fusion-Module.pt')
model.eval()
predicted_mos_list = []

val_loop = tqdm(val_loader, leave=True)
with torch.no_grad():
    for imgs, dis, mos in val_loop:
        imgs, dis, mos = imgs.float().cuda(), dis.float().cuda(), mos.float().cuda()
            
        # Forward & Loss
        predicted_mos_list += model(imgs, dis).reshape(-1).tolist()
        
# 결과 저장
result_df = pd.DataFrame({
    'img_name': val_data['img_name'][:-6],
    'mos': predicted_mos_list
})
result_df.to_csv('val-fusion.csv', index=False)

import matplotlib.pyplot as plt

x = np.linspace(2.5,8)
plt.plot(result_df['mos'], val_data['mos'][:-6], 'g*')
plt.plot(x,x)
plt.show()
x_simple = result_df['mos']
y_simple = val_data['mos'][:-6]
my_rho = np.corrcoef(x_simple, y_simple)[0][1]

from scipy import stats
res = stats.spearmanr(x_simple, y_simple).statistic

print(f"PLCC: {my_rho:.5f}, SRCCL: {res:.5f}\nScore: {(my_rho + res) / 2}")

## Inference & Submit
test_dataset = CustomDataset(test_data, transform)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=2)

model.eval()
predicted_mos_list = []

test_loop = tqdm(test_loader, leave=True)
with torch.no_grad():
    for imgs, dis, mos in test_loop:
        imgs, dis, mos = imgs.float().cuda(), dis.float().cuda(), mos.float().cuda()
            
        # Forward & Loss
        predicted_mos_list += model(imgs, dis).reshape(-1).tolist()

# 결과 저장
result_df = pd.DataFrame({
    'img_name': test_data['img_name'],
    'mos': predicted_mos_list
})

result_df['comments'] = 'Nice Image.'

result_df.to_csv('Ref-Dis-fusion.csv', index=False)

print("Inference completed and results saved to submit.csv.")