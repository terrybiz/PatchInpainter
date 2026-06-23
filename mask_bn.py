import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from  torchvision import datasets, transforms

import nets.bagnet
import nets.resnet

import os 
import joblib
import argparse
from tqdm import tqdm
import numpy as np 
from scipy.special import softmax
from math import ceil
import PIL

parser = argparse.ArgumentParser()

parser.add_argument("--model_dir",default='checkpoints',type=str,help="path to checkpoints")
parser.add_argument('--data_dir', default='data', type=str,help="path to data")
parser.add_argument('--dataset', default='imagenette', choices=('imagenette','imagenet','cifar'),type=str,help="dataset")
parser.add_argument("--model",default='bagnet17',type=str,help="model name")
parser.add_argument("--clip",default=-1,type=int,help="clipping value; do clipping when this argument is set to positive")
parser.add_argument("--aggr",default='none',type=str,help="aggregation methods. set to none for local feature")
parser.add_argument("--skip",default=1,type=int,help="number of example to skip")
parser.add_argument("--thres",default=0.0,type=float,help="detection threshold for robust masking")
parser.add_argument("--patch_size",default=-1,type=int,help="size of the adversarial patch")
args = parser.parse_args()

MODEL_DIR=os.path.join('.',args.model_dir)
DATA_DIR=os.path.join(args.data_dir,args.dataset)
DATASET = args.dataset
def get_dataset(ds,data_dir):
    if ds in ['imagenette','imagenet']:
        ds_dir=os.path.join(data_dir,'val')
        ds_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        dataset_ = datasets.ImageFolder(ds_dir,ds_transforms)
        class_names = dataset_.classes
    elif ds == 'cifar':
        ds_transforms = transforms.Compose([
            transforms.Resize(192, interpolation=PIL.Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_ = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=ds_transforms)
        class_names = dataset_.classes
    return dataset_,class_names

val_dataset_,class_names = get_dataset(DATASET,DATA_DIR)
skips = list(range(0, len(val_dataset_), args.skip))
val_dataset = torch.utils.data.Subset(val_dataset_, skips)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8,shuffle=False)

#build and initialize model
device = 'cuda' #if torch.cuda.is_available() else 'cpu'

if args.clip > 0:
    clip_range = [0,args.clip]
else:
    clip_range = None

if 'bagnet17' in args.model:
    model = nets.bagnet.bagnet17(pretrained=True,clip_range=clip_range,aggregation=args.aggr)
    rf_size=17
elif 'bagnet33' in args.model:
    model = nets.bagnet.bagnet33(pretrained=True,clip_range=clip_range,aggregation=args.aggr)
    rf_size=33
elif 'bagnet9' in args.model:
    model = nets.bagnet.bagnet9(pretrained=True,clip_range=clip_range,aggregation=args.aggr)
    rf_size=9


if DATASET == 'imagenette':
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(os.path.join(MODEL_DIR,args.model+'_nette.pth'))
    model.load_state_dict(checkpoint['model_state_dict']) 
    args.patch_size = args.patch_size if args.patch_size>0 else 32     
elif  DATASET == 'imagenet':
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(os.path.join(MODEL_DIR,args.model+'_net.pth'))
    model.load_state_dict(checkpoint['state_dict'])
    args.patch_size = args.patch_size if args.patch_size>0 else 32 
elif  DATASET == 'cifar':
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(os.path.join(MODEL_DIR,args.model+'_192_cifar.pth'))
    model.load_state_dict(checkpoint['net'])
    args.patch_size = args.patch_size if args.patch_size>0 else 30


def laplace_inpaint(feature, mask, max_iter=1500, tol=0.0025):
    inpainted = feature.copy()
    H, W, C = feature.shape
    
    valid_mask = ~mask
    if not np.any(mask) or not np.any(valid_mask):
        return feature

    for channel in range(C):
        channel_data = inpainted[:, :, channel]
        
        mean_val = np.mean(channel_data[valid_mask])
        channel_data[mask] = mean_val
        
        for i in range(max_iter):
            up = np.roll(channel_data, -1, axis=0)
            down = np.roll(channel_data, 1, axis=0)
            left = np.roll(channel_data, -1, axis=1)
            right = np.roll(channel_data, 1, axis=1)

            up[-1, :] = channel_data[-1, :]    # 上移後，最底部的像素應該等於原底部
            down[0, :] = channel_data[0, :]    # 下移後，最頂部的像素應該等於原頂部
            left[:, -1] = channel_data[:, -1]  # 左移後，最右側的像素應該等於原右側
            right[:, 0] = channel_data[:, 0]   # 右移後，最左側的像素應該等於原左側

            new_val = 0.25 * (up + down + left + right)

            diff = np.abs(new_val[mask] - channel_data[mask])
            max_diff = np.max(diff) if diff.size > 0 else 0
            
            channel_data[mask] = new_val[mask]
            
            if max_diff < tol:
                break
                
        inpainted[:, :, channel] = channel_data
        
    return inpainted


import itertools

def masking_defense(local_feature, clipping=-1, window_shape=[6,6]):
    H, W, num_cls = local_feature.shape
    win_h, win_w = window_shape
    
    processed_feature = np.clip(local_feature, 0, clipping) if clipping > 0 else np.clip(local_feature, 0, np.inf)

    def compute_state(mask_windows):
        mask = np.zeros((H, W), dtype=bool)
        for (r, c) in mask_windows:
            mask[r:r+win_h, c:c+win_w] = True
        masked_feat = np.where(mask[:, :, np.newaxis], 0, processed_feature)
        global_vec = softmax(np.sum(masked_feat, axis=(0, 1)))
        pred = np.argmax(global_vec)
        conf = global_vec[pred]
        return mask, pred, conf
        
    initial_windows = tuple()
    _, init_pred, _ = compute_state(initial_windows)
    
    active_windows = []
    fallback_mask = np.zeros((H, W), dtype=bool)
    
    num_win_x = H - win_h + 1 
    num_win_y = W - win_w + 1

    for r in range(num_win_x):
        for c in range(num_win_y):
            m, p, _ = compute_state([(r, c)])
            if p != init_pred:
                active_windows.append((r, c))
                fallback_mask |= m

    if not active_windows:
        return init_pred 
    else:
        best_mask = fallback_mask
        restored_feature = laplace_inpaint(processed_feature, best_mask)
        final_vec = softmax(np.sum(restored_feature, axis=(0, 1)))
        return np.argmax(final_vec)




def provable_masking(local_feature, label, clipping=-1, window_shape=[6,6]):
    H, W, num_cls = local_feature.shape
    win_h, win_w = window_shape
    processed_feature = np.clip(local_feature, 0, clipping) if clipping > 0 else np.clip(local_feature, 0, np.inf)

    def compute_state(mask_windows):
        mask = np.zeros((H, W), dtype=bool)
        for (r, c) in mask_windows:
            mask[r:r+win_h, c:c+win_w] = True
        masked_feat = np.where(mask[:, :, np.newaxis], 0, processed_feature)
        global_vec = softmax(np.sum(masked_feat, axis=(0, 1)))
        return mask, np.argmax(global_vec), global_vec[np.argmax(global_vec)]
        
    _, init_pred, _ = compute_state(tuple())
    
    active_windows = []
    fallback_mask = np.zeros((H, W), dtype=bool)
    
    num_win_x, num_win_y = H - win_h + 1, W - win_w + 1 
    for r in range(num_win_x):
        for c in range(num_win_y):
            m, p, _ = compute_state([(r, c)])
            if p != init_pred:
                active_windows.append((r, c))
                fallback_mask |= m

    if not active_windows:
        return 0 if init_pred != label else 2
    else:
        best_mask = fallback_mask

        tmp_mask = best_mask.copy()
        tmp_mask[1:, :] |= best_mask[:-1, :]
        tmp_mask[:-1, :] |= best_mask[1:, :]
        tmp_mask[:, 1:] |= best_mask[:, :-1]
        tmp_mask[:, :-1] |= best_mask[:, 1:]
        non_best_mask = tmp_mask & ~best_mask
        
        for c in range(num_cls):
            channel = processed_feature[:, :, c]
            valid_vals = channel[non_best_mask]
            if valid_vals.size == 0:
                continue
            fill_val = np.min(valid_vals) if c == label else np.max(valid_vals)
            channel[best_mask] = fill_val
            processed_feature[:, :, c] = channel

        final_vec = softmax(np.sum(processed_feature, axis=(0, 1)))
        return 1 if np.argmax(final_vec) != label else 2


rf_stride=8
window_size = ceil((args.patch_size + rf_size -1) / rf_stride)
print("window_size",window_size)

    
model = model.to(device)
model.eval()
cudnn.benchmark = True

accuracy_list=[]
result_list=[]
clean_corr=0

for data,labels in tqdm(val_loader):
    
    data=data.to(device)
    labels = labels.numpy()
    output_clean = model(data).detach().cpu().numpy()

    for i in range(len(labels)):
        local_feature = output_clean[i]
        result = provable_masking(local_feature, labels[i], window_shape=[window_size,window_size])
        result_list.append(result)
        clean_pred = masking_defense(local_feature, window_shape=[window_size,window_size])
        clean_corr += clean_pred == labels[i]
            
    acc_clean = np.sum(np.argmax(np.mean(output_clean,axis=(1,2)),axis=1) == labels)
    accuracy_list.append(acc_clean)


cases,cnt=np.unique(result_list,return_counts=True)
print("Provable robust accuracy:",cnt[-1]/len(result_list) if len(cnt)==3 else 0)
print("Clean accuracy with defense:",clean_corr/len(result_list))
print("Clean accuracy without defense:",np.sum(accuracy_list)/len(val_dataset))
print("------------------------------")
print("Provable analysis cases (0: incorrect prediction; 1: vulnerable; 2: provably robust):",cases)
print("Provable analysis breakdown",cnt/len(result_list))
