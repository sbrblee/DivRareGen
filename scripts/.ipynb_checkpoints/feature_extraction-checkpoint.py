import argparse
import numpy as np
import random
from functools import partial
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def VGG16_feature(dataset, layer, layer_num, batch_size, device, binarized):
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    _, height, width = dataset[0].shape
    if height != 224 or width != 224:
        resize = partial(F.interpolate, size=(224, 224))
    else:
        def resize(x): return x

    print('loading vgg16...', end='', flush=True)
    vgg16 = models.vgg16(pretrained=True).eval().to(device)
    print('done')
    
    modules = []
    if layer == "features":
        modules.append(vgg16.features[:layer_num+1])
    elif layer == "classifier":
        modules.append(vgg16.features)
        modules.append(vgg16.classifier[:layer_num+1])
    
    features = []
    total_iteration = len(dataset)//batch_size
    for i, batch in enumerate(tqdm(data_loader)):
        if i % 100 == 0:
            print(f'iteration #: {i}/{total_iteration}')
        batch = resize(torch.Tensor(batch)).to(device)
        
        for module in modules:
            batch = module(batch)
            batch = batch.view(batch.shape[0], -1)
        
        if binarized == 0:  
            features.append(batch.cpu().data.numpy())
        else:
            features.append(((batch>0) + 0).cpu().data.numpy())
    return np.concatenate(features, axis=0)

def VGG19_feature(dataset, layer, layer_num, batch_size, device, binarized):
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    _, height, width = dataset[0].shape
    if height != 224 or width != 224:
        resize = partial(F.interpolate, size=(224, 224))
    else:
        def resize(x): return x

    print('loading vgg19...', end='', flush=True)
    vgg19 = models.vgg19(pretrained=True).eval().to(device)
    print('done')
    
    modules = []
    if layer == "features":
        modules.append(vgg19.features[:layer_num+1])
    elif layer == "classifier":
        modules.append(vgg19.features)
        modules.append(vgg19.classifier[:layer_num+1])
    
    features = []
    total_iteration = len(dataset)//batch_size
    for i, batch in enumerate(tqdm(data_loader)):
        if i % 100 == 0:
            print(f'iteration #: {i}/{total_iteration}')
        batch = resize(torch.Tensor(batch)).to(device)
        
        for module in modules:
            batch = module(batch)
            batch = batch.view(batch.shape[0], -1)
        
        if binarized == 0:  
            features.append(batch.cpu().data.numpy())
        else:
            features.append(((batch>0) + 0).cpu().data.numpy())
    return np.concatenate(features, axis=0)

class ImageFolder(torch.utils.data.Dataset): 
    def __init__(self, root, transform=None):
        self.root = root
        self.T = transform
        self.fnames = sorted(list(os.listdir(root)))
    def __len__(self):
        return len(self.fnames)
    def __getitem__(self, idx): 
        img = Image.open(self.root+self.fnames[idx])
        if self.T is not None:
            img = self.T(img)
        return img

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--model", type=str, default="vgg16", help="select the feature extractor or pre-trained classifier"
    )
    parser.add_argument(
        "--layer", type=str, default="classifier", help="select the layer to define features"
    )
    parser.add_argument(
        "--layer_num", type=int, default=4, help="select the layer number to define features"
    )
    parser.add_argument(
        "--img_size", type=int, default=1024, help="image size"
    )
    parser.add_argument(
        "--batch_size", type=int, default=200, help="batch size"
    )
    parser.add_argument(
        "--model_path", type=str, default="", help="the path of the feature extractor if torch pretrained version model is not used"
    )
    parser.add_argument(
        "--data_path", type=str, default="",help="the path of data to feature be extracted"
    )
    parser.add_argument(
        "--save_path", type=str, default="./", help="the save path for features"
    )
    parser.add_argument(
        "--binarized", type=int, default=0, help="0 if float feature value is saved, 1 if binarized feature value is saved"
    )
    parser.add_argument(
        "--data_type", type=str, default="real", help="data type: real or fake"
    )
    
    args = parser.parse_args()
    
    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])  
    
    dataset = ImageFolder(args.data_path, transform=transform)
    
    if args.model == "vgg16":
        features = VGG16_feature(dataset, args.layer, args.layer_num, args.batch_size, device, args.binarized)
    elif args.model == "vgg19":
        features = VGG19_feature(dataset, args.layer, args.layer_num, args.batch_size, device, args.binarized)

    if args.binarized == 1:
        np.savez_compressed(args.save_path + f"{args.data_type}_{args.model}_{args.layer}_{str(args.layer_num)}_feature_binarized.npz", feature=features)
    else:
        np.savez_compressed(args.save_path + f"{args.data_type}_{args.model}_{args.layer}_{str(args.layer_num)}_feature.npz", feature=features)