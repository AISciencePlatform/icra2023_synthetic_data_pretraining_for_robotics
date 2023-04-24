"""
This code is based on Pytorch-UNet (https://github.com/milesial/Pytorch-UNet)
"""
import sys
sys.path.insert(1, '../Pytorch-UNet/')
import logging
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from utils.data_loading import BasicDataset
from unet import UNet

def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()



def get_predictions(input_path, output_path,model_path):
    scale = 1
    mask_threshold = 0.5
    net = UNet(n_channels=1, n_classes=2, bilinear=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {model_path}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(model_path, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    os.chdir(input_path)
    for filename in glob.glob("*.jpg"):
        #logging.info(f'Predicting image {filename} ...')
        #img = Image.open(filename)
        print('input: ' + filename)
        out_filename = output_path+filename[:len(filename)-3]+"png"
        print('output: '+ out_filename)

        img = Image.open(filename)
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=scale,
                           out_threshold=mask_threshold,
                           device=device)

        result = mask_to_image(mask, mask_values)
        result.save(out_filename)
        logging.info(f'Mask saved to {out_filename}')