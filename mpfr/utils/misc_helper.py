import os
import cv2
import logging
import numpy as np
from datetime import datetime
from PIL import Image

import torch
import torchvision.transforms as transforms

import torch.nn.functional as F


def get_current_time():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return current_time

def basicConfig(*args, **kwargs):
    return

logging.basicConfig = basicConfig
def create_logger(name, log_file, level=logging.INFO):
    log = logging.getLogger(name)
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)20s][line:%(lineno)4d][%(levelname)4s] %(message)s"
    )
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    log.setLevel(level)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def save_checkpoint(model, save_dir, task_id):
    logger = logging.getLogger("global_logger")
    checkpoint_path = os.path.join(save_dir, f"task{task_id}.pth")
    checkpoint = {
        "reconstructive_network": model.reconstructive_network.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(model, checkpoint_path):
    logger = logging.getLogger("global_logger")
    """加载 reconstructive_network 的权重以及训练状态"""
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.reconstructive_network.load_state_dict(checkpoint["reconstructive_network"])
    logger.info(f"Checkpoint loaded: {checkpoint_path}")
    

def convert2heatmap(x):
    """
        x: [h, w, 3] np.uin8
    """
    x = cv2.applyColorMap(x, cv2.COLORMAP_JET)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    return x

def float2uint8(x):
    """
        x: [h, w], (0, 1) np.float
    """
    x = np.repeat(x[..., np.newaxis], 3, 2)
    x = np.uint8(x * 255)
    return x 

def unnormalize(img):
    reverse_transform = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        transforms.ToPILImage()])
    return reverse_transform(img)

def read_rgb_image(path, size):
    image_bgr = cv2.imread(path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (size[0], size[1]))
    return image_rgb
    
    
def visualize(save_dir, image_path, mask, pred, size):
    # ./result /home/c1c/workbench/data/Public/mvtec/bottle/test/contamination/001.png 
    # torch.Size([1, 3, 224, 224]) 
    # (224, 224) 
    # (224, 224)
    size = [size, size]
    clsname = image_path.split('/')[-4]
    save_dir = os.path.join(save_dir, 'images', clsname)
    os.makedirs(save_dir, exist_ok=True)
    
    display_pil_images = []
    
    image = read_rgb_image(image_path, size)
    display_pil_images.append(Image.fromarray(image))
    
    mask_np = float2uint8(mask)
    mask_pil = Image.fromarray(mask_np)
    display_pil_images.append(mask_pil)
    
    pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
    pred = float2uint8(pred)
    pred = convert2heatmap(pred)
    display_pil_images.append(Image.fromarray(pred))
    
    overlay =  cv2.addWeighted(image, 0.7, pred, 0.3, 0)
    display_pil_images.append(Image.fromarray(overlay))
    
    num_images = len(display_pil_images)
    combination = Image.new('RGB', (size[0]*num_images+5*num_images, size[0]), (255, 255, 255))
    for idx, img in enumerate(display_pil_images):
        img = img.resize(size)
        combination.paste(img, (idx*(size[0] + 5), 0))
    save_name = '_'.join(image_path.split('/')[-2:])
    combination.save(os.path.join(save_dir, save_name))

def normalize(tensor):
    return F.normalize(tensor, p=2, dim=-1)