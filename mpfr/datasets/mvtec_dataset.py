import os
import cv2
import glob
import random
import numpy as np
import imgaug.augmenters as iaa

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from datasets.perlin import create_perlin_mask

def read_rgb_image(path, size):
    image_bgr = cv2.imread(path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, size)
    return image_rgb

class MVTecTrain(Dataset):

    def __init__(self, args, task_id, memory):
        
        self.current_image_paths = []
        for class_name in args.task[task_id]:
            class_path = os.path.join(args.data_dir, class_name, 'train', 'good', '*.png')
            images = glob.glob(class_path)
            self.current_image_paths.extend(images)
            
        self.memory = memory
        self.memory_class_names = list(memory.keys())
        self.memory_image_paths = []
        
        for class_name in memory.keys():
            self.memory_image_paths.extend(memory[class_name])
            
        self.image_paths = self.current_image_paths + self.memory_image_paths
            
        self.resource_paths = glob.glob(args.dtd_dir + "/*/*.jpg")
            
        self.resize_shape = (args.image_size, args.image_size)
        
        self.augmentations = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                              iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                              iaa.pillike.EnhanceSharpness(),
                              iaa.Solarize(0.5, threshold=(32,128)),
                              iaa.Posterize(),
                              iaa.Invert(),
                              iaa.pillike.Autocontrast(),
                              iaa.pillike.Equalize(),
                              iaa.Affine(rotate=(-45, 45))]
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        
        self.memory_augmentations = iaa.OneOf([
            iaa.Affine(rotate=(-90, 90)),
            iaa.Fliplr(),
            iaa.Flipud(),
        ])
        
        self.transform = transforms.Compose([transforms.ToPILImage(), 
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std =[0.229, 0.224, 0.225])])


    def __len__(self):
        return len(self.image_paths)
    
    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmentations)), 3, replace=False)
        aug = iaa.Sequential([self.augmentations[aug_ind[0]],
                              self.augmentations[aug_ind[1]],
                              self.augmentations[aug_ind[2]]])
        return aug
    
    def __getitem__(self, idx):
        
        image_path = self.image_paths[idx]
        image_class_name = image_path.split('/')[-4]
        
        image = read_rgb_image(image_path, self.resize_shape) # np.uint8 (224, 224, 3)
        normal_mask = torch.zeros(self.resize_shape, dtype=torch.long)
        
        if image_class_name in self.memory_class_names:
            if image_class_name not in ['toothbrush', 'transistor']: 
                if torch.rand(1).item() < 0.5:
                    image = self.memory_augmentations(image=image)
        
        k = random.randint(0, len(self.resource_paths)-1) # randomly choose the resource image
        aug = self.randAugmenter()
        resource_image = read_rgb_image(self.resource_paths[k], self.resize_shape) # np.uint8 (256, 256, 3)
        resource_image = aug(image=resource_image)
        pseudo_mask = create_perlin_mask(self.resize_shape) # binary np.float32 [256, 256, 1]
        pseudo_mask = self.rot(image=pseudo_mask) # rotate the mask
        beta = np.random.rand() * 0.8
        pseudo_image = image * (1 - pseudo_mask) + (1 - beta) * resource_image * pseudo_mask + beta * image * pseudo_mask # np.float64 [256, 256, 3]
        pseudo_image = np.uint8(pseudo_image)

        if image_class_name in self.memory_class_names:
            prompt_pool = self.memory[image_class_name]
        else:
            prompt_pool = self.current_image_paths
            
        k = random.randint(0, len(prompt_pool) - 1)
        prompt_path = prompt_pool[k]
        prompt = read_rgb_image(prompt_path, self.resize_shape)
        
        # if image_class_name in self.memory_class_names:
        #     path = '/mnt/SSD8T/home/cyc/workbench/paper5_memory_onenip/misc'
        #     cv2.imwrite(os.path.join(path, 'image.png'), image)
        #     cv2.imwrite(os.path.join(path, 'prompt.png'), prompt)
        #     cv2.imwrite(os.path.join(path, 'pseudo.png'), pseudo_image)
        #     exit(0)
            
        image = self.transform(image)
        prompt = self.transform(prompt)
        
        if torch.rand(1).item() < 0.5:
            pseudo_image = self.transform(pseudo_image)
            pseudo_mask = torch.from_numpy(pseudo_mask).squeeze().long()
        else:
            pseudo_image = image
            pseudo_mask = normal_mask
    
        return {
            'image': image,
            'prompt': prompt,
            'pseudo_image': pseudo_image,
            'pseudo_mask': pseudo_mask,
            'clsname': image_class_name,
            'image_path': image_path,
        }


class MVTecTest(Dataset):

    def __init__(self, args, current_task, previous_task, memory):
        
        self.current_task = current_task
        self.previous_task = previous_task
        self.memory = memory
        
        all_task = current_task + previous_task

        self.test_image_paths = []
        for class_name in all_task:
            class_path = os.path.join(args.data_dir, class_name, 'test', '*', '*.png')
            images = glob.glob(class_path)
            self.test_image_paths.extend(images)
        
        self.current_prompt_paths = []
        for class_name in current_task:
            class_path = os.path.join(args.data_dir, class_name, 'train', 'good', '*.png')
            images = glob.glob(class_path)
            self.current_prompt_paths.extend(images)
            
        self.resize_shape = (args.image_size, args.image_size)
        self.transform = transforms.Compose([transforms.ToPILImage(), 
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        
    def __len__(self):
        return len(self.test_image_paths)
    
    def __getitem__(self, idx):
        image_path = self.test_image_paths[idx]
        image = read_rgb_image(image_path, self.resize_shape)
        image = self.transform(image)
        
        clsname = image_path.split('/')[-4]
        anoname = image_path.split('/')[-2] # thread or good
        if anoname == 'good':
            mask = np.zeros((1, self.resize_shape[0], self.resize_shape[1]), dtype=np.float32)
            ano_label = 0
        else:
            mask_path = image_path.replace('/test/', '/ground_truth/').replace('.png', '_mask.png')
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.resize_shape).astype(np.float32) / 255.0
            mask = mask[None, ...]
            ano_label = 1
            
        if clsname in self.current_task:
            prompt_pool = self.current_prompt_paths
        else:
            prompt_pool = self.memory[clsname]
            
        k = random.randint(0, len(prompt_pool) - 1)
        prompt_path = prompt_pool[k]
        prompt = read_rgb_image(prompt_path, self.resize_shape)
        prompt = self.transform(prompt)

        return {
            'image': image,
            'prompt': prompt,
            'image_path': image_path, 
            'clsname': clsname,
            'mask': mask,
            'label': ano_label
        }