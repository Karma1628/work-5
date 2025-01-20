import os
import copy
import glob
import random
import logging
import numpy as np
from rich import print

def update_memory_vanilla(args, task_id, memory):
    
    current_task = args.task[task_id]
    previous_task = list(memory.keys())
    all_task = current_task + previous_task

    memory_per_class = args.memory_size // len(all_task)
    
    memory_updated = copy.deepcopy(memory)
        
    for class_name in previous_task:
        memory_updated[class_name] = memory_updated[class_name][:memory_per_class]
        
    memory_per_current_class = (args.memory_size - memory_per_class * len(previous_task)) // (len(current_task))
        
    for class_name in current_task:
        memory_updated[class_name] = []
        class_path = os.path.join(args.data_dir, class_name, 'train', 'good', '*.png')
        images = glob.glob(class_path)
        random.shuffle(images)
        for idx in range(memory_per_current_class):
            memory_updated[class_name].append(images[idx])
        
    return memory_updated


def update_memory_ours(args, task_id, memory, model, train_loader):
    
    logger = logging.getLogger("global_logger")
    logger.info("Constructing the memory ...")
    
    current_task = args.task[task_id]
    previous_task = list(memory.keys())
    all_task = current_task + previous_task
    memory_per_class = args.memory_size // len(all_task)
    
    memory_updated = copy.deepcopy(memory)
    
    for class_name in previous_task:
        memory_updated[class_name] = memory_updated[class_name][:memory_per_class]
        
    memory_per_current_class = (args.memory_size - memory_per_class * len(previous_task)) // (len(current_task)) 
    
    model.eval()
    
    current_latent_set = {}
    for class_name in current_task:
        memory_updated[class_name] = []
        current_latent_set[class_name] = {
            'latent': [],
            'image_path': [],
        }
    
    for _, batch in enumerate(train_loader):

        clsname = batch['clsname']
        image_path = batch['image_path']
        image = batch["image"].cuda()
        prompt = batch["prompt"].cuda()
        
        outputs = model(image, prompt, None)
        
        b = image.shape[0]
        
        latents = outputs['latent']

        latents = latents.reshape(b, -1).detach().cpu().numpy() # (8, -1)
        
        for i in range(b):
            clsname_single = clsname[i]
            if clsname_single in current_task:
                current_latent_set[clsname_single]['latent'].append(latents[i]) # (256*256,)
                current_latent_set[clsname_single]['image_path'].append(image_path[i]) # 
                
    for cls in current_task:
        cls_latents = np.array(current_latent_set[cls]['latent']) # (n, 256*256)
        mean_cls_latents = np.mean(cls_latents, axis=0) # (256*256,)
        distances = np.linalg.norm(cls_latents - mean_cls_latents, axis=1)  # (n,)
        sorted_indices = np.argsort(distances)[::-1]  # 降序排序
        sorted_indices_choose = np.array(sorted_indices[:memory_per_current_class])
        for idx in sorted_indices_choose:
            memory_updated[cls].append(current_latent_set[cls]['image_path'][idx])
        
    logger.info("The memory is updated !")
    
    del current_latent_set
            
    return memory_updated
        
    
    
    
    
    
    
    
    