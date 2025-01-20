import os
import copy
import argparse
import pandas as pd
from rich import print
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metric_helper import *
from utils.misc_helper import *
from utils.memory_helper import *
from datasets.data_builder import build_dataloader
from models.my_model import FeatureReconstruction
from utils.loss_helper import DiceBCELoss

parser = argparse.ArgumentParser(description="Class-Incremental Anomaly Detection.")
parser.add_argument('--data-dir', type=str, default='/home/gysj_cyc/workbench/data/mvtec', help='Path to the root directory of the mvtec dataset.')
parser.add_argument('--dtd-dir', type=str, default='/home/gysj_cyc/workbench/data/dtd/images', help='Path to the root directory of the dtd dataset.')
parser.add_argument('--result-dir', type=str, default='./result', help='Directory to save training results.')
parser.add_argument('--model-name', type=str, default='vit_small_patch14_dinov2', help='Name of the pretrained model.')
parser.add_argument('--weight-path', type=str, default='/home/gysj_cyc/workbench/ckpts/dinov2_vits14_pretrain.pth', help='Path to the pretrained model weights.')
parser.add_argument('--memory-size', type=int, default=30, help='Size of the memory buffer.')
parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer.')
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs for training.')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay (L2 regularization) for the optimizer.')
parser.add_argument('--val_freq_epoch', type=int, default=100, help='How often to val in training progress.')
parser.add_argument('--print_freq_step', type=int, default=200, help='How often to log in training progress.')
parser.add_argument('--image-size', type=int, default=224, help='Image size for input images.')
parser.add_argument('--feature-size', type=int, default=16, help='Size of the feature vector.')
parser.add_argument('--stages', type=int, nargs='+', default=[11], help='Stages to be used in the model.')
parser.add_argument('--num-workers', type=int, default=8, help='Number of workers for data loading.')
args = parser.parse_args()

checkpoint_dir = os.path.join(args.result_dir, 'checkpoints')
log_dir = os.path.join(args.result_dir, 'log')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# 起始任务类数，后续每个任务类数，后续任务数
task_dict = {
    'test': [['zipper']],
    '15-0-0': [['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']],
    '1-1-14': [['bottle'], ['cable'], ['capsule'], ['carpet'], ['grid'], ['hazelnut'], ['leather'], ['metal_nut'], ['pill'], ['screw'], ['tile'], ['toothbrush'], ['transistor'], ['wood'], ['zipper']],
    '3-3-4': [['bottle', 'cable', 'capsule'], ['carpet', 'grid', 'hazelnut'], ['leather', 'metal_nut', 'pill'], ['screw', 'tile', 'toothbrush'], ['transistor', 'wood', 'zipper']],
    '5-5-2': [['bottle', 'cable', 'capsule', 'carpet', 'grid'], ['hazelnut', 'leather', 'metal_nut', 'pill', 'screw'], ['tile', 'toothbrush', 'transistor', 'wood', 'zipper']]
}
args.task = task_dict['1-1-14']
# args.task = task_dict['15-0-0']
# args.task = task_dict['test']

current_time = get_current_time()
logger = create_logger("global_logger", log_dir + "/dec_{}.log".format(current_time))

for key, value in vars(args).items():
    print(f"{key}: {value}")

memory = {} # each class 2 samples
for task_id in range(len(args.task)):
    
    logger.info(f"Task {task_id} is begining ...")
    
    train_loader, test_loader = build_dataloader(args, task_id, memory)
    
    logger.info(f'Train loader: {len(train_loader)}, Val loader: {len(test_loader)}')
    result = 'Memory: '
    if task_id != 0:
        for key in memory.keys():
            result += f"{key}({len(memory[key])}) "
    else:
        result += 'Empty'
    logger.info(result)
    
    current_clsnames = args.task[task_id]
    previous_clsnames = list(memory.keys())
    
    model = FeatureReconstruction(
        model_name=args.model_name,
        weight_path=args.weight_path,
        image_size=args.image_size,
        feature_size=args.feature_size,
        stages=args.stages,
        num_encoder=4,
        num_decoder=4
    )
    
    if task_id > 0:
        checkpoint_path = os.path.join(checkpoint_dir, f'task{task_id-1}.pth')
        load_checkpoint(model, checkpoint_path)
        model_old = copy.deepcopy(model)
        model_old.cuda()

        for param in model_old.parameters():
            param.requires_grad = False
    
    model.cuda()

    l2_loss = nn.MSELoss(reduction='mean')
    dicebce_loss = DiceBCELoss(bce_weight=0.5)
    
    # optimizer = torch.optim.Adam(model.reconstructive_network.parameters(), lr=args.lr)
    optimizer = torch.optim.AdamW([
        {'params': model.reconstructive_network.parameters(), 'lr': args.lr},
    ], lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [args.epochs * 0.8], gamma=0.1, last_epoch=-1)

    for epoch in range(0, args.epochs):
        
        model.train()
        
        if task_id > 0:
            model_old.eval()
        
        start_step = epoch * len(train_loader)
        for step, batch in enumerate(train_loader):
            
            # dict_keys(['image', 'image_path', 'clsname'])
            clsname = batch['clsname']
            image = batch["image"].cuda() # torch.Size([8, 3, 224, 224])
            prompt = batch["prompt"].cuda() # torch.Size([8, 3, 224, 224])
            pseudo_image = batch["pseudo_image"].cuda() # torch.Size([8, 3, 224, 224])
            pseudo_mask = batch['pseudo_mask'].cuda() # torch.Size([8, 224, 224])
            pseudo_mask = (pseudo_mask >= 0.5).long() # torch.Size([8, 224, 224])
            
            # torch.Size([8, 384, 16, 16])
            # torch.Size([8, 384, 16, 16])
            # torch.Size([256, 8, 256])
            # torch.Size([8, 1, 224, 224])
            outputs = model(image, prompt, pseudo_image)
            
            loss = 0.0
            rec_loss = l2_loss(outputs['reconstruction'], outputs['feature'].detach().data)   
            seg_loss = dicebce_loss(outputs['prediction'], pseudo_mask)
            loss += rec_loss + seg_loss
            
            if task_id > 0:
                outputs_old = model_old(image, prompt, pseudo_image) 
                match_all_loss = l2_loss(normalize(outputs['latent']), normalize(outputs_old['latent']).detach().data)
            else:
                match_all_loss = torch.tensor(0.0, requires_grad=False)
                
            loss += match_all_loss
       
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            current_step = start_step + step
            if (current_step + 1) % args.print_freq_step == 0:
                logger.info(
                    f"Task id: [{task_id+1}/{len(args.task)}]\t"
                    f"Epoch: [{epoch}/{args.epochs}]\t"
                    f"Step: [{current_step}/{len(train_loader) * args.epochs}]\t"
                    f"Loss: {loss.item():.5f}\t"
                    f"Rec Loss: {rec_loss.item():.5f}\t"  
                    f"Seg Loss: {seg_loss.item():.5f}\t"
                    f"Match All Loss: {match_all_loss.item():.5f}\t"
                    f"LR: {scheduler.get_last_lr()[0]: .5f}\t"
                )
                
        scheduler.step()
        
        if (epoch + 1) % args.val_freq_epoch == 0:
            
            logger.info("Start Validating the learned classes ...")
            val_class_names = current_clsnames + previous_clsnames
            
            results_dict = {}
            
            for class_name in val_class_names:
                results_dict[class_name] = {
                    'pixel_pred': [],
                    'image_pred': [],
                    'mask': [],
                    'label': []
                }
            
            model.eval()
            
            for step, batch in enumerate(test_loader):
                
                # dict_keys(['image', 'image_path', 'clsname', 'mask', 'label'])
                image_path = batch['image_path'][0]
                clsname = batch['clsname'][0]
                image = batch['image'].cuda() # torch.Size([1, 3, 224, 224])
                prompt = batch['image'].cuda() # # torch.Size([1, 3, 224, 224])
                mask = batch['mask'].squeeze().numpy() # torch.Size([1, 1, 224, 224]) -> (224, 224)
                label = batch['label'].numpy() # tensor([1]) -> (1,)
                
                outputs = model(image, prompt, None)
                
                pixel_pred = outputs['prediction'][: ,1: ,: ,:]
                # pixel_pred = torch.sum((feature - reconstruction) ** 2, dim=1, keepdim=True) # torch.Size([1, 1, 16, 16])
                # pixel_pred = F.interpolate(pixel_pred, size=args.image_size, mode='bilinear', align_corners=True) # [1, 1, 256, 256]
                pooled_pixel_pred = nn.functional.avg_pool2d(pixel_pred, 32, stride=1, padding=16).squeeze(0) # pool_size 32
                # pooled_pixel_pred = nn.functional.avg_pool2d(pixel_pred, 16, stride=1).squeeze(0) # uniad
                image_pred = pooled_pixel_pred.amax(dim=(1, 2)).detach().cpu().numpy() # (1,)
                pixel_pred = pixel_pred.squeeze().detach().cpu().numpy() # (224, 224)
                
                results_dict[clsname]['pixel_pred'].append(pixel_pred)
                results_dict[clsname]['image_pred'].append(image_pred)
                results_dict[clsname]['mask'].append(mask)
                results_dict[clsname]['label'].append(label)
                
                visualize(args.result_dir, image_path, mask, pixel_pred, args.image_size)
                
            logger.info('Start calculating the AUROC and AP metrics for learned classes ...')
            
            results_print = []
            for class_name in results_dict:
                class_metrics = calculate_metrics(results_dict[class_name])
                results_print.append([class_name, 
                                      round(class_metrics['image_auc'], 4), 
                                      round(class_metrics['pixel_auc'], 4), 
                                      round(class_metrics['pixel_ap'], 4)])
                
            df = pd.DataFrame(results_print, columns=["Category", "image_auc", "pixel_auc", "pixel_ap"])
            mean_values = df[["image_auc", "pixel_auc", "pixel_ap"]].mean()
            mean_row = ["Mean"] + mean_values.round(4).tolist()
            # mean_row_df = pd.DataFrame([mean_row], columns=df.columns) 
            # df_with_mean = pd.concat([df, mean_row_df], ignore_index=True)
            results_print += [mean_row]
            headers = ["Category", "image_auc", "pixel_auc", "pixel_ap"]
            logger.info("\n" + (tabulate(results_print, headers=headers, tablefmt="pretty")))
            
            del results_dict
            save_checkpoint(model, checkpoint_dir, task_id)
            
    # memory = update_memory_vanilla(args, task_id, memory)
    memory = update_memory_ours(args, task_id, memory, model, train_loader)

