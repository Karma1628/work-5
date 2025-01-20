import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

def calculate_metrics(results):
    
    masks = np.array(results['mask'], dtype=np.int32)
    score_maps = np.array(results['pixel_pred'], dtype=np.float32)
    
    image_score = np.array(results['image_pred'], dtype=np.float32)
    image_label = np.array(results['label'], dtype=np.int32)

    image_auc = roc_auc_score(image_label, image_score)

    masks_flat = masks.ravel()
    score_maps_flat = score_maps.ravel()
    pixel_auc = roc_auc_score(masks_flat, score_maps_flat)
    pixel_ap = average_precision_score(masks_flat, score_maps_flat)
    precision, recall, _ = precision_recall_curve(masks_flat, score_maps_flat)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AP = {pixel_ap:.2f}', linewidth=2, color='blue')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves for Different APs', fontsize=14)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('/home/gysj_cyc/workbench/paper5_mem_distill_onenip_cat/misc/pr_curve.png')

    return {'image_auc': image_auc,'pixel_auc': pixel_auc,'pixel_ap': pixel_ap}