import logging
logger = logging.getLogger("global")

from torch.utils.data import DataLoader

from datasets.mvtec_dataset import MVTecTrain, MVTecTest

def build_dataloader(args, task_id, memory):
    train_dataset = MVTecTrain(args, task_id, memory)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True
    )
    
    current_task = args.task[task_id]
    previous_task = list(memory.keys())
    # all_task = current_task + previous_task
            
    test_dataset = MVTecTest(args, current_task, previous_task, memory)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        # drop_last=True
    )
    
    return train_loader, test_loader