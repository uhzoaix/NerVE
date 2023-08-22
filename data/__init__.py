import os, sys
from torch.utils.data import DataLoader
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

import dataset

def get_dataset(dataset_name):
    if hasattr(dataset, dataset_name):
        return getattr(dataset, dataset_name)

    return None

def get_dataloader(opt, dataset_mode='train'):
    if dataset_mode not in ['train', 'test', 'val']:
        raise NotImplementedError

    DatasetType = get_dataset(opt.dataset_type)
    if DatasetType is None:
        raise NotImplementedError('Not recognized dataset type')
    
    opt.dataset_param['mode'] = dataset_mode
    my_dataset = DatasetType(opt.dataset_param)
    opt['dataset_size'] = len(my_dataset)

    do_shuffle = True
    num_batch = opt.num_batch
    if not dataset_mode == 'train':
        do_shuffle = False
        num_batch = 1

    ### Dataset loader
    ds_collate_fn = None
    num_workers = 1
    if hasattr(my_dataset, 'collate_fn'):
        ds_collate_fn = my_dataset.collate_fn
        num_workers = 8

    my_dataloader = DataLoader(
            my_dataset, 
            collate_fn=ds_collate_fn,
            shuffle=do_shuffle, 
            batch_size=num_batch, 
            pin_memory=True, 
            num_workers=num_workers
    )

    return my_dataloader