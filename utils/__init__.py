import os, sys, yaml, pickle
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

import torch
import network
import os.path as op
import numpy as np

from options import *


def load_model(log_path, device, checkpoint='final'):
    config_path = op.join(log_path, 'config.yaml')
    opt = yaml.safe_load(open(config_path))
    opt = DottedDict(opt)
    opt['model_type'] = opt.model.type
    opt['dataset_type'] = opt.dataset.type

    if checkpoint == 'final':
        ckpt_name = 'model_final.pth'
    else:
        ckpt_name = 'model_epoch_%04d.pth' % int(checkpoint)

    model = network.define_model(opt)
    checkpoint_path = op.join(log_path, f'checkpoints/{ckpt_name}')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def nerve2pwl(pred_nerve, output_path):
    cube_idx = pred_nerve['cube_idx']
    faces = pred_nerve['cube_faces']
    cube_pts = pred_nerve['cube_points']

    k = pred_nerve['grid_size']
    k_base = np.asarray([1, k, k**2])
    k_idx = cube_idx @ k_base
    dict_cube_idx = {idx: i for i,idx in enumerate(k_idx)}

    edges = []
    for idx, face_status in zip(cube_idx, faces):
        cid = idx @ k_base
        for num in range(3):
            if not face_status[num]:
                continue
            kidx = idx.copy()
            kidx[num] -= 1
            kidx = kidx @ k_base
            if kidx not in dict_cube_idx:
                continue

            edges.append([dict_cube_idx[cid], dict_cube_idx[kidx]])

    pwl_curve = {
        'points': cube_pts,
        'edges': edges
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(pwl_curve, f)
    
    return pwl_curve