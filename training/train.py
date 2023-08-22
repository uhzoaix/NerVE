'''Implements a generic training loop.
'''

import os, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from time import time


def get_optimizer(opt, model):
    p = opt.optim
    if p.type == 'Adam':
        return {
            'optimizer' : torch.optim.Adam(params=model.parameters(), 
                lr=p.lr, betas=(p.beta1, p.beta2), amsgrad=p.amsgrad),
            'epoch_lr': None,
            'step_lr' : None
        }
    elif p.type == 'SGD':
        res = {}
        optim = torch.optim.SGD(model.parameters(), lr=p.lr, momentum=p.momentum)
        res['optimizer'] = optim
        res['epoch_lr'] = None
        res['step_lr'] = None
        if p.lr_scheduler == 'MultiStep':
            lr_sch = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=p.milestones, gamma=p.gamma)
            res['epoch_lr'] = lr_sch
        elif p.lr_scheduler == 'ROP':
            lr_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=p.factor, patience=p.patience)
            res['epoch_lr'] = lr_sch
        elif p.lr_scheduler == 'CLR':
            lr_sch = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=p.base_lr, max_lr=p.max_lr, step_size_up=p.step)
            res['step_lr'] = lr_sch
        else:
            raise NotImplementedError('Not implemented lr scheduler')
        return res
    else:
        raise NotImplementedError('Not implemented optimizer type')


def train_model(opt, model):
    res = get_optimizer(opt, model)
    optim, epoch_lr, step_lr = res['optimizer'], res['epoch_lr'], res['step_lr']

    model_dir = opt.log_path
    os.makedirs(model_dir, exist_ok=True)
    summaries_dir = os.path.join(model_dir, 'summaries')
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    os.makedirs(summaries_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    writer = SummaryWriter(summaries_dir)
    train_dataloader = opt['train_dataloader']
    val_dataloader = opt['val_dataloader']
    train_loss_fn = opt['train_loss']

    total_steps = 0
    with tqdm(total=len(train_dataloader) * opt.num_epochs) as pbar:
        for epoch in range(opt.num_epochs):
            if not epoch % opt.epochs_til_ckpt and epoch:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))

            # -----------------------
            # training
            # -----------------------
            for data in train_dataloader:
                model_input, gt, info = data
                model_input = {key: val.cuda() for key,val in model_input.items()}
                gt = {key: val.cuda() for key,val in gt.items()}
                model_input['info'] = info
                gt['info'] = info
                model_output = model(model_input)
                # losses = train_loss_fn(model_output, gt, opt.loss)                
                losses = train_loss_fn(model_output, gt)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    factor = opt.loss[loss_name].factor
                    loss_name = f'{loss_name}(X{factor})'
                    writer.add_scalar(loss_name, loss, total_steps)
                    train_loss += loss

                writer.add_scalar("total_train_loss", train_loss, total_steps)

                optim.zero_grad()
                train_loss.backward()

                if opt.clip_grad:
                    if isinstance(opt.clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.clip_grad)

                optim.step()

                pbar.update(1)
                # default to be only one parameter group
                current_lr = optim.param_groups[0]['lr']
                writer.add_scalar('lr', current_lr, total_steps)

                if not total_steps % opt.steps_til_summary:
                    message = "Epoch {}|Iter:{}, Loss {:0.4f}, ".format(
                        epoch, total_steps, train_loss)
                    for name, loss in losses.items():
                        message = message + '{}(X{}): {:.4f}, '.format(name, opt.loss[name].factor, loss.item())
                    tqdm.write(message)

                if step_lr is not None:
                    step_lr.step()

                total_steps += 1
                
            # -----------------------
            # Validation and epoch lr scheduler
            # -----------------------
            vals = {}
            if opt.val_type == 'None':
                continue

            with torch.no_grad():
                t1 =time()
                # print('Validation Start')
                model.eval()
                for data in val_dataloader:
                    model_input, gt, info = data
                    model_input = {key: val.cuda() for key,val in model_input.items()}
                    gt = {key: val.cuda() for key,val in gt.items()}
                    model_input['info'] = info

                    res = model.forward_val(model_input, gt)
                    if len(vals) == 0:
                        for key,val in res.items():
                            vals[key] = [val]
                    else:
                        for key,val in res.items():
                            vals[key].append(val)

                for key,val in vals.items():
                    writer.add_scalar(f'val_mean_{key}', np.mean(val), epoch)

                if not epoch % opt.epochs_til_showval or epoch == opt.num_epochs-1:
                    for key,val in vals.items():
                        val = np.asarray(val)
                        print('Epoch {} | {} mean:{:.6f}, min:{:.6f}, max:{:.6f}'.format(
                            epoch, key, val.mean(), val.min(), val.max())  )

                    print(f'Validation Done, time cost: {time()-t1}')
                    if epoch == opt.num_epochs-1:
                        val_path = os.path.join(model_dir, 'final_val.pkl')
                        with open(val_path, 'wb') as f:
                            pickle.dump(vals, f)
                
                model.train()

            if epoch_lr is not None:
                # val_loss = np.mean(errs)
                if opt.optim.lr_scheduler == 'MultiStep':
                    epoch_lr.step()
                # elif opt.optim.lr_scheduler == 'ROP':
                #     epoch_lr.step(val_loss)
                else:
                    raise NotImplementedError

        torch.save(model.state_dict(), 
            os.path.join(checkpoints_dir, 'model_final.pth'))
