import torch
import torch.nn as nn

class LossHandler():
    def __init__(self):
        self.loss_names = [
            'cube_loss', 'face_loss', 'pair_loss', 'point_loss', 'tan_loss',
            'batch_cube_loss', 'batch_face_loss', 'batch_point_loss',
        ]
        self.metric_types = ['NLL', 'BCE', 'L1', 'L2', 'cos_sim', 'None']

    def get_metric_fn(self, loss_config):
        metric_type = loss_config.metric
        if metric_type not in self.metric_types:
            raise NameError('Not supported metric')

        loss = None
        if metric_type == 'NLL':
            loss = nn.NLLLoss()
            if 'weight' in loss_config:
                loss.weight = torch.Tensor(loss_config.weight).cuda()
            if 'ignore_idx' in loss_config:
                loss.ignore_index = 0
        elif metric_type == 'BCE':
            loss = nn.BCELoss()
        elif metric_type == 'L1':
            loss = nn.L1Loss()
        elif metric_type == 'L2':
            loss = nn.MSELoss()
        elif metric_type == 'cos_sim':
            loss = nn.CosineSimilarity(eps=1e-6)
        elif metric_type == 'None':
            loss = None
        else:
            raise NotImplementedError('Not implemented metric')

        return loss


    def parse_config(self, loss_schedule):
        self.loss_fn = {}
        for name, loss_config in loss_schedule.items():
            if name not in self.loss_names:
                raise NameError('Not supported loss')
            
            metric_fn = self.get_metric_fn(loss_config)
            self.loss_fn[name] = {'metric_fn': metric_fn, 'factor': loss_config.factor}
        
        return

    def cube_loss(self, output, gt, metric_fn):
        out_gc = output['pc_cube']
        gt_gc = gt['pc_cube']
        return metric_fn(out_gc, gt_gc)

    def face_loss(self, output, gt, metric_fn):
        out_gf = output['pc_face']
        gt_gf = gt['pc_face']
        return metric_fn(out_gf, gt_gf)

    def pair_loss(self, output, gt, metric_fn):
        out_gp = output['pc_pair']
        gt_gp = gt['pc_pair']
        return metric_fn(out_gp, gt_gp)

    def point_loss(self, output, gt, metric_fn):
        out_p = output['pc_point']
        gt_p = gt['pc_point']
        return metric_fn(out_p, gt_p)


    def __call__(self, output, gt):
        res = {}
        for name, loss in self.loss_fn.items():
            func = getattr(self, name)
            loss_term = func(output, gt, loss['metric_fn'])
            res[name] = loss['factor']*loss_term

        return res

