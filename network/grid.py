import torch
import torch.nn as nn

from mlp import MLP
from grid_pooling_func import AvgPoolingModule


def get_conv_activation(name):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'lrelu':
        return nn.LeakyReLU(inplace=True)
    elif name == 'selu':
        return nn.SELU(inplace=True)
    else:
        raise NotImplementedError('Not supported activation')


def define_convs(conv_dim, num_conv, activation, latent_size, kernel_size, padding=0):
    if num_conv == 0:
        return None

    conv = getattr(nn, f'Conv{conv_dim}d')
    seq = []
    ls = latent_size
    for i in range(num_conv):
        seq.append(conv(ls, ls, kernel_size=kernel_size, padding=padding))
        seq.append(get_conv_activation(activation))

    return nn.Sequential(*seq)


class PointGridEncoder(nn.Module):
    def __init__(self, params):
        super(PointGridEncoder, self).__init__()
        p = params
        self.grid_size = p.grid_size
        self.mlp_feat = MLP(**p.mlp)
        self.avg_pooling = AvgPoolingModule(self.grid_size)

        if 'max_pooling' not in p:
            self.use_max_pooling = False
            self.knn_feat = nn.Sequential(
                    nn.Conv1d(p.N_knn, p.N_knn, 1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv1d(p.N_knn, 1, 1)
                )
        else:
            self.use_max_pooling = True

        self.grid_feat = define_convs(**p.grid_conv)


    def forward(self, model_input):
        # pos: (N_pts, N_knn, 3), 
        pc_KNN_pos = model_input['pc_KNN_pos']
        points = model_input['info']['points']
        feat = self.mlp_feat.forward_simple(pc_KNN_pos)
        # After MLP  feat: (N_pts, N_knn, N_gfeat)
        if self.use_max_pooling:
            feat = torch.max(feat, dim=1, keepdim=True)[0]
        else:
            feat = self.knn_feat(feat)
        # After Conv  feat: (N_pts, 1, N_gfeat)
        Nf = feat.shape[-1]
        Np = pc_KNN_pos.shape[0]
        feat = feat.view((Np, Nf))
        
        temp_grid = self.avg_pooling(feat, points)
        if self.grid_feat is not None:
            temp_grid = temp_grid.permute((3,0,1,2))
            # (N_gfeat, k,k,k)
            feature_grid = self.grid_feat(temp_grid).permute((1,2,3,0))
            # final feature: (k,k,k, N_gfeat）
        else:
            feature_grid = temp_grid

        return feature_grid


if __name__ == '__main__':
    from dotted.collection import DottedDict
    params = DottedDict({
        'max_pooling': True,
        'grid_size': 8,
        'mlp': {
            'size': [3,128,128],
            'activation_type': 'lrelu',
            'num_pos_encoding': -1
        },
        'grid_conv': {
            'latent_size': 128,
            'conv_dim': 3,
            'num_conv': 3,
            'activation': 'lrelu',
            'kernel_size': 3,
            'padding': 1,
        }
    })

    points = torch.rand(10, 3)*2 - 1
    model_input = {
        'pc_KNN_pos': torch.rand((10, 4, 3)),
        'points': points
    }

    encoder = PointGridEncoder(params)
    feat = encoder.forward(model_input)
    print(feat.shape)