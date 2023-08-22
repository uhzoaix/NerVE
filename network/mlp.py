import torch
import torch.nn as nn

from activation import *
from pos_encoding import *

class MLP(nn.Module):
    """
    customed MLP
    :param size (input_size, hidden_size1, ..., output_size)
    :param activation_type activation function, ['relu', 'sine']
    :param bias nn.Linear w/wo bias
    :param output_softmax whether to add softmax to output
    
    """
    def __init__(self, size, activation_type='relu', bias=True, 
        init_type=None, num_pos_encoding=-1):
        super(MLP, self).__init__()
        self.size = size
        self.num_layer = len(size) - 1
        self.bias = bias

        self.func, weights_init, first_layer_init = get_activation_with_init(activation_type)
        
        self.fc_layers = []
        if num_pos_encoding > 0:
            pos_encoder = PosEncoding(d_in=size[0], num_freqs=num_pos_encoding)
            size[0] = pos_encoder.d_out
            # first_layer_init = first_layer_sine_init_normal
            first_layer_init = None
            self.fc_layers.append(pos_encoder)

        for i in range(self.num_layer):
            fc = nn.Linear(
                size[i], size[i+1], bias=self.bias)
            self.fc_layers.append(fc)

            if i < self.num_layer - 1:
                # not the last one
                self.fc_layers.append(self.func)

        self.net = nn.Sequential(*self.fc_layers)
        
        # weights initializaiton
        self.net.apply(weights_init)

        # NOTICE: first layer init deprecated now
        if first_layer_init is not None:
            self.net[0].apply(first_layer_init)


    def forward(self, x):
        # x is of shape (Nb, batch_num, input_size)
        coords_org = x.clone().detach().requires_grad_(True)
        coords = coords_org

        out = self.net(coords)
        return {'model_input': coords_org, 'model_pred': out}


    def forward_simple(self, x):
        return self.net(x)

    def layer_feature(self, x, k):
        return self.net[:k](x)




if __name__ == '__main__':
    mlp = MLP((3, 4, 5))
    print(mlp)
    print('num of param: ', len(list(mlp.parameters())))
    a = torch.rand(10, 3)
    b = mlp(a)
    print(b.shape)