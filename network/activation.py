import torch
import numpy as np
import torch.nn as nn

sine_w0 = 30

def get_activation_with_init(func_type):
    if func_type == 'relu':
        return nn.ReLU(inplace=True), init_weights_normal_relu, None
    elif func_type == 'lrelu':
        return nn.LeakyReLU(inplace=True), init_weights_normal_LRelu, None
    elif func_type == 'selu':
        return nn.SELU(inplace=True), init_weights_normal_selu, None
    elif func_type == 'sigmoid':
        return nn.Sigmoid(), init_weights_normal_sigmoid, None
    elif func_type == 'sine':
        # return Sine(), siren_init, first_layer_siren_init
        return Sine(), siren_init, None
        # if init_type =='myinit':
        #     return Sine(), sine_init, first_layer_sine_init
        # elif init_type == 'siren_init':
        #     return Sine(), siren_init, first_layer_siren_init
        # elif init_type == 'siren_all_layer':
        #     return Sine(), siren_init, None
        # else:
        #     raise NotImplementedError
    else:
        raise NotImplementedError


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(sine_w0 * input)


def sine_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            
            cons = 2*sine_w0 / np.pi
            c = np.sqrt(6 / num_input) / cons
            m.weight.uniform_(-c, c)


def first_layer_sine_init_normal(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            cons = 2*sine_w0 / np.pi
            c = np.sqrt(1 / num_input) / cons
            m.weight.normal_(0, c)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            cons = 2*sine_w0 / np.pi
            c = np.sqrt(9 / num_input) / cons
            m.weight.uniform_(-c, c)


def siren_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            cons = 30.0
            c = np.sqrt(6 / num_input) / cons
            m.weight.uniform_(-c, c)


def first_layer_siren_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def init_weights_normal_relu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

def init_weights_normal_LRelu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=1e-2, mode='fan_in')

def init_weights_normal_selu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            # nonlinearity can be 'linear' or 'selu'
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='linear', mode='fan_in')

def init_weights_normal_sigmoid(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            # xavier_normal_
            nn.init.xavier_normal_(m.weight)