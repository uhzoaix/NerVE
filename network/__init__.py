import os, sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

import model


def define_model(opt):
    net_name = '%sNet' % opt.model_type
    if hasattr(model, net_name):
        return getattr(model, net_name)(opt)

    return None