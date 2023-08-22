import os
from dotted.collection import DottedDict

def process_options(opt, mode='train'):
    res = DottedDict({})
    res['name'] = opt.experiment_name
    res['description'] = opt.description
    res['root_path'] = opt.root_path
    res['log_path'] = os.path.join(opt.root_path, opt.logging_root, opt.experiment_name)
    if mode == 'train':
        res.update(eval(opt.training))
        loss = eval(opt.loss)
        res['loss'] = loss
    elif mode == 'test':
        res.update(eval(opt.testing))
    else:
        raise NotImplementedError

    dataset = eval(opt.dataset)
    res['dataset_type'] = dataset.pop('type')
    dataset['mode'] = mode
    res['dataset_param'] = dataset

    model_config = eval(opt.model)
    model_type = model_config.pop('type')
    res['model_type'] = model_type
    res['model'] = model_config

    return res
