import os
import configargparse
import torch
import training, network, data, utils


p = configargparse.ArgumentParser(config_file_parser_class = configargparse.YAMLConfigFileParser)
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--root_path', type=str, default='./', help='root path')
p.add_argument('--description', type=str, default='', help='description')
p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

p.add_argument('--training', type=str, default=None, help='training parameters')
p.add_argument('--dataset', type=str, default=None, help='dataset parameters')
p.add_argument('--model', type=str, default=None, help='model config')
p.add_argument('--loss', type=str, default=None, help='loss parameters')

opt,_ = p.parse_known_args()
opt = utils.process_options(opt, mode='train')

### Train Dataset
train_dataloader = data.get_dataloader(opt, dataset_mode='train')
opt['train_dataloader'] = train_dataloader

### Validation Dataset
val_dataloader = data.get_dataloader(opt, dataset_mode='val')
opt['val_dataloader'] = val_dataloader

### define model and train
model = network.define_model(opt)
model.cuda()

opt['train_loss'] = training.config_loss(opt.loss)
training.train_model(opt, model)


# save the current config file to the output folder
src_config_path = list(p.get_source_to_settings_dict().keys())[-2].replace('config_file|', '')
src_config_path = os.path.join(opt.root_path, src_config_path)
cp_config_path = os.path.join(opt.log_path, 'config.yaml')
os.system(f'cp {src_config_path} {cp_config_path}')