import torch
import yaml
import time

from omegaconf import DictConfig, OmegaConf, open_dict
import hydra

# custom imports
#%% custom imports
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

import fourierimaging as fi

from fourierimaging.utils.helpers import load_model, init_opt, fix_seed
from fourierimaging.utils import datasets as data
import fourierimaging.train as train



@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: DictConfig) -> None:
    print(OmegaConf.to_yaml(conf))

    #%% fix random seed
    fix_seed(conf.seed)

    #%% get train, validation and test loader
    train_loader, valid_loader, test_loader = data.load(conf.dataset)

    #%% define the model
    if conf.CUDA.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda" + ":" + str(conf.CUDA.cuda_device))
    else:
        device = "cpu"
    
    with open_dict(conf):
        conf.train.device = device

        if conf.model.type == 'simple_cnn':
            time_str = time.strftime("%Y%m%d-%H%M%S")
            save_name = 'saved_models/' + 'simple_cnn'
            if conf.model.spectral.use:
                save_name += '-' +\
                            conf.model.parametrization +\
                            '-' + str(conf.model.spectral.ksize[0])+'-'+str(conf.model.spectral.ksize[1])
            save_name += '-' + time_str
    
    model = load_model(conf).to(device)

    #%% Initialize optimizer and lamda scheduler
    opt, lr_scheduler = init_opt(model, conf['train']['opt'])
    # initalize history
    tracked = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
    history = {key: [] for key in tracked}
    trainer = train.trainer(model, opt, lr_scheduler, train_loader, valid_loader, conf['train'])
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    #%%
    print(50*'#')
    print('Starting training.')
    print(conf['model'])
    print('Total number of params: ' + str(total_params) + ' parameters')
    print('Number of trainable params: ' + str(total_trainable_params))
    
    for i in range(conf['train']['epochs']):
        print(50*".")
        print('Startig epoch: ' + str(i))
        print(50*"=")
        
        # train_step
        train_data = trainer.train_step()
        
        # validation step
        val_data = trainer.validation_step()
        
        # update history
        for key in tracked:
            if key in val_data:
                history[key].append(val_data[key])
            if key in train_data:
                history[key].append(train_data[key])

    tester = train.Tester(test_loader, conf['train'])
    tester(model)

    if conf['train']['save']:
        torch.save({
                'conf': conf,
                'history': history,
                'model_state_dict': model.state_dict(),
                }, save_name)
                
if __name__ == '__main__':
    main()
    