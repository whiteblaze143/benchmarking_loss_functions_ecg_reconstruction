import torch
import numpy as np
import yaml
import wandb
from utils.yaml_config_hook import yaml_config_hook
import argparse
from data_process import get_VD, get_TD
from model import KEREPESModel, train_kerepes_model
from contrastive import contrastive_loss_function, SpectralContrastiveLoss, SimCLRLoss
from reconstruction import AE_loss, kpca_loss
from utils import representer_point_interpretabiliy, log_spectrum_plot_wandb
from non_contrastive import BT_loss, VICReg_loss, BYOL_loss


class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

def load_config(config_file='config.yaml'):
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f) 
    return Config(config_dict) 

def main():
    wandb.login()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="config.yaml", type=str, help="Path to the YAML config file")
    
    args, unknown = parser.parse_known_args()
    config = yaml_config_hook(args.config_file)


    for k, v in config.items():
        if f"--{k}" not in [action.option_strings[0] for action in parser._actions]:
            # parser.add_argument(f"--{k}", default=v, type=type(v))
            if isinstance(v, bool):
                parser.add_argument(f"--{k}", type=lambda x: (str(x).lower() == 'true'), default=v)
            else:
                parser.add_argument(f"--{k}", default=v, type=type(v))
  
    args = parser.parse_args()
    args_dict = vars(args)

    for key, value in args_dict.items():
        if isinstance(value, str) and value.lower() in ['none', 'null']:
            setattr(args, key, None)
            
    config.update(args_dict)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    
    with wandb.init(project="KEREPES_blck", config=config):
        cnf = wandb.config

        if cnf.kcache:
            metadata = torch.load(cnf.Metadata_path, weights_only=False)
            y_train = metadata["y_train"]
            y_valid = metadata["y_valid"]
            y_test = metadata["y_test"]
            C_var = metadata["landmarks"]
            y_C = metadata["y_landmarks"]

            data = (y_train, y_valid, y_test, C_var, y_C)
        else:
            if cnf.data_mod == 'img':
                train_loader, posX_loader, valid_loader, test_loader, C_var, y_C = get_VD(cnf)
                data = (train_loader, posX_loader, valid_loader, test_loader, C_var, y_C)

            elif cnf.data_mod == 'tab':
                train_loader, valid_loader, test_loader, C_var, y_C = get_TD(cnf)
                data = (train_loader, valid_loader, test_loader, C_var, y_C)

        if cnf.loss_f == 'simple':
            criterion = contrastive_loss_function
        elif cnf.loss_f == 'spectral':
            criterion = SpectralContrastiveLoss(cnf.lambda_reg)
        elif cnf.loss_f == 'simclr':
            criterion = SimCLRLoss(cnf.tau)
        elif cnf.loss_f == 'ae':
            criterion = AE_loss
        elif cnf.loss_f == 'bt':
            criterion = BT_loss
        elif cnf.loss_f == 'vicreg':
            criterion = VICReg_loss
        elif cnf.loss_f == 'kpca':
            criterion = kpca_loss
        elif cnf.loss_f == 'byol':
            criterion = BYOL_loss
        
        model = KEREPESModel(
            M=C_var.shape[0], k=cnf.k, d=C_var.shape[1], 
            device=cnf.device, loss_f=cnf.loss_f, 
            criterion=criterion,
            initialization=cnf.init_meth,
            )
        

        outputs = train_kerepes_model(model, cnf, data)

        if outputs[0] is None:
            import sys
            sys.exit(0)


if __name__ == '__main__':

    main()