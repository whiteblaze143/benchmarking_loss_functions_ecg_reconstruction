import torch
import numpy as np
import wandb
import os
from simple_nn import train
from utils import yaml_config_hook
import argparse
from data_process import get_VD, get_TD
from simple_nn import BT_loss, VICReg_loss, \
AE_loss_nn, SimCLRLoss, spectral_loss, simple_contrastive_loss, byol_loss_fn

wandb.login()

def main():
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("nn_config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    with wandb.init(project="NN_comp", config=args):
        cnf = wandb.config
        if cnf.data_mod == 'img':
            posX_loader, valid_loader, test_loader = get_VD(cnf)

        elif cnf.data_mod == 'tab':
            posX_loader, valid_loader, test_loader = get_TD(cnf)

        
        if cnf.loss_f == 'simple':
            criterion = simple_contrastive_loss
        elif cnf.loss_f == 'spectral':
            criterion = spectral_loss
        elif cnf.loss_f == 'simclr':
            criterion = SimCLRLoss(cnf.tau)
        elif cnf.loss_f == 'ae':
            criterion = AE_loss_nn
        elif cnf.loss_f == 'bt':
            criterion = BT_loss
        elif cnf.loss_f == 'vicreg':
            criterion = VICReg_loss
        elif cnf.loss_f == 'byol':
            criterion = byol_loss_fn

        data = (posX_loader, valid_loader, test_loader)
        train_acc, test_acc, NN_model = train(data, criterion, cnf)

        if cnf.dataset == 'adult':
            save_dir = "checkpoints"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"model_{cnf.dataset}_{cnf.loss_f}.pth")

            print(f"Saving model to {save_path}...")
            torch.save({
                'model_state_dict': NN_model.state_dict(),
                'config': dict(cnf),
                'train_acc': train_acc,
                'test_acc': test_acc
            }, save_path)

        wandb.log({
            "Train Accuracy": train_acc,
            "Test Accuracy": test_acc
        })

if __name__ == '__main__':
    main()