import os
import json
import yaml

wandb_dir = '/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/wandb'

for root, dirs, files in os.walk(wandb_dir):
    if 'config.yaml' in files and 'wandb-summary.json' in files:
        config_path = os.path.join(root, 'config.yaml')
        summary_path = os.path.join(root, 'wandb-summary.json')
        
        try:
            with open(config_path, 'r') as f:
                config_content = f.read()
                
            # Check if this run is related to fm_vae, ecgfm, hubert, or fm_checkpoint
            if any(k in config_content.lower() for k in ['fm_vae', 'fm_checkpoint', 'ecgfm', 'hubert']):
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                    
                print(f"--- Found relevant run in {root} ---")
                
                # Extract some config details
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    name = config.get('name', {}).get('value', 'Unknown')
                    model_family = config.get('model_family', {}).get('value', 'Unknown')
                    backbone = config.get('backbone', {}).get('value', 'Unknown')
                    print(f"Name: {name}, Family: {model_family}, Backbone: {backbone}")
                
                # Extract metrics
                mse = summary.get('val/mse', summary.get('val_mse', summary.get('val/reconstruction_loss', 'N/A')))
                r2 = summary.get('val/r2', summary.get('val_r2', 'N/A'))
                print(f"Metrics -> MSE: {mse}, R2: {r2}")
                print()
                
        except Exception as e:
            pass
