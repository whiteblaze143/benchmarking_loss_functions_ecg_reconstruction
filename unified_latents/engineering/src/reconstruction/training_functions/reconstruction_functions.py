import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def process_element(input_leads,
                    output_leads,
                    min_value: float,
                    amplitude: float,
                    device):
    
    model_input = [torch.unsqueeze(torch.unsqueeze(torch.from_numpy(lead).float().to(device), 0), 0) for lead in input_leads]

    model_target = [torch.from_numpy(lead * amplitude + min_value).float().to(device).detach() for lead in output_leads]

    return model_input, model_target


def deprocess_element(model_output):

    return [lead.detach().cpu().numpy() for lead in model_output]


def process_batch(batch,
                  input_lead_num: int,
                  output_lead_num: int,                  
                  min_value: float,
                  amplitude: float,
                  device):

    model_input = [[] for _ in range(input_lead_num)]
    model_target = [[] for _ in range(output_lead_num)]

    for input_leads, output_leads, _, _, _ in batch:

        for lead_idx, lead in enumerate(input_leads):

            model_input[lead_idx].append(torch.unsqueeze(torch.from_numpy(lead).float().to(device), 0))

        for lead_idx, lead in enumerate(output_leads):

            model_target[lead_idx].append(torch.from_numpy(lead * amplitude + min_value).float().to(device))

    model_input = [torch.stack(lead) for lead in model_input]
    model_target = [torch.stack(lead).detach() for lead in model_target]

    return model_input, model_target


def element_crosscov_function(model_output, model_target, lead_num: int):
    
    element_crosscov = 0

    for output_lead, target_lead in zip(model_output, model_target):

        lead_crosscov = torch.mean( (output_lead - output_lead.mean()) * (target_lead - target_lead.mean()) ) / torch.sqrt( torch.mean( (output_lead - output_lead.mean()) ** 2) * torch.mean( (target_lead - target_lead.mean()) ** 2 ) )
        
        element_crosscov += lead_crosscov / lead_num

    return element_crosscov



def element_r2_function(model_output, model_target, lead_num: int):
    
    element_r2 = 0

    for output_lead, target_lead in zip(model_output, model_target):

        lead_ssr = ((output_lead - target_lead)) ** 2
        
        lead_sst = ((target_lead - target_lead.mean())) ** 2
        
        lead_r2 = 1 - torch.sum(lead_ssr) / torch.sum(lead_sst)
        
        element_r2 += lead_r2 / lead_num

    return element_r2


def batch_r2_function(model_output, model_target, lead_num: int, batch_size: int, compute_loss_per_element: bool):
    
    batch_r2 = 0

    r2_per_element = np.zeros(batch_size)

    for output_lead, target_lead in zip(model_output, model_target):

        if compute_loss_per_element:
            
            lead_ssr = torch.sum( ( (output_lead - target_lead) ) ** 2, 1)
        
            lead_sst = torch.sum( ( (target_lead - target_lead.mean()) ) ** 2, 1)
            
            lead_r2 = 1 - lead_ssr / lead_sst

            batch_r2 += lead_r2.mean() / lead_num

            r2_per_element += lead_r2.detach().cpu().numpy() / lead_num

        else:

            lead_ssr = torch.sum( ( (output_lead - target_lead) ) ** 2, 1)
        
            lead_sst = torch.sum( ( (target_lead - target_lead.mean()) ) ** 2, 1)
            
            lead_r2 = 1 - lead_ssr / lead_sst

            batch_r2 += lead_r2.mean() / lead_num

    return - batch_r2, - r2_per_element


def element_mse_function(model_output, model_target, lead_num: int, qrs_times: np.ndarray = None, sample_num = None):
    
    element_loss = 0
    losses_per_sample = []
    distances_per_sample = []
    leads_per_sample = []

    for lead_idx, output_lead, target_lead in zip(range(lead_num), model_output, model_target):

        lead_loss = ((output_lead - target_lead) ) ** 2

        element_loss += lead_loss.mean() / lead_num

        if qrs_times is not None:

            for i in range(len(lead_loss)):

                dist_idx = np.argmin(np.abs(qrs_times-i))

                dist = (i - qrs_times[dist_idx]) / (sample_num / 10)

                losses_per_sample.append(lead_loss[i].item()) 

                distances_per_sample.append(dist)

                leads_per_sample.append(lead_idx)

    return element_loss, losses_per_sample, distances_per_sample, leads_per_sample



def batch_mse_function(model_output, model_target, lead_num: int, batch_size: int, compute_loss_per_element: bool):

    batch_loss = 0

    loss_per_element = np.zeros(batch_size)

    for output_lead, target_lead in zip(model_output, model_target):

        if compute_loss_per_element:

            lead_loss = torch.mean( ( (output_lead - target_lead) ) ** 2, 1)

            batch_loss += lead_loss.mean() / lead_num

            loss_per_element += lead_loss.detach().cpu().numpy() / lead_num

        else:

            lead_loss = torch.mean( ( (output_lead - target_lead) ) ** 2, 1)

            batch_loss += lead_loss.mean() / lead_num

    return batch_loss, loss_per_element

class CardiacVAELoss(nn.Module):
    """
    ELBO loss for the hierarchical cardiac state VAE.
    ℒ = Reconstruction Loss + β_g * KL_global + β_l * KL_local
    """
    def __init__(self, beta_global=0.01, beta_local=0.01, use_r2_loss=False):
        super().__init__()
        self.beta_g = beta_global
        self.beta_l = beta_local
        self.use_r2_loss = use_r2_loss
    
    def forward(self, recon, target, mu_g, logvar_g, mu_l, logvar_l, recon_indices=None):
        if recon_indices is not None:
            r_target = target[:, recon_indices, :]
            r_recon = recon[:, recon_indices, :]
        else:
            r_target = target
            r_recon = recon

        if self.use_r2_loss:
            # recon, target should be (B, L, T)
            # unbind leads to list for batch_r2_function
            n_leads = r_recon.shape[1]
            batch_sz = r_recon.shape[0]
            # Use the existing batch_r2_function from this file
            recon_loss, _ = batch_r2_function(r_recon.unbind(dim=1), r_target.unbind(dim=1), n_leads, batch_sz, False)
        else:
            recon_loss = F.mse_loss(r_recon, r_target)
        
        # KL divergence: KL(N(μ,σ²) || N(0,1))
        if mu_g is not None and logvar_g is not None:
            kl_global = -0.5 * torch.mean(1 + logvar_g - mu_g.pow(2) - logvar_g.exp())
        else:
            kl_global = torch.tensor(0.0, device=recon.device)
            
        if mu_l is not None and logvar_l is not None:
            kl_local = -0.5 * torch.mean(1 + logvar_l - mu_l.pow(2) - logvar_l.exp())
        else:
            kl_local = torch.tensor(0.0, device=recon.device)
        
        total_loss = recon_loss + self.beta_g * kl_global + self.beta_l * kl_local
        
        return total_loss, recon_loss, kl_global, kl_local
