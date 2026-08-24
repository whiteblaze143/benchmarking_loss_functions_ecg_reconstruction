import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from scripts.evaluate_comprehensive_registry import load_adapter

spec = {
    'id': 'factorial_msvae_1000000_s42',
    'kind': 'msvae',
    'checkpoint': 'checkpoints/factorial_msvae_1000000_s42.pt',
    'observed_leads': [0, 1, 7]
}
device = torch.device('cpu')
adapter = load_adapter(spec, device)
print('Successfully loaded adapter:', type(adapter))

dummy_target = torch.randn(2, 12, 5000)
recon = adapter.reconstruct(dummy_target)
print('Successfully reconstructed shape:', recon.shape)
print('Observed lead 0 max diff:', torch.max(torch.abs(recon[:, 0] - dummy_target[:, 0])).item())
print('Observed lead 1 max diff:', torch.max(torch.abs(recon[:, 1] - dummy_target[:, 1])).item())
print('Observed lead 7 max diff:', torch.max(torch.abs(recon[:, 7] - dummy_target[:, 7])).item())
print('Missing lead 8 max diff:', torch.max(torch.abs(recon[:, 8] - dummy_target[:, 8])).item())
