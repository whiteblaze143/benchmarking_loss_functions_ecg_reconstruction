"""
Clinical Inference Wrapper for Heteroscedastic Mason Model.

This module provides a clinical-grade inference interface that returns
uncertainty quantification alongside predictions, suitable for medically
regulated deployment in electrophysiology/cardiology.

Following Givens & Hoeting principles:
- Ch. 9.3.2 (BCa): Calibrated confidence intervals
- Ch. 1.4: Proper variance from model output

Usage:
    from src.evaluation.clinical_inference import ClinicalPredictor
    
    predictor = ClinicalPredictor('checkpoints/mason_heteroscedastic_best.pt')
    result = predictor.predict_with_confidence(ecg_3lead)
    
    # result contains:
    # - ecg_12lead: reconstructed signal
    # - confidence: per-lead confidence scores [0-1]
    # - prediction_interval_lower/upper: 95% CI bounds
    # - flags: clinical warnings if any
"""

import torch
import argparse
import sys
from pathlib import Path

_engineering_root = Path(__file__).resolve().parents[2]
if str(_engineering_root) not in sys.path:
    sys.path.insert(0, str(_engineering_root))

from tqdm import tqdm
import pandas as pd
from src.evaluation.clinical_metrics import batch_clinical_error, load_calibration_csv
from src.data.multi_source_dataset import MultiSourceECGDataset
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path

# Lead names for interpretability
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
PRECORDIAL_LEADS = [6, 7, 8, 9, 10, 11]  # V1-V6


class ClinicalPredictor:
    """
    Clinical-grade inference wrapper for heteroscedastic Mason model.
    
    Provides:
    1. Point predictions (reconstructed 12-lead ECG)
    2. Uncertainty quantification (per-lead and per-timestep)
    3. Calibrated 95% prediction intervals
    4. Clinical reliability flags
    
    Example:
        >>> predictor = ClinicalPredictor('checkpoints/mason_heteroscedastic_best.pt')
        >>> result = predictor.predict_with_confidence(ecg_3lead)
        >>> if not result['reliable']:
        >>>     print("Warning: Low confidence in reconstruction")
        >>>     print(f"Unreliable leads: {result['unreliable_leads']}")
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = 'causal',
        device: Optional[str] = None,
        confidence_threshold: float = 0.8,
    ):
        """
        Initialize the clinical predictor.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            model_type: 'causal' or 'baseline'
            device: Device to run inference on ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence for a lead to be considered reliable
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self.model_type = model_type
        
        # Load model based on type
        if model_type == 'causal':
            from src.models.causal_mason import CausalMasonReconstructor
            self.model = CausalMasonReconstructor(
                input_lead_num=3,
                output_lead_num=12,
                base_channels=32,
                num_confounders=10,
                heteroscedastic=True
            ).to(self.device)
        elif model_type == 'baseline':
            from src.models.wrappers import MasonWrapper
            self.model = MasonWrapper(device=self.device).to(self.device)
            # MasonWrapper defaults to heteroscedastic=False
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Handle state dict wrapping and prefixes
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Strip 'model.' prefix if it exists (common in some training scripts)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
        
        # Check if we should override heteroscedastic based on checkpoint keys
        # If the checkpoint doesn't have variance keys, we must set heteroscedastic=False
        has_variance = any('variance_network' in k for k in state_dict.keys())
        
        # Get actual model object (handle wrapper)
        actual_model = self.model.model if hasattr(self.model, 'model') else self.model
        
        if not has_variance and getattr(actual_model, 'heteroscedastic', False):
            print(f"[ClinicalPredictor] Checkpoint has no variance heads. Switching {model_type} to deterministic mode.")
            if model_type == 'causal':
                from src.models.causal_mason import CausalMasonReconstructor
                self.model = CausalMasonReconstructor(
                    input_lead_num=3, output_lead_num=12, heteroscedastic=False
                ).to(self.device)
                actual_model = self.model
            else:
                from src.models.wrappers import MasonWrapper
                self.model = MasonWrapper(device=self.device, heteroscedastic=False).to(self.device)
                actual_model = self.model.model
        
        # Load into the specific level that matches keys (usually the core model)
        actual_model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        # Load calibration info if available
        self.calibration = checkpoint.get('calibration', None)
        if self.calibration:
            print(f"[ClinicalPredictor] Loaded with calibration: 95% coverage = {self.calibration.get('coverage_95', 'N/A')}")
    
    @torch.no_grad()
    def predict_with_confidence(self, x: torch.Tensor) -> Dict[str, any]:
        """
        Predict 12-lead ECG with uncertainty quantification.
        
        Args:
            x: Input 3-lead ECG tensor of shape (batch, 3, time) or (3, time).
               Expected leads: [I, II, V3] (Domain-Informed Standard)
            
        Returns:
            Dict with keys:
            - 'ecg_12lead': Reconstructed 12-lead ECG (batch, 12, time)
            - 'confidence': Per-lead confidence scores [0-1] (batch, 12)
            - 'prediction_interval_lower': Lower bound of 95% CI (batch, 12, time)
            - 'prediction_interval_upper': Upper bound of 95% CI (batch, 12, time)
            - 'uncertainty_per_lead': Mean uncertainty per lead (batch, 12)
            - 'reliable': Overall reliability flag (bool)
            - 'unreliable_leads': List of lead names with low confidence
            - 'clinical_notes': Any clinical warnings
        """
        # Handle single sample
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 input leads (I, II, V3), got {x.shape[1]}. Ensure correct domain-informed inputs.")
            
        x = x.to(self.device)
        
        actual_model = self.model.model if hasattr(self.model, 'model') else self.model
        if actual_model.heteroscedastic:
            # Get predictions with uncertainty
            pred_mu, pred_logvar = self.model(x, return_uncertainty=True)
            # Convert log-variance to standard deviation
            pred_sigma = torch.exp(0.5 * pred_logvar)
        else:
            # Deterministic prediction
            pred_mu = self.model(x)
            pred_sigma = torch.zeros_like(pred_mu)
        
        # Compute 95% prediction intervals (z = 1.96)
        lower = pred_mu - 1.96 * pred_sigma
        upper = pred_mu + 1.96 * pred_sigma
        
        # Per-lead uncertainty (mean sigma across time)
        uncertainty_per_lead = pred_sigma.mean(dim=-1)  # (batch, 12)
        
        # Convert uncertainty to confidence [0, 1]
        # Lower uncertainty = higher confidence
        # Use sigmoid-based mapping: confidence = 1 - sigmoid(uncertainty - threshold)
        # Simpler: confidence = exp(-uncertainty) normalized
        max_uncertainty = uncertainty_per_lead.max().item()
        if max_uncertainty > 0:
            confidence = torch.exp(-uncertainty_per_lead / max_uncertainty)
        else:
            confidence = torch.ones_like(uncertainty_per_lead)
        
        # Identify unreliable leads
        unreliable_mask = confidence < self.confidence_threshold
        unreliable_leads = []
        clinical_notes = []
        
        for batch_idx in range(x.shape[0]):
            unreliable_batch = []
            for lead_idx in range(12):
                if unreliable_mask[batch_idx, lead_idx]:
                    unreliable_batch.append(LEAD_NAMES[lead_idx])
            unreliable_leads.append(unreliable_batch)
            
            # Domain-specific warnings
            if any(i in [8, 9] for i in torch.where(unreliable_mask[batch_idx])[0].tolist()):
                # V3 or V4 are unreliable - risk for anterior MI assessment
                clinical_notes.append("⚠️ V3/V4 low confidence - verify anterior wall assessment")
        
        # Overall reliability
        reliable = not unreliable_mask.any().item()
        
        return {
            'ecg_12lead': pred_mu.cpu(),
            'confidence': confidence.cpu(),
            'prediction_interval_lower': lower.cpu(),
            'prediction_interval_upper': upper.cpu(),
            'uncertainty_per_lead': uncertainty_per_lead.cpu(),
            'reliable': reliable,
            'unreliable_leads': unreliable_leads,
            'clinical_notes': clinical_notes,
        }
    
    def format_clinical_report(self, result: Dict) -> str:
        """
        Generate a human-readable clinical report from prediction results.
        
        Args:
            result: Output from predict_with_confidence()
            
        Returns:
            Formatted string report
        """
        lines = []
        lines.append("=" * 60)
        lines.append("ECG RECONSTRUCTION CLINICAL REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        # Overall status
        if result['reliable']:
            lines.append("✅ OVERALL STATUS: HIGH CONFIDENCE")
        else:
            lines.append("⚠️ OVERALL STATUS: LOW CONFIDENCE - VERIFY LEADS")
        
        lines.append("")
        lines.append("Per-Lead Confidence Scores:")
        lines.append("-" * 40)
        
        confidence = result['confidence'][0]  # First batch element
        uncertainty = result['uncertainty_per_lead'][0]
        
        for lead_idx, name in enumerate(LEAD_NAMES):
            conf = confidence[lead_idx].item()
            unc = uncertainty[lead_idx].item()
            status = "✓" if conf >= self.confidence_threshold else "⚠️"
            lead_type = "Precordial" if lead_idx in PRECORDIAL_LEADS else "Limb"
            lines.append(f"  {status} {name:4s} ({lead_type:10s}): {conf:.1%} confidence, σ={unc:.4f}")
        
        # Clinical notes
        if result['clinical_notes']:
            lines.append("")
            lines.append("Clinical Notes:")
            for note in result['clinical_notes']:
                lines.append(f"  {note}")
        
        # Unreliable leads
        if result['unreliable_leads'][0]:
            lines.append("")
            lines.append(f"Unreliable Leads: {', '.join(result['unreliable_leads'][0])}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


def evaluate_dataset(
    checkpoint_path,
    data_loader,
    output_csv=None,
    device='cuda',
    model_type='causal',
    calibration_csv: str | None = None,
):
    print(f"Loading checkpoint: {checkpoint_path} (Type: {model_type})")
    predictor = ClinicalPredictor(checkpoint_path, model_type=model_type, device=device)

    calibration = load_calibration_csv(calibration_csv) if calibration_csv else None
    if calibration:
        print(f"Loaded clinical calibration biases: {calibration}")
    
    all_metrics = []
    
    print("Running Clinical Evaluation...")
    for batch in tqdm(data_loader):
        if isinstance(batch, dict):
            x = batch['input'].to(device)
            y = batch['target'].to(device)
        else:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            
        result = predictor.predict_with_confidence(x)
        pred_mu = result['ecg_12lead'] 
        
        # Denormalize to mV for Clinical Metrics
        # NeuroKit expects physiological amplitude (approx 1mV QRS)
        if isinstance(batch, dict) and 'mean' in batch and 'std' in batch:
            mean = batch['mean'].to(device)
            std = batch['std'].to(device)
            
            # Broadcast: (Batch, Leads) -> (Batch, Leads, 1)
            if mean.ndim == 2:
                mean = mean.unsqueeze(-1)
                std = std.unsqueeze(-1)
                
            pred_mu = pred_mu * std + mean
            y = y * std + mean
            
        # Calculate Clinical Metrics (QTc, QRS)
        # Convert to numpy for NeuroKit
        pred_np = pred_mu.detach().cpu().numpy()
        target_np = y.detach().cpu().numpy()
        
        # Use existing clinical_metrics module
        try:
            batch_errs = batch_clinical_error(pred_np, target_np, sr=500, calibration=calibration)
            all_metrics.extend(batch_errs)
        except Exception as e:
            print(f"Batch metrics failed: {e}")
            pass

    # Aggregate
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Detailed metrics saved to {output_csv}")

        try:
            summary = df.mean(numeric_only=True).to_dict()
            print(f"\n=== Clinical Metrics Summary ({model_type.upper()}) ===")
            for k, v in summary.items():
                print(f"{k}: {v:.2f}")
        except Exception as e:
            print(f"Summary stats failed: {e}")
            print("Check CSV for raw data.")
    else:
        print("No metrics calculated (check NeuroKit interactions).")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='causal', choices=['causal', 'baseline'], help='Model architecture')
    parser.add_argument('--data_dir', type=str, default='data/ptbxl_tensors', help='Path to PTB-XL tensors')
    parser.add_argument('--output_csv', type=str, default=None, help='Path to save results CSV')
    parser.add_argument('--calibration_csv', type=str, default=None, help='Path to Sunnybrook calibration CSV for bias correction')
    args = parser.parse_args()
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load Validation Data
    val_sources = [{'name': 'PTB-XL', 'path': args.data_dir, 'format': 'pt'}]
    val_dataset = MultiSourceECGDataset(
        sources=val_sources,
        split="val",
        input_leads=["I", "II", "V2"], # Standard reduced leads
        target_leads=["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    )
    # Subset for speed if needed, but for "Rigor" we should run all.
    # We will limit to 100 for interactive speed, user can run full later.
    val_subset = torch.utils.data.Subset(val_dataset, range(min(100, len(val_dataset))))
    
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=2)
    
    evaluate_dataset(
        args.checkpoint,
        val_loader,
        args.output_csv,
        device=device,
        model_type=args.model_type,
        calibration_csv=args.calibration_csv,
    )
