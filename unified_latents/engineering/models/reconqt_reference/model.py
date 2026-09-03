import torch
import torch.nn as nn
from typing import Optional, Dict

from .theta import ThetaEncoder, SpatialCodebook
from .seresnext1d import SEResNeXt1D
from .feature_fusion import FeatureExtractionFusion
from .ecg_decoder import ECGDecoder
from .qt_head import QTTemporalHead


class SpatialReconstructionHead(nn.Module):
    """
    Spatial Query-Conditioned ECG Reconstruction Head.
    
    Implements viewpoint synthesis via channelwise multiplication:
      Z_l = Z * query_l
      hat_X_l = D_conv(Z_l)
    
    CRITICAL ARCHITECTURAL PROPERTY:
    All 12 target leads pass through ONE SINGLE SHARED ECGDecoder instance,
    reproducing the Panorama / 3DRECON-QT query mechanism.
    """

    def __init__(
        self,
        latent_channels: int = 256,
        theta_hidden: int = 128,
        target_length: int = 5000,
        code_mode: str = "theta",
    ):
        super().__init__()
        self.code_mode = code_mode
        self.theta_encoder = ThetaEncoder(1)
        self.codebook = SpatialCodebook(mode=code_mode)

        # Query MLP with Mish activation as visually confirmed in Fig. 2
        self.query_mlp = nn.Sequential(
            nn.Linear(12, theta_hidden),
            nn.Mish(),
            nn.Linear(theta_hidden, latent_channels),
        )

        # One single shared 1D convolutional upsampling decoder instance
        self.decoder = ECGDecoder(
            latent_channels=latent_channels,
            target_length=target_length,
        )

    def forward(
        self,
        latent: torch.Tensor,
        target_leads: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, C, T = latent.shape

        if target_leads is not None:
            # Single sampled target lead per sample: closest to public Panorama / Fig. 2
            # codes: [B, 12]
            codes = self.codebook(
                batch_size=B,
                theta_encoder=self.theta_encoder,
                target_indices=target_leads,
            )
            # query: [B, C]
            query = self.query_mlp(codes)

            # Elementwise multiplicative channel gating: [B, C, T]
            conditioned = latent * query[..., None]

            # Pass through the ONE shared decoder: [B, 1, target_length] -> [B, target_length]
            reconstructed = self.decoder(conditioned).squeeze(1)

            return {
                "reconstruction": reconstructed,
                "query_embedding": query,
                "target_leads": target_leads,
            }

        # Full 12-lead reconstruction (for evaluation or simultaneous_12 training)
        # [B, 12, 12]
        codes = self.codebook(
            batch_size=B,
            theta_encoder=self.theta_encoder,
            target_indices=None,
        )

        # [B, 12, C]
        query = self.query_mlp(codes)

        # Elementwise multiplicative channel gating: [B, 12, C, T]
        conditioned = latent[:, None, :, :] * query[:, :, :, None]

        # Reshape to batch-decode all 12 views through the shared decoder
        conditioned = conditioned.reshape(B * 12, C, T)

        # [B * 12, 1, target_length]
        reconstructed = self.decoder(conditioned)

        # [B, 12, target_length]
        reconstructed = reconstructed.reshape(B, 12, -1)

        return {
            "reconstruction": reconstructed,
            "query_embedding": query,
            "target_leads": None,
        }


class ThreeDReconQTReference(nn.Module):
    """
    3DRECON-QT Architectural Reference Model (Ansari et al., 2026; Chen et al., 2021).
    
    Pipeline:
      Single-Lead ECG (10s, 500Hz)
               |
          SE-ResNeXt-1D
               |
      Feature Extraction & Z1/Z2 Fusion (Temporal Attention on Z2)
               |
         Shared Latent Z
            /         \
     (No Theta)     (Theta-Conditioned)
         |                   |
      QT Head      Spatial Reconstruction Head
         |         (Z * query_l => Shared Conv Decoder)
         v                   v
     QT / QTc (ms)     Target Lead [B, T] or 12-Lead ECG [B, 12, T]
    """

    def __init__(
        self,
        input_channels: int = 1,
        latent_channels: int = 256,
        theta_hidden: int = 128,
        attention_heads: int = 8,
        code_mode: str = "theta",
        target_length: int = 5000,
        enable_qt_head: bool = False,
    ):
        super().__init__()
        self.code_mode = code_mode
        self.enable_qt_head = enable_qt_head

        self.encoder = SEResNeXt1D(in_channels=input_channels)

        self.fusion = FeatureExtractionFusion(
            encoder_channels=self.encoder.out_channels,
            latent_channels=latent_channels,
            attention_heads=attention_heads,
        )

        self.reconstruction_head = SpatialReconstructionHead(
            latent_channels=latent_channels,
            theta_hidden=theta_hidden,
            target_length=target_length,
            code_mode=code_mode,
        )

        if enable_qt_head:
            self.qt_head = QTTemporalHead(channels=latent_channels)
        else:
            self.qt_head = None

    def forward(
        self,
        source: torch.Tensor,
        rr_seconds: Optional[torch.Tensor] = None,
        target_leads: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # cuDNN 9.x can abort with ptrDesc->finalize() on 1D convolutions on the installed A100 stack.
        # Restrict execution to ATen native CUDA convolutions when running on GPU.
        if source.is_cuda and torch.backends.cudnn.is_available():
            with torch.backends.cudnn.flags(enabled=False):
                return self._forward_impl(source, rr_seconds=rr_seconds, target_leads=target_leads)
        return self._forward_impl(source, rr_seconds=rr_seconds, target_leads=target_leads)

    def _forward_impl(
        self,
        source: torch.Tensor,
        rr_seconds: Optional[torch.Tensor] = None,
        target_leads: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(source)
        fusion = self.fusion(encoded)
        latent = fusion["latent"]

        spatial = self.reconstruction_head(latent, target_leads=target_leads)

        output = {
            "y_pred": spatial["reconstruction"],
            "latent": latent,
            "z1": fusion["z1"],
            "z2": fusion["z2"],
            "query_embedding": spatial["query_embedding"],
            "target_leads": target_leads,
        }

        if self.enable_qt_head and self.qt_head is not None:
            qt_ms = self.qt_head(latent)
            output["qt_ms"] = qt_ms

            if rr_seconds is not None:
                rr_seconds = rr_seconds.clamp_min(1e-3)
                output["qtc_ms"] = qt_ms / torch.sqrt(rr_seconds)

        return output
