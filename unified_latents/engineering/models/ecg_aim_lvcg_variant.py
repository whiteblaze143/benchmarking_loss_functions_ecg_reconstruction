"""ECG-AIM with an auxiliary LVCG-style record embedding.

The upstream LVCG source is used unchanged.  Its package root is not imported
because the released repository omits ``lvcg.data`` modules required by that
root.  Instead, this module loads the self-contained upstream beat module from
its vendored source path and reuses ``BeatEncoder`` and ``GlobalRREmbedding``.

This is deliberately not called a VCG reconstruction from Lead I: a learned
1-to-3 adapter supplies a latent cardiac field because a physical 3-D field is
not identifiable from one scalar lead.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


def _load_upstream_beat_modules() -> ModuleType:
    root = Path(__file__).resolve().parents[3]
    source = root / "external" / "LVCG" / "lvcg" / "models" / "blocks" / "beat_modules.py"
    if not source.is_file():
        raise FileNotFoundError(f"Missing vendored LVCG beat modules: {source}")
    spec = importlib.util.spec_from_file_location("_vendored_lvcg_beat_modules", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load vendored LVCG beat modules: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_LVCG_BEAT_MODULES = _load_upstream_beat_modules()
BeatEncoder = _LVCG_BEAT_MODULES.BeatEncoder
BeatDecoder = _LVCG_BEAT_MODULES.BeatDecoder
GlobalRREmbedding = _LVCG_BEAT_MODULES.GlobalRREmbedding


class LVCGStyleLeadIEmbedding(nn.Module):
    """Build structure/dynamics/rhythm embeddings from Lead-I beats.

    Beat boundaries are required inputs.  This keeps peak detection outside the
    model and makes preprocessing failures explicit.
    """

    def __init__(
        self,
        *,
        beat_len: int = 128,
        state_dim: int = 256,
        rhythm_dim: int = 128,
        max_beats: int = 40,
        stem_channels: int = 48,
        stage_channels: tuple[int, ...] = (96, 192, 256, 256),
        stage_blocks: tuple[int, ...] = (3, 3, 3, 2),
        dropout: float = 0.1,
        field_mode: str = "learned",
    ) -> None:
        super().__init__()
        self.beat_len = int(beat_len)
        self.max_beats = int(max_beats)
        self.state_dim = int(state_dim)
        self.rhythm_dim = int(rhythm_dim)
        if field_mode not in {"learned", "triplicate"}:
            raise ValueError("field_mode must be 'learned' or 'triplicate'")
        self.field_mode = field_mode

        # A learned latent field, not a physically identifiable VCG.
        self.lead_i_to_field = nn.Conv1d(1, 3, kernel_size=1) if field_mode == "learned" else None
        self.beat_encoder = BeatEncoder(
            beat_len=beat_len,
            state_dim=state_dim,
            stem_channels=stem_channels,
            stage_channels=list(stage_channels),
            stage_blocks=list(stage_blocks),
            dropout=dropout,
        )
        self.beat_decoder = BeatDecoder(
            state_dim=state_dim,
            beat_len=beat_len,
            initial_channels=stage_channels[-1],
            hidden_channels=list(reversed(stage_channels)),
            dropout=dropout,
        )
        self.dynamic_encoder = nn.GRU(
            input_size=state_dim,
            hidden_size=state_dim,
            num_layers=1,
            batch_first=True,
        )
        self.rhythm_encoder = GlobalRREmbedding(
            out_dim=rhythm_dim,
            max_beats=max_beats,
        )
        self.norm_struct = nn.LayerNorm(state_dim)
        self.norm_dynamic = nn.LayerNorm(state_dim)
        self.norm_rhythm = nn.LayerNorm(rhythm_dim)
        self.out_features = 2 * state_dim + rhythm_dim

    def _segment(
        self,
        lead_i: torch.Tensor,
        beat_bounds: torch.Tensor,
        beat_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if lead_i.ndim != 3 or lead_i.shape[1] != 1:
            raise ValueError(f"lead_i must have shape [B,1,T], got {tuple(lead_i.shape)}")
        if beat_bounds.ndim != 3 or beat_bounds.shape[-1] != 2:
            raise ValueError("beat_bounds must have shape [B,N,2]")
        if beat_mask.shape != beat_bounds.shape[:2] or beat_mask.dtype != torch.bool:
            raise ValueError("beat_mask must be bool with shape [B,N]")
        if beat_bounds.shape[0] != lead_i.shape[0] or beat_bounds.shape[1] > self.max_beats:
            raise ValueError("beat bounds do not match batch or exceed max_beats")
        if not torch.all(beat_mask[:, 1:].logical_not() | beat_mask[:, :-1]):
            raise ValueError("beat_mask must be left-aligned")

        batch, beats = beat_mask.shape
        field = self.lead_i_to_field(lead_i) if self.lead_i_to_field is not None else lead_i.repeat(1, 3, 1)
        windows = field.new_zeros((batch, beats, 3, self.beat_len))
        lead_windows = lead_i.new_zeros((batch, beats, self.beat_len))
        for b in range(batch):
            for n in range(beats):
                if not bool(beat_mask[b, n]):
                    continue
                start, end = (int(v) for v in beat_bounds[b, n])
                if start < 0 or end > lead_i.shape[-1] or end <= start:
                    raise ValueError(f"invalid beat boundary ({start}, {end})")
                windows[b, n] = F.interpolate(
                    field[b : b + 1, :, start:end],
                    size=self.beat_len,
                    mode="linear",
                    align_corners=False,
                )[0]
                lead_windows[b, n] = F.interpolate(
                    lead_i[b : b + 1, :, start:end],
                    size=self.beat_len,
                    mode="linear",
                    align_corners=False,
                )[0, 0]
        return windows, lead_windows

    def forward(
        self,
        lead_i: torch.Tensor,
        beat_bounds: torch.Tensor,
        beat_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not torch.all(beat_mask.any(dim=1)):
            raise ValueError("every record must contain at least one valid beat")
        beat_windows, lead_windows = self._segment(lead_i, beat_bounds, beat_mask)
        # Do not let padded beats alter the upstream encoder's BatchNorm state.
        valid_states = self.beat_encoder(beat_windows[beat_mask].unsqueeze(0))[0]
        beat_states = lead_i.new_zeros(
            (*beat_mask.shape, self.state_dim), dtype=valid_states.dtype
        )
        beat_states[beat_mask] = valid_states

        weights = beat_mask.to(beat_states.dtype).unsqueeze(-1)
        emb_struct = (beat_states * weights).sum(1) / weights.sum(1).clamp_min(1)

        packed = nn.utils.rnn.pack_padded_sequence(
            beat_states,
            beat_mask.sum(1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_dynamic, hidden = self.dynamic_encoder(packed)
        dynamic_states, _ = nn.utils.rnn.pad_packed_sequence(
            packed_dynamic, batch_first=True, total_length=beat_mask.shape[1]
        )
        emb_dynamic = hidden[-1]

        decoded_field = self.beat_decoder(beat_states)
        reconstructed_lead = decoded_field.mean(dim=2)
        beat_reconstruction_loss = F.l1_loss(
            reconstructed_lead[beat_mask], lead_windows[beat_mask]
        )
        pair_mask = beat_mask[:, :-1] & beat_mask[:, 1:]
        temporal_loss = (
            F.smooth_l1_loss(dynamic_states[:, :-1][pair_mask], beat_states[:, 1:][pair_mask].detach())
            if bool(pair_mask.any())
            else beat_reconstruction_loss.new_zeros(())
        )
        embedding_ssl_loss = beat_reconstruction_loss + 0.1 * temporal_loss

        rr_intervals = (beat_bounds[..., 1] - beat_bounds[..., 0]).to(lead_i.dtype)
        emb_rhythm = self.rhythm_encoder(rr_intervals, beat_mask)

        emb_struct = self.norm_struct(emb_struct)
        emb_dynamic = self.norm_dynamic(emb_dynamic)
        emb_rhythm = self.norm_rhythm(emb_rhythm)
        ecg_emb = torch.cat((emb_struct, emb_dynamic, emb_rhythm), dim=-1)
        return {
            "ecg_emb": ecg_emb,
            "emb_struct": emb_struct,
            "emb_dynamic": emb_dynamic,
            "emb_rhythm": emb_rhythm,
            "beat_states": beat_states,
            "beat_mask": beat_mask,
            "rr_intervals": rr_intervals,
            "beat_reconstruction": reconstructed_lead,
            "beat_reconstruction_loss": beat_reconstruction_loss,
            "temporal_prediction_loss": temporal_loss,
            "embedding_ssl_loss": embedding_ssl_loss,
        }


class ECGAIMLVCGVariant(nn.Module):
    """Preserve ECG-AIM reconstruction and add an explicit record embedding."""

    def __init__(self, ecg_aim: nn.Module, **embedding_kwargs) -> None:
        super().__init__()
        self.ecg_aim = ecg_aim
        self.embedding = LVCGStyleLeadIEmbedding(**embedding_kwargs)
        self.out_features = self.embedding.out_features

    def forward(
        self,
        masked_ecg: torch.Tensor,
        beat_bounds: torch.Tensor,
        beat_mask: torch.Tensor,
        observed_lead_index: int = 0,
        **ecg_aim_kwargs,
    ) -> Dict[str, torch.Tensor]:
        if masked_ecg.ndim != 3 or masked_ecg.shape[1] != 12:
            raise ValueError(f"masked_ecg must have shape [B,12,T], got {tuple(masked_ecg.shape)}")
        if not 0 <= observed_lead_index < 12:
            raise ValueError("observed_lead_index must be in [0, 11]")
        reconstruction = self.ecg_aim(masked_ecg, **ecg_aim_kwargs)
        if not isinstance(reconstruction, dict) or "y_pred" not in reconstruction:
            raise TypeError("ecg_aim must return a dict containing y_pred")
        lead_i = masked_ecg[:, observed_lead_index : observed_lead_index + 1]
        embedding = self.embedding(lead_i, beat_bounds, beat_mask)
        overlap = reconstruction.keys() & embedding.keys()
        if overlap:
            raise KeyError(f"ECG-AIM/LVCG output key collision: {sorted(overlap)}")
        return {**reconstruction, **embedding}

    def ext_ecg_emb(
        self,
        masked_ecg: torch.Tensor,
        beat_bounds: torch.Tensor,
        beat_mask: torch.Tensor,
        *,
        observed_lead_index: int = 0,
        normalize: bool = False,
    ) -> torch.Tensor:
        if masked_ecg.ndim != 3 or masked_ecg.shape[1] != 12:
            raise ValueError(f"masked_ecg must have shape [B,12,T], got {tuple(masked_ecg.shape)}")
        lead_i = masked_ecg[:, observed_lead_index : observed_lead_index + 1]
        embedding = self.embedding(lead_i, beat_bounds, beat_mask)["ecg_emb"]
        return F.normalize(embedding, p=2, dim=-1) if normalize else embedding
