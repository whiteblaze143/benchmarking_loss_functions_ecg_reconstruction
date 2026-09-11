"""Encoder registry for LVCG linear probing."""

from .base import BaseEncoder

_RESERVED_KEYS = {"type", "enabled", "tag", "source"}


def create_encoder(model_config: dict, device: str = "cpu") -> BaseEncoder:
    model_type = model_config["type"]
    if model_type == "grail":
        from .grail_encoder import GRAILEncoder
        encoder_cls = GRAILEncoder
    elif model_type == "lvcg":
        from .lvcg_encoder import LVCGEncoder
        encoder_cls = LVCGEncoder
    else:
        raise ValueError(f"Unknown encoder type: {model_type}. Available: ['grail', 'lvcg']")

    kwargs = {k: v for k, v in model_config.items() if k not in _RESERVED_KEYS}
    encoder = encoder_cls(**kwargs)
    encoder = encoder.to(device)
    encoder.eval()
    return encoder


__all__ = ["BaseEncoder", "create_encoder"]
