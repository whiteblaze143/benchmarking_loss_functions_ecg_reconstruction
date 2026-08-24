"""Re-export ResNet1d and ResBlock1d from classifier for scripts that expect src.models.resnet."""
from learn_functions.classifier import ResNet1d, ResBlock1d

__all__ = ["ResNet1d", "ResBlock1d"]
