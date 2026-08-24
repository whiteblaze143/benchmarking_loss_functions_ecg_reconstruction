"""CPU-only compatibility for the authors' unconditional CUDA synchronization."""

import os
import sys
import types

# The shared environment contains an incomplete ``tensorflow`` namespace.
# TensorBoard supports a no-TensorFlow stub, selected by this marker module.
sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))

# The 2022 vendor code imports ``torch._six.inf``, removed in modern PyTorch.
torch_six = types.ModuleType("torch._six")
torch_six.inf = float("inf")
sys.modules.setdefault("torch._six", torch_six)

if os.environ.get("SEMISEG_CPU_ONLY") == "1":
    import torch

    torch.cuda.synchronize = lambda *args, **kwargs: None
