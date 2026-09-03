Yes. The correct next step is to build a **separate 3DRECON-QT architectural reference implementation**, rather than continuing to mutate the D-series ECG-AIM model and calling it 3DRECON-QT.

I checked the literature through Scite and the public GitHub implementation that 3DRECON-QT explicitly cites. The key discovery is that there is enough public Electrocardio-Panorama code to reuse the **exact angular encoding, lead coordinates, multiplicative query-conditioning pattern, and closely related convolutional decoder**, while the 3DRECON-QT-specific SE-ResNeXt, modified Z-stack/temporal-attention block, and QT head will have to be reconstructed from the paper/figure because I did not find a public 3DRECON-QT source repository under the paper/title/DOI searches.

The public Panorama implementation is particularly valuable. Its released `ThetaEncoder` exactly constructs the 12-D angular representation with the unusual `torch.stack(...).view(...)` ordering.  Its PTB loader also contains the exact 12 angular coordinate pairs we have been using.  Most importantly, the released Nef-Net projects the query angle and **multiplies it channelwise with the latent representation before decoding**, and its decoder is built from interpolation/upsampling and 1-D DoubleConv blocks.  This matches the core mechanism in the 3DRECON-QT figure much more closely than our present Transformer waveform decoder.

Chen et al. describe the mechanism directly: “A queried viewpoint is processed by Angular Encoding and an MLP, and is then multiplied by M” (Chen et al., 2021). Scite also confirms that the angular encoding converts the two angles into a 12-element representation before the MLP. That is the mechanism we should reproduce.

# PRD — 3DRECON-QT Architectural Reference and Spatial-Semantics Benchmark

**Project name:** `3DRECONQT_REFERENCE`

**Status:** New isolated experimental branch.

**Scientific purpose:** determine whether the physical-theta effect that was almost null in our D-series remains null when the spatial-conditioning mechanism is embedded in an architecture substantially closer to 3DRECON-QT.

**Do not replace, alter, rename, or overwrite D0–D8.**

The previous D-series should henceforth be called:

> **3D-theta mechanism ablation in ECG-AIM**

The new RQ-series should be called:

> **3DRECON-QT architectural reference**

It is an architectural reference, not an “exact reproduction,” unless the authors later provide the remaining implementation details.

---

## 1. Why we need this experiment

3DRECON-QT takes a 10-s, 500-Hz single-lead signal and processes it with an SE-ResNeXt. Its latent representation serves two heads: a theta-conditioned ECG reconstruction head and a Transformer-based QT head.  The paper explicitly cites Squeeze-and-Excitation, ResNeXt, Electrocardio-Panorama, and Transformer architectures as the components from which the system is constructed. 

Their reconstruction head is therefore not merely:

$$
H+\text{lead embedding}\rightarrow \hat X.
$$

It is much closer to

$$
x
\overset{\text{SE-ResNeXt}}{\longrightarrow}
W
\overset{\text{feature fusion}}{\longrightarrow}
Z
$$

followed by

$$
a_l=(\theta_l,\phi_l)
$$

$$
e_l=\operatorname{MLP}\left(\Theta(a_l)\right)
$$

$$
Z_l=e_l\odot Z
$$

$$
\hat X_l=D_{\mathrm{conv}}(Z_l).
$$

The QT branch sees the shared latent without target-angle conditioning:

$$
Z
\rightarrow
T_{\mathrm{QT}}
\rightarrow
\widehat{QT}.
$$

Ansari et al. state that optimization jointly uses waveform reconstruction and QT regression, and that the final model was selected by QTc validation performance. 

Our D3 model did not replicate that architecture.

---

# 2. Evidence/provenance registry

The agent must create a machine-readable `architecture_provenance.yaml` assigning every implementation choice one of:

`PAPER_CONFIRMED`, `FIGURE_CONFIRMED`, `PUBLIC_CODE_EXACT`, `PUBLIC_CODE_ADAPTED`, or `ASSUMPTION`.

| Component                                     | Status               | Source                      |
| --------------------------------------------- | -------------------- | --------------------------- |
| 10-s input                                    | PAPER_CONFIRMED      | Ansari et al.               |
| 500 Hz                                        | PAPER_CONFIRMED      | Ansari et al.               |
| 0.5–40 Hz, fifth-order zero-phase Butterworth | PAPER_CONFIRMED      | paper                       |
| development source \(V_3-V_2\)                | PAPER_CONFIRMED      | paper                       |
| SE-ResNeXt source encoder                     | PAPER_CONFIRMED      | paper                       |
| feature extraction → Z1/Z2 split              | FIGURE_CONFIRMED     | provided Fig. 2             |
| Z2 temporal attention                         | FIGURE_CONFIRMED     | provided Fig. 2             |
| Z2 LayerNorm                                  | FIGURE_CONFIRMED     | provided Fig. 2             |
| Z1/Z2 concatenation                           | FIGURE_CONFIRMED     | provided Fig. 2             |
| angular encoding                              | PAPER_CONFIRMED      | Chen citation               |
| exact 12-D Panorama encoder                   | PUBLIC_CODE_EXACT    | WhatAShot                   |
| standard lead coordinate table                | PUBLIC_CODE_EXACT    | WhatAShot                   |
| query MLP with Mish                           | FIGURE_CONFIRMED     | provided Fig. 2             |
| multiplicative query-latent fusion            | PAPER + PUBLIC CODE  | both                        |
| Upsample + DoubleConv ECG decoder             | FIGURE + PUBLIC CODE | both                        |
| L1 ECG reconstruction                         | FIGURE_CONFIRMED     | Fig. 2                      |
| Transformer QT branch                         | PAPER_CONFIRMED      | Ansari et al.               |
| L2 QT regression                              | FIGURE_CONFIRMED     | Fig. 2                      |
| SGD                                           | PAPER_CONFIRMED      | supplement excerpt          |
| LR \(10^{-3}\)                                | PAPER_CONFIRMED      | supplement excerpt          |
| weight decay \(10^{-5}\)                      | PAPER_CONFIRMED      | supplement excerpt          |
| cosine annealing/warm restarts                | PAPER_CONFIRMED      | supplement excerpt          |
| batch 128                                     | PAPER_CONFIRMED      | supplement excerpt          |
| max 100 epochs                                | PAPER_CONFIRMED      | supplement excerpt          |
| exact SE-ResNeXt depth/cardinality            | **UNKNOWN**          | do not invent silently      |
| exact latent dimensions                       | **UNKNOWN**          | configurable                |
| exact Z-stack channels                        | **UNKNOWN**          | configurable                |
| exact temporal-attention heads                | **UNKNOWN**          | configurable                |
| exact QT Transformer depth/heads              | **UNKNOWN**          | configurable                |
| exact reconstruction/QT loss weights          | **UNKNOWN**          | must not claim reproduction |
| exact normalization scope                     | **AMBIGUOUS**        | sensitivity required        |
| early-stopping patience                       | **UNKNOWN**          | configurable                |

The paper reports record-wide standardization after filtering, but the precise channel/statistic scope needs clarification. 

---

# 3. Public code policy

Clone/vendor the following source into:

```text
external/Electrocardio-Panorama/
```

from:

```text
WhatAShot/Electrocardio-Panorama
```

The repository is MIT-licensed.

Do **not** rewrite the angular encoder from memory. Import or vendor the exact released implementation.

The useful public files are:

```text
codes/network/utils/theta_encoder.py
codes/network/model_nefnet.py
codes/dataset/ptbv2.py
```

The source model is highly informative: it splits latent information into Z1/Z2 components, projects angular queries, multiplies the query embedding into the latent, then uses a convolutional upsampling decoder.

For the 1-D ResNeXt building blocks, `DeepPSP/torch_ecg` contains a native ECG-oriented 1-D ResNeXt implementation and is also MIT-licensed.

I would **not** take its whole model as the reference, however. Use it as an implementation sanity check because 3DRECON-QT additionally requires SE channel recalibration.

---

# 4. Repository structure

Create:

```text
unified_latents/
└── engineering/
    └── models/
        └── reconqt_reference/
            ├── __init__.py
            ├── theta.py
            ├── seresnext1d.py
            ├── feature_fusion.py
            ├── ecg_decoder.py
            ├── qt_head.py
            ├── model.py
            └── preprocessing.py

scripts/
├── train_3dreconqt_reference.py
├── run_3dreconqt_stage_r1.sh
├── evaluate_3dreconqt_reference.py
├── compute_3dreconqt_bootstrap.py
└── test_3dreconqt_reference.py

protocols/
└── 3DRECONQT_REFERENCE_PROTOCOL.md

refine-logs/
└── 3dreconqt_reference/
```

Keep this completely separate from `three_d_theta_reconstruction.py`.

---

# 5. Exact target-angle implementation

Use the released Panorama lead coordinates.

The public PTB implementation orders them as:

```text
I, II, V1, V2, V3, V4, V5, V6, III, aVR, aVL, aVF
```

with the following actual values.

Our model uses:

```text
I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
```

so reorder once, explicitly.

```python
# reconqt_reference/theta.py

import math
import torch
import torch.nn as nn


LEAD_NAMES = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)


# Panorama source ordering:
# I, II, V1, V2, V3, V4, V5, V6, III, aVR, aVL, aVF
PANORAMA_ANGLES = torch.tensor([
    [math.pi / 2,        math.pi / 2],   # I
    [5 * math.pi / 6,    math.pi / 2],   # II
    [math.pi / 2,       -math.pi / 18],  # V1
    [math.pi / 2,        math.pi / 18],  # V2
    [19 * math.pi / 36,  math.pi / 12],  # V3
    [11 * math.pi / 20,  math.pi / 6],   # V4
    [16 * math.pi / 30,  math.pi / 3],   # V5
    [16 * math.pi / 30,  math.pi / 2],   # V6
    [5 * math.pi / 6,   -math.pi / 2],   # III
    [math.pi / 3,       -math.pi / 2],   # aVR
    [math.pi / 3,        math.pi / 2],   # aVL
    [math.pi,            math.pi / 2],   # aVF
], dtype=torch.float32)


PANORAMA_TO_STANDARD = torch.tensor(
    [0, 1, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7],
    dtype=torch.long,
)


STANDARD_ANGLES = PANORAMA_ANGLES[PANORAMA_TO_STANDARD]


class ThetaEncoder(nn.Module):
    """
    Keep behavior identical to the released Electrocardio-Panorama
    ThetaEncoder.

    Input:
        theta [B, L, 2]

    Output:
        encoded [B, L, 12]
    """

    def __init__(self, encoder_len: int = 1):
        super().__init__()
        self.encoder_len = encoder_len
        self.omega = 1.0

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        b, lead_num = theta.shape[:2]

        sum_theta = theta[..., 0:1] + theta[..., 1:2]
        sub_theta = theta[..., 0:1] - theta[..., 1:2]

        base = torch.cat(
            [theta, sum_theta, sub_theta],
            dim=-1,
        )

        features = [
            base,
            torch.sin(base * self.omega),
            torch.cos(base * self.omega),
        ]

        # IMPORTANT:
        # This ordering matches the public implementation.
        return torch.stack(
            features,
            dim=-1,
        ).view(b, lead_num, -1)
```

The exact public implementation uses this `stack(...).view(...)` construction.

Do not replace it with a mathematically equivalent-looking concatenation without testing equivalence; the feature ordering changes.

---

# 6. Query encoder

3DRECON-QT's figure specifically shows:

```text
angular encoding
      ↓
    flatten
      ↓
    Linear
      ↓
     Mish
      ↓
    Linear
```

Implement that literally.

```python
class ThetaQueryEncoder(nn.Module):

    def __init__(
        self,
        latent_channels: int = 256,
        hidden: int = 128,
    ):
        super().__init__()

        self.theta = ThetaEncoder(1)

        self.mlp = nn.Sequential(
            nn.Linear(12, hidden),
            nn.Mish(),
            nn.Linear(hidden, latent_channels),
        )

    def forward(
        self,
        angles: torch.Tensor,
    ) -> torch.Tensor:

        # [B,L,2] -> [B,L,12]
        z = self.theta(angles)

        # [B,L,12] -> [B,L,C]
        return self.mlp(z)
```

Do not add learned lead embeddings to the true-theta cell.

---

# 7. Target-code controls

Implement all code modes behind one interface.

```python
class SpatialCodebook(nn.Module):

    def __init__(
        self,
        mode: str,
        seed: int = 20260903,
    ):
        super().__init__()

        self.mode = mode

        self.register_buffer(
            "angles",
            STANDARD_ANGLES.clone(),
        )

        # Fixed derangement.
        self.register_buffer(
            "permutation",
            (torch.arange(12) + 5) % 12,
        )

        if mode == "learned":
            self.learned = nn.Parameter(
                torch.randn(12, 12) * 0.02
            )
        else:
            self.learned = None

        if mode == "random_fixed":
            gen = torch.Generator()
            gen.manual_seed(seed)

            x = torch.randn(
                12,
                12,
                generator=gen,
            )

            x = (
                x - x.mean(0, keepdim=True)
            ) / (
                x.std(0, keepdim=True) + 1e-6
            )

            self.register_buffer(
                "random_fixed",
                x,
            )
        else:
            self.random_fixed = None

    def forward(
        self,
        batch_size: int,
        theta_encoder: ThetaEncoder,
    ):

        if self.mode == "theta":

            angles = self.angles[None].expand(
                batch_size, -1, -1
            )

            return theta_encoder(angles)

        if self.mode == "permuted_theta":

            angles = self.angles[
                self.permutation
            ]

            angles = angles[None].expand(
                batch_size, -1, -1
            )

            return theta_encoder(angles)

        if self.mode == "learned":

            return self.learned[None].expand(
                batch_size, -1, -1
            )

        if self.mode == "random_fixed":

            return self.random_fixed[None].expand(
                batch_size, -1, -1
            )

        if self.mode == "constant":

            return torch.zeros(
                batch_size,
                12,
                12,
                device=self.angles.device,
            )

        raise ValueError(self.mode)
```

The `constant` cell is the actual no-target-information control.

---

# 8. SE-ResNeXt-1D encoder

The paper cites Hu et al.'s Squeeze-and-Excitation network and Xie et al.'s ResNeXt as the basis for the encoder. 

The exact depth/cardinality is not reported in the material we have, so it must be configurable and recorded as an assumption.

I recommend a canonical 32×4d ResNeXt-50-like 1-D implementation for the first reference.

```python
# reconqt_reference/seresnext1d.py

import torch
import torch.nn as nn


class SE1D(nn.Module):

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
    ):
        super().__init__()

        hidden = max(
            channels // reduction,
            8,
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(
                channels,
                hidden,
                kernel_size=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                hidden,
                channels,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class SEResNeXtBottleneck1D(nn.Module):

    expansion = 4

    def __init__(
        self,
        in_channels,
        planes,
        stride=1,
        groups=32,
        width_per_group=4,
        se_reduction=16,
    ):
        super().__init__()

        width = int(
            planes * width_per_group / 64.0
        ) * groups

        out_channels = (
            planes * self.expansion
        )

        self.conv1 = nn.Conv1d(
            in_channels,
            width,
            1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm1d(width)

        self.conv2 = nn.Conv1d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )

        self.bn2 = nn.BatchNorm1d(width)

        self.conv3 = nn.Conv1d(
            width,
            out_channels,
            1,
            bias=False,
        )

        self.bn3 = nn.BatchNorm1d(
            out_channels
        )

        self.se = SE1D(
            out_channels,
            reduction=se_reduction,
        )

        self.relu = nn.ReLU(inplace=True)

        if (
            stride != 1
            or in_channels != out_channels
        ):
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(
                    out_channels
                ),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):

        identity = self.downsample(x)

        out = self.relu(
            self.bn1(self.conv1(x))
        )

        out = self.relu(
            self.bn2(self.conv2(out))
        )

        out = self.bn3(
            self.conv3(out)
        )

        out = self.se(out)

        out = out + identity

        return self.relu(out)


class SEResNeXt1D(nn.Module):

    def __init__(
        self,
        in_channels=1,
        layers=(3, 4, 6, 3),
        groups=32,
        width_per_group=4,
    ):
        super().__init__()

        self.inplanes = 64

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                64,
                kernel_size=15,
                stride=2,
                padding=7,
                bias=False,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

        self.layer1 = self._make_layer(
            64,
            layers[0],
            stride=1,
            groups=groups,
            width_per_group=width_per_group,
        )

        self.layer2 = self._make_layer(
            128,
            layers[1],
            stride=2,
            groups=groups,
            width_per_group=width_per_group,
        )

        self.layer3 = self._make_layer(
            256,
            layers[2],
            stride=2,
            groups=groups,
            width_per_group=width_per_group,
        )

        self.layer4 = self._make_layer(
            512,
            layers[3],
            stride=2,
            groups=groups,
            width_per_group=width_per_group,
        )

        self.out_channels = 2048

        self._init_weights()

    def _make_layer(
        self,
        planes,
        blocks,
        stride,
        groups,
        width_per_group,
    ):

        modules = [
            SEResNeXtBottleneck1D(
                self.inplanes,
                planes,
                stride=stride,
                groups=groups,
                width_per_group=width_per_group,
            )
        ]

        self.inplanes = (
            planes
            * SEResNeXtBottleneck1D.expansion
        )

        for _ in range(1, blocks):
            modules.append(
                SEResNeXtBottleneck1D(
                    self.inplanes,
                    planes,
                    groups=groups,
                    width_per_group=width_per_group,
                )
            )

        return nn.Sequential(*modules)

    def _init_weights(self):

        for module in self.modules():

            if isinstance(
                module,
                nn.Conv1d,
            ):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(
                module,
                nn.BatchNorm1d,
            ):
                nn.init.ones_(
                    module.weight
                )
                nn.init.zeros_(
                    module.bias
                )

    def forward(self, x):

        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
```

Do **not** describe `layers=(3,4,6,3)` as paper-confirmed. It is the canonical ResNeXt-50 assumption.

---

# 9. Feature Extraction and Fusion block

This is the part our earlier experiment omitted entirely.

Implement the figure literally:

```text
SE-ResNeXt latent
       │
    Conv1D
       │
     Chunk
   ┌───┴────┐
   │        │
 Z1 Conv  Z2 Conv
            │
      temporal attention
            │
         LayerNorm
   │        │
   └─ concatenate ─┘
            │
       shared latent
```

The public Panorama model likewise forms separate Z1 and Z2 representations and concatenates them into a shared latent representation.

```python
# feature_fusion.py

class ConvFeatureBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(
                out_channels
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TemporalAttention1D(nn.Module):

    def __init__(
        self,
        channels,
        heads=8,
        dropout=0.0,
    ):
        super().__init__()

        assert (
            channels % heads == 0
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(
            channels
        )

    def forward(self, x):

        # [B,C,T] -> [B,T,C]
        tokens = x.transpose(1, 2)

        attended, _ = self.attn(
            tokens,
            tokens,
            tokens,
            need_weights=False,
        )

        # Figure shows attention -> LN.
        tokens = self.norm(attended)

        return tokens.transpose(1, 2)


class FeatureExtractionFusion(nn.Module):

    def __init__(
        self,
        encoder_channels=2048,
        latent_channels=256,
        attention_heads=8,
    ):
        super().__init__()

        assert (
            latent_channels % 2 == 0
        )

        half = latent_channels // 2

        self.extract = ConvFeatureBlock(
            encoder_channels,
            latent_channels,
        )

        self.z1_conv = ConvFeatureBlock(
            half,
            half,
        )

        self.z2_conv = ConvFeatureBlock(
            half,
            half,
        )

        self.z2_attention = (
            TemporalAttention1D(
                half,
                heads=attention_heads,
            )
        )

    def forward(self, x):

        w = self.extract(x)

        z1, z2 = torch.chunk(
            w,
            2,
            dim=1,
        )

        z1 = self.z1_conv(z1)

        z2 = self.z2_conv(z2)
        z2 = self.z2_attention(z2)

        latent = torch.cat(
            [z1, z2],
            dim=1,
        )

        return {
            "latent": latent,
            "z1": z1,
            "z2": z2,
        }
```

No graph layers.

No ECG-AIM target-query attention.

No wavelet branch.

No foundation model.

No metadata yet.

---

# 10. Reconstruction decoder

The public Panorama decoder uses repeated upsampling plus DoubleConv and then a final 1-D convolution.

That is exactly the family shown in the 3DRECON-QT figure.

Rewrite it for 5000 samples rather than copying it verbatim.

```python
# ecg_decoder.py

import torch.nn.functional as F


class DoubleConv1D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(
                out_channels
            ),
            nn.ReLU(inplace=True),

            nn.Conv1d(
                out_channels,
                out_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(
                out_channels
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UpsampleDoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.conv = DoubleConv1D(
            in_channels,
            out_channels,
        )

    def forward(self, x):

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="linear",
            align_corners=False,
        )

        return self.conv(x)


class ECGDecoder(nn.Module):

    def __init__(
        self,
        latent_channels=256,
        target_length=5000,
    ):
        super().__init__()

        self.target_length = (
            target_length
        )

        self.blocks = nn.Sequential(
            UpsampleDoubleConv(
                latent_channels,
                128,
            ),
            UpsampleDoubleConv(
                128,
                64,
            ),
            UpsampleDoubleConv(
                64,
                32,
            ),
            UpsampleDoubleConv(
                32,
                16,
            ),
            UpsampleDoubleConv(
                16,
                8,
            ),
        )

        self.output = nn.Conv1d(
            8,
            1,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):

        x = self.blocks(x)

        x = self.output(x)

        # Encoder stride rounding can give
        # 4992/5008/etc. Enforce exact ECG length.
        if x.shape[-1] != self.target_length:

            x = F.interpolate(
                x,
                size=self.target_length,
                mode="linear",
                align_corners=False,
            )

        return x
```

### Critical detail

Do **not** use Panorama's final sigmoid.

Panorama min-max normalized its signals. 3DRECON-QT describes standardized signals with positive and negative values. Therefore our reference decoder must remain unbounded/sign-preserving.

---

# 11. Spatial reconstruction head

This should be query-by-viewpoint generation, not 12 independent decoders.

$$
\hat X_l
=
D\left(
Z\odot
g(\Theta_l)
\right).
$$

```python
class SpatialReconstructionHead(nn.Module):

    def __init__(
        self,
        latent_channels=256,
        theta_hidden=128,
        target_length=5000,
        code_mode="theta",
    ):
        super().__init__()

        self.code_mode = code_mode

        self.theta_encoder = (
            ThetaEncoder(1)
        )

        self.codebook = SpatialCodebook(
            mode=code_mode
        )

        self.query_mlp = nn.Sequential(
            nn.Linear(
                12,
                theta_hidden,
            ),
            nn.Mish(),
            nn.Linear(
                theta_hidden,
                latent_channels,
            ),
        )

        self.decoder = ECGDecoder(
            latent_channels=latent_channels,
            target_length=target_length,
        )

    def forward(self, latent):

        B, C, T = latent.shape

        codes = self.codebook(
            batch_size=B,
            theta_encoder=self.theta_encoder,
        )

        query = self.query_mlp(
            codes
        )  # [B,12,C]

        conditioned = (
            latent[:, None, :, :]
            *
            query[:, :, :, None]
        )

        # [B,12,C,T]
        conditioned = conditioned.reshape(
            B * 12,
            C,
            T,
        )

        reconstructed = self.decoder(
            conditioned
        )

        reconstructed = reconstructed.reshape(
            B,
            12,
            -1,
        )

        return {
            "reconstruction": reconstructed,
            "query_embedding": query,
        }
```

This is substantially closer to the actual spatial synthesis mechanism than our current Transformer decoder.

---

# 12. Full reference reconstruction model

```python
# model.py

class ThreeDReconQTReference(nn.Module):

    def __init__(
        self,
        input_channels=1,
        latent_channels=256,
        theta_hidden=128,
        attention_heads=8,
        code_mode="theta",
        target_length=5000,
        enable_qt_head=False,
    ):
        super().__init__()

        self.encoder = SEResNeXt1D(
            in_channels=input_channels
        )

        self.fusion = (
            FeatureExtractionFusion(
                encoder_channels=
                    self.encoder.out_channels,
                latent_channels=
                    latent_channels,
                attention_heads=
                    attention_heads,
            )
        )

        self.reconstruction_head = (
            SpatialReconstructionHead(
                latent_channels=
                    latent_channels,
                theta_hidden=
                    theta_hidden,
                target_length=
                    target_length,
                code_mode=
                    code_mode,
            )
        )

        self.enable_qt_head = (
            enable_qt_head
        )

        if enable_qt_head:
            self.qt_head = QTTemporalHead(
                channels=latent_channels,
            )

    def forward(
        self,
        source,
        rr_seconds=None,
    ):

        encoded = self.encoder(
            source
        )

        fusion = self.fusion(
            encoded
        )

        latent = fusion["latent"]

        spatial = (
            self.reconstruction_head(
                latent
            )
        )

        output = {
            "y_pred":
                spatial["reconstruction"],

            "latent":
                latent,

            "z1":
                fusion["z1"],

            "z2":
                fusion["z2"],

            "query_embedding":
                spatial["query_embedding"],
        }

        if self.enable_qt_head:

            qt_ms = self.qt_head(
                latent
            )

            output["qt_ms"] = qt_ms

            if rr_seconds is not None:

                rr_seconds = (
                    rr_seconds.clamp_min(
                        1e-3
                    )
                )

                output["qtc_ms"] = (
                    qt_ms
                    /
                    torch.sqrt(
                        rr_seconds
                    )
                )

        return output
```

---

# 13. QT branch

Do not silently call this exact yet.

The paper tells us the latent representation enters a Transformer decoder, where self-attention is intended to capture relationships across the cardiac cycle, followed by an MLP scalar regressor. 

The exact topology is unavailable.

Implement the following as an explicitly tagged approximation:

```python
class SinusoidalPosition1D(nn.Module):

    def __init__(
        self,
        channels,
        max_len=2048,
    ):
        super().__init__()

        position = torch.arange(
            max_len
        ).float()[:, None]

        div = torch.exp(
            torch.arange(
                0,
                channels,
                2,
            ).float()
            *
            (
                -math.log(10000.0)
                / channels
            )
        )

        pe = torch.zeros(
            max_len,
            channels,
        )

        pe[:, 0::2] = torch.sin(
            position * div
        )

        pe[:, 1::2] = torch.cos(
            position * div
        )

        self.register_buffer(
            "pe",
            pe[None],
        )

    def forward(self, x):

        return (
            x
            + self.pe[:, :x.shape[1]]
        )


class QTTemporalHead(nn.Module):

    def __init__(
        self,
        channels=256,
        depth=4,
        heads=8,
        dropout=0.1,
    ):
        super().__init__()

        self.position = (
            SinusoidalPosition1D(
                channels
            )
        )

        layer = (
            nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=heads,
                dim_feedforward=
                    4 * channels,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                layer,
                num_layers=depth,
                norm=nn.LayerNorm(
                    channels
                ),
            )
        )

        self.regressor = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(
                channels,
                channels // 2,
            ),
            nn.Mish(),
            nn.Linear(
                channels // 2,
                1,
            ),
        )

    def forward(self, latent):

        # [B,C,T] -> [B,T,C]
        tokens = latent.transpose(
            1,
            2,
        )

        tokens = self.position(
            tokens
        )

        tokens = self.transformer(
            tokens
        )

        pooled = tokens.mean(
            dim=1
        )

        return (
            self.regressor(
                pooled
            ).squeeze(-1)
        )
```

Call it:

```text
QT_TEMPORAL_TRANSFORMER_APPROX
```

until the author confirms exact decoder topology.

---

# 14. Input construction

3DRECON-QT's development signal was not Lead I. It was

$$
\boxed{x_{\mathrm{ICM}}=V_3-V_2}
$$

for the large development/external cohorts. 

Therefore we need two orthogonal source conditions.

### Deployment condition

```text
SOURCE_LEAD_I
```

$$
x=x_I.
$$

This is primary for your CardioSTAT/Fitbit study.

### Paper-style condition

```text
SOURCE_V3_MINUS_V2
```

$$
x=V_3-V_2.
$$

This is **not** deployable with a standard Lead-I wearable. It exists to answer:

> Does the 3DRECON spatial mechanism behave differently when presented with the same kind of narrow precordial vector for which it was designed?

That is an important experiment.

---

# 15. Filtering

Implement exactly:

```python
from scipy.signal import butter, sosfiltfilt


def bandpass_3drecon(
    signal,
    fs=500.0,
    low=0.5,
    high=40.0,
    order=5,
):

    sos = butter(
        order,
        [low, high],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    return sosfiltfilt(
        sos,
        signal,
        axis=-1,
    )
```

The paper reports a zero-phase fifth-order 0.5–40-Hz Butterworth preprocessing step. 

Precompute/cache this offline. Do not run SciPy filtering repeatedly inside every GPU batch.

---

# 16. Normalization must have three explicit modes

This is unresolved enough that the code should make it impossible to conflate them.

### `strict_deployable`

Source:

$$
\mu_s,\sigma_s
$$

computed only from the measured signal.

Record those values so output amplitude can be restored.

Target normalization must use **training-set constants** or another inference-available transform.

Never hidden target leads.

### `per_channel_record`

Every individual lead receives its own temporal mean/SD.

Potentially compatible with the wording “all leads entered the network with zero mean and unit variance,” but amplitude relations are lost.

### `shared_12lead_record`

One shared \(\mu,\sigma\) computed over the entire simultaneous 12-lead record.

This may approximate “record-wide” literally, but is:

$$
\boxed{\text{NONDEPLOYABLE FOR SINGLE-LEAD INFERENCE}}
$$

if those hidden leads contribute to the source normalization statistics.

Do not select the final model from this mode.

The agent must put the normalization mode directly into every run name and checkpoint provenance.

---

# 17. Reconstruction loss

For the faithful reference:

$$
\boxed{
\mathcal L_{\mathrm{ECG}}
=
\frac{1}{12T}
\sum_{l=1}^{12}
\sum_t
|\hat X_l(t)-X_l(t)|
}
$$

because 3DRECON-QT's original source is an ICM-like signal that is not itself one of the 12 standard target leads.

```python
def full12_l1(
    prediction,
    target,
):
    return torch.nn.functional.l1_loss(
        prediction,
        target,
    )
```

For your current Lead-I deployment evaluation, separately report missing-lead metrics excluding Lead I.

Do **not** copy Lead I into the model output for the faithful reference.

That would make reconstruction architecture different from the paper.

---

# 18. Multitask objective

When a valid QT label exists:

$$
\mathcal L
=
\lambda_{\mathrm{ECG}}
\mathcal L_1
+
\lambda_{\mathrm{QT}}
\mathcal L_2
$$

with

$$
\mathcal L_{\mathrm{QT}}
=
(\widehat{QT}-QT)^2.
$$

The source states that a weighted combination of reconstruction and QT regression was used. 

Do **not** guess the exact weights and then call the result a reproduction.

Implement:

```python
class ReconQTLoss(nn.Module):

    def __init__(
        self,
        lambda_ecg,
        lambda_qt=None,
    ):
        super().__init__()

        self.lambda_ecg = (
            lambda_ecg
        )

        self.lambda_qt = (
            lambda_qt
        )

    def forward(
        self,
        output,
        ecg_target,
        qt_target=None,
    ):

        ecg = F.l1_loss(
            output["y_pred"],
            ecg_target,
        )

        result = {
            "ecg_l1": ecg,
        }

        if qt_target is None:

            result["total"] = (
                self.lambda_ecg
                * ecg
            )

            return result

        if self.lambda_qt is None:

            raise RuntimeError(
                "QT multitask enabled but "
                "lambda_qt was not explicitly "
                "specified."
            )

        qt = F.mse_loss(
            output["qt_ms"],
            qt_target,
        )

        result["qt_l2"] = qt

        result["total"] = (
            self.lambda_ecg * ecg
            +
            self.lambda_qt * qt
        )

        return result
```

---

# 19. QT labels: do not manufacture a “faithful” label

Ansari et al.'s development QT/RR measurements came from clinical ECG systems and were cardiologist-adjudicated. 

Therefore:

$$
\boxed{\text{PTB-XL delineator-derived QT} \neq \text{their clinician QT target}.}
$$

If we use our P/QRS/T delineations to derive:

$$
QT =
T_{\mathrm{offset}}
-
QRS_{\mathrm{onset}},
$$

call the experiment:

```text
RQ_MTL_SURROGATE_QT
```

not:

```text
3DRECONQT_REPLICATION
```

It can still test whether a physiology-targeted interval auxiliary task regularizes reconstruction.

That is potentially useful later.

---

# 20. RQ experiment matrix

Do not run every possible combination at once.

### Stage R1 — reconstruction architecture

| Cell    | Source | Spatial code          | Fusion   | Decoder | Loss |
| ------- | ------ | --------------------- | -------- | ------- | ---- |
| **RQ0** | Lead I | constant/no target ID | multiply | conv    | L1   |
| **RQ1** | Lead I | true theta            | multiply | conv    | L1   |
| **RQ2** | Lead I | permuted theta        | multiply | conv    | L1   |
| **RQ3** | Lead I | learned 12-D          | multiply | conv    | L1   |
| **RQ4** | Lead I | fixed random 12-D     | multiply | conv    | L1   |

This is the architectural analogue of D3/D5/D4/D8.

The central question is:

$$
\boxed{RQ1-RQ2}.
$$

If the gap remains approximately zero, the theta-permutation null survives a 3DRECON-like architecture.

---

# 21. Stage R2 — source-vector interaction

Run only:

$$
RQ1,\ RQ2,\ RQ3
$$

using:

$$
x=V_3-V_2.
$$

Now evaluate

$$
\Delta_{\mathrm{geometry}}
=
RQ1-RQ2
$$

under both source vectors.

This gives a particularly interesting interaction:

$$
\Delta_{\theta,I}
$$

versus

$$
\Delta_{\theta,V3-V2}.
$$

If:

$$
\Delta_{\theta,I}\approx0
$$

but

$$
\Delta_{\theta,V3-V2}>0,
$$

then physical target geometry may matter only when the source vector resembles the nonstandard chest-wall orientation used by 3DRECON-QT.

That would explain our current result without invalidating either model.

---

# 22. Stage R3 — multitask only when a defensible QT target exists

Run:

| Cell    | Theta    | ECG | QT |
| ------- | -------- | --- | -- |
| **RQ5** | true     | L1  | L2 |
| **RQ6** | permuted | L1  | L2 |
| **RQ7** | no theta | L1  | L2 |

Now evaluate **both** reconstruction and QT.

This is essential because the published theta ablation is a QT/QTc ablation. The paper reports dramatic QTc degradation without spatial encoding, while the reported overall reconstruction performance is approximately \(r=0.70\). 

The published result therefore does not establish that theta improves waveform reconstruction by the same magnitude.

---

# 23. Training regime

Use the paper regime for the RQ-series instead of forcing the ECG-AIM 10→15-epoch regime onto it.

Confirmed settings are:

$$
\text{optimizer}=\mathrm{SGD}
$$

$$
LR_0=10^{-3}
$$

$$
weight\ decay=10^{-5}
$$

$$
B=128
$$

with cosine annealing/warm restarts, up to 100 epochs and early stopping. 

Implementation:

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-3,
    momentum=0.9,      # ASSUMPTION
    weight_decay=1e-5,
)

scheduler = (
    torch.optim.lr_scheduler
    .CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,       # ASSUMPTION
        T_mult=1,     # ASSUMPTION
        eta_min=1e-6,
    )
)
```

Momentum, `T_0`, `T_mult`, and early-stopping patience must be marked as assumptions unless your friend gives us the exact values.

If batch 128 does not fit, do not reduce the effective batch. Use accumulation:

```python
effective_batch = 128
physical_batch = 32
accum_steps = 4
```

---

# 24. Model-selection rule

For reconstruction-only RQ0–RQ4:

$$
\boxed{\text{select checkpoint using PTB-XL validation reconstruction only}}
$$

Primary:

$$
\bar r_{\mathrm{missing}}.
$$

Do not use test.

For RQ5–RQ7 multitask, the paper chose its checkpoint according to QTc Pearson on validation. 

If using surrogate QT:

```text
selection_policy = SURROGATE_QT_APPROX
```

must appear in the checkpoint metadata.

---

# 25. Evaluation

For every cell report both paper-style and our-task-style metrics.

### Paper-style reconstruction metric

$$
\bar r_{12}
=
\frac1{12}
\sum_{l=1}^{12}
r_l.
$$

Their reported reconstruction performance was approximately \(0.70\) mean Pearson over all 12 leads. 

Do not interpret our ability or inability to reproduce \(0.70\) as exact replication because our cohort and source conditions differ.

### Our strict metrics

Report:

$$
\bar r_{\mathrm{missing}}
$$

$$
r_{05}
$$

$$
\bar r_{V1:V6}
$$

$$
L1
$$

$$
MSE
$$

and all 12 per-lead correlations.

Continue the precordial transition metric:

$$
\Delta V_k=V_{k+1}-V_k.
$$

Also report P/QRS/T delineation metrics **only as frozen downstream evaluation**, not as a training objective.

---

# 26. Physics audit remains reporting-only

For every reconstruction compute:

$$
E_{\mathrm{III}}
=
|\hat{II}-\hat I-\hat{III}|
$$

and corresponding Goldberger residuals.

Do not algebraically derive the limb leads inside RQ0–RQ7.

3DRECON-QT is described as a 12-lead query reconstruction architecture. Hard limb algebra would create a different model.

Keep the algebraic-basis model as a later independent extension.

---

# 27. Paired statistical evaluation

For central comparisons:

$$
RQ1-RQ2
$$

$$
RQ1-RQ3
$$

$$
RQ1-RQ4,
$$

perform patient-level paired bootstrap:

```python
for patient i:
    delta_i = metric_i(model_A) - metric_i(model_B)
```

10,000 bootstrap resamples.

Report:

$$
E[\Delta]
$$

and

$$
CI_{95\%}.
$$

After seed-42 screen, replicate:

$$
RQ1,RQ2,RQ3
$$

with seeds:

$$
42,43,44.
$$

This separates patient-sampling uncertainty from optimization uncertainty.

---

# 28. Required smoke tests

Before any full training, the agent must pass all of these.

1. Input `[B,1,5000]` produces a shared latent `[B,C,T_z]`.

2. Exact public ThetaEncoder gives `[B,12,12]`.

3. Angle table names and coordinates match the Panorama source.

4. `RQ1` and `RQ2` differ only in angle-to-target assignment.

5. `RQ1` and `RQ3` have identical source encoder/fusion/decoder architecture.

6. All 12 targets go through **the same** ECGDecoder instance.

7. No learned lead IDs are accessed by RQ1/RQ2.

8. Query fusion is true elementwise multiplication:

```python
conditioned = latent * query[..., None]
```

9. Reconstruction output has shape:

```python
[B, 12, 5000]
```

10. No sigmoid/tanh is applied to standardized ECG outputs.

11. `V3-V2` mode verifies numerically:

```python
source == target[:, V3] - target[:, V2]
```

before source normalization.

12. `strict_deployable` normalization reads no hidden lead.

13. Source/QT branches receive the same shared latent.

14. `model.parameters()` confirms one shared ECG decoder rather than 12 duplicated heads.

15. RQ0 constant query produces identical target condition vectors.

16. RQ1 produces 12 distinct query condition vectors.

17. Permutation in RQ2 is persisted in checkpoint metadata.

18. Filtering is zero-phase and exactly 0.5–40 Hz.

---

# 29. Provenance saved with every run

Every checkpoint must contain:

```python
checkpoint["provenance"] = {
    "model_family":
        "3dreconqt_reference",

    "fidelity_status":
        "architecture_reference",

    "source_mode":
        source_mode,

    "code_mode":
        code_mode,

    "angle_table":
        STANDARD_ANGLES.cpu().tolist(),

    "lead_names":
        list(LEAD_NAMES),

    "theta_permutation":
        permutation.cpu().tolist(),

    "normalization":
        normalization_mode,

    "filter":
        {
            "low_hz": 0.5,
            "high_hz": 40.0,
            "order": 5,
            "zero_phase": True,
        },

    "encoder":
        {
            "family":
                "SE-ResNeXt-1D",
            "layers":
                list(layers),
            "groups":
                groups,
            "width_per_group":
                width_per_group,
            "paper_confirmed_depth":
                False,
        },

    "latent_channels":
        latent_channels,

    "theta_hidden":
        theta_hidden,

    "optimizer":
        "SGD",

    "lr":
        1e-3,

    "weight_decay":
        1e-5,

    "effective_batch_size":
        128,

    "max_epochs":
        100,

    "qt_head":
        enable_qt_head,

    "qt_label_source":
        qt_label_source,

    "git_commit":
        git_commit,

    "split_hashes":
        split_hashes,
}
```

---

# 30. What counts as a meaningful result

Suppose we obtain:

$$
RQ1=.720,\qquad
RQ2=.719.
$$

Then our previous conclusion strengthens considerably:

> Correct anatomical angle assignment did not provide a practically meaningful advantage over scrambled assignment even within a 3DRECON-style convolutional reconstruction architecture.

If instead:

$$
RQ1=.730,\qquad
RQ2=.710,
$$

then our earlier D-series null was architecture-specific.

If:

$$
RQ1_I\approx RQ2_I
$$

but

$$
RQ1_{V3-V2}>RQ2_{V3-V2},
$$

then the spatial prior interacts with source-vector orientation.

If reconstruction remains insensitive to scrambling but RQ5 QT prediction becomes strongly theta-sensitive:

$$
RQ5_{QT}\gg RQ6_{QT},
$$

that would be particularly interesting because it would reproduce the **qualitative pattern implied by the original paper**:

$$
\boxed{
\text{geometry may matter more for physiologic task representation than pointwise waveform similarity}.
}
$$

That is a much more sophisticated story than “theta helps reconstruction.”

---

# 31. Things the agent must **not** do

Do not silently add:

* wavelets;
* SSL;
* ECG foundation-model features;
* patient demographics;
* P/QRS/T training masks;
* target-query Transformer attention;
* graph lead interactions;
* per-lead decoders;
* Einthoven hard constraints;
* algebraic limb derivation;
* metadata conditioning;
* cross-attention;
* perceptual losses;
* ECG-FM;
* HuBERT;
* CSFM.

Those are subsequent experiments.

The purpose here is to recreate the architecture responsible for the scientific comparison.

---

# 32. Blocking questions for your 3DRECON-QT friend

These now become very specific. Ask him for:

1. Exact SE-ResNeXt depth, cardinality, base width, stem kernel and stride.
2. Exact dimensions after SE-ResNeXt.
3. What precisely happens in “Feature Extraction and Fusion.”
4. Channel dimensions of Z1 and Z2.
5. How many heads in Z2 temporal attention.
6. Whether attention has residual connections.
7. Exact ThetaEncoder source code.
8. Whether it is literally Panorama's public 12-D encoder.
9. Theta MLP dimensions.
10. Whether `Mish` is used exactly as shown.
11. Exact ECGDecoder channel schedule and number of upsampling stages.
12. Whether the ECG output has any activation.
13. QT Transformer depth, heads, hidden dimension and pooling.
14. Exact ECG/QT loss weights.
15. SGD momentum.
16. `CosineAnnealingWarmRestarts` values.
17. Early stopping patience.
18. What “record-wide mean and SD” means mathematically.
19. Whether target 12-lead information enters normalization of the single input.
20. How the no-theta ablation still distinguishes target leads.
21. Whether the theta ablation affected **reconstruction r**, not just QT.
22. Whether all 12 standard leads were directly generated or dependent limb leads were derived.

Those answers would eliminate almost every remaining assumption in this PRD.

---

# Agent handoff

Copy the following to Antigravity:

```text
Implement a new isolated architecture family named
3DRECONQT_REFERENCE.

DO NOT MODIFY:
- existing D0-D8 3D-theta models,
- existing ECG-AIM architecture,
- convergence experiments,
- foundation-model experiments,
- RDB evaluation logic.

SCIENTIFIC PURPOSE

We previously tested Panorama-style physical theta conditioning inside
our ECG-AIM/Transformer architecture and found:

D3 correct theta r = 0.7134
D5 permuted theta r = 0.7118

This does NOT constitute a faithful reproduction of 3DRECON-QT.

We now need an architectural reference that follows the published
3DRECON-QT Figure 2 substantially more closely.

PUBLIC CODE SOURCES

Clone/read:

WhatAShot/Electrocardio-Panorama

Use the exact public code as the source of truth for:
- codes/network/utils/theta_encoder.py
- standard lead angle coordinates from codes/dataset/ptbv2.py

Use model_nefnet.py as architectural evidence for:
- query-angle MLP
- multiplicative query-latent conditioning
- shared target decoder
- Upsample + DoubleConv decoder family
- Z1/Z2 latent construction

Do NOT import Panorama source-view angular conditioning into the
3DRECON-QT primary reference unless separately requested.
The 3DRECON-QT figure shows target/query spatial conditioning.

ARCHITECTURE

Input:
[B,1,5000], 500 Hz.

Path:

single ECG
  -> SE-ResNeXt-1D
  -> 1D feature extraction conv
  -> channel chunk
      -> Z1 Conv
      -> Z2 Conv -> temporal self-attention -> LayerNorm
  -> concatenate Z1/Z2
  -> shared latent Z [B,C,Tz]

For each target lead l:

(theta_l, phi_l)
  -> exact Panorama ThetaEncoder
  -> 12-D angular code
  -> Linear
  -> Mish
  -> Linear
  -> C-dimensional target embedding

Then:

Z_l = Z * target_embedding_l[...,None]

All target leads MUST use one shared ECG decoder:

Upsample
-> DoubleConv1D
-> Upsample
-> DoubleConv1D
-> ...
-> Conv1D(1)

Return:
[B,12,5000].

DO NOT:
- use target-query Transformer waveform decoder,
- use learned lead ID with true-theta cells,
- create 12 lead-specific decoders,
- add graph modules,
- add metadata,
- add wavelets,
- add SSL,
- add FM features,
- add segmentation training,
- derive limb leads algebraically.

SE-RESNEXT

Implement canonical 1-D SE-ResNeXt with configurable:
- stage depths
- groups
- width per group
- SE ratio

Initial assumption:
layers=(3,4,6,3)
groups=32
width_per_group=4
SE reduction=16

Mark these as ASSUMPTIONS because the exact 3DRECON-QT depth is
not confirmed.

FEATURE FUSION

Initial:
encoder C -> 256 channels
split -> 128 Z1 + 128 Z2
Z1 Conv
Z2 Conv + 8-head self-attention + LayerNorm
concat -> 256-channel shared latent

Mark dimensions/heads as assumptions.

THETA

Use exact public ThetaEncoder behavior, including:

torch.stack(out_all, dim=-1).view(...)

Do not rewrite its feature ordering.

Use the public lead-angle table and explicitly reorder to:

I, II, III, aVR, aVL, aVF,
V1, V2, V3, V4, V5, V6.

Theta query MLP:
12 -> 128 -> Mish -> 256.

CODE MODES

Implement:
- constant
- theta
- permuted_theta
- learned
- random_fixed

All code modes must use the same downstream projection and decoder
where conceptually possible.

Primary permutation:
(arange(12) + 5) % 12

Persist it in provenance.

DECODER

Use one shared decoder.

No sigmoid/tanh output.

Input data are signed ECG signals.

PREPROCESSING

Implement offline/cached:
0.5-40 Hz fifth-order zero-phase Butterworth.

Implement explicit normalization modes:
1 strict_deployable
2 per_channel_record
3 shared_12lead_record

shared_12lead_record must be marked NONDEPLOYABLE whenever hidden
target leads contribute to source normalization.

SOURCE MODES

Implement:
lead_I:
    source = target[:,0:1,:]

v3_minus_v2:
    source = target[:,V3:V3+1,:] - target[:,V2:V2+1,:]

Verify V3-V2 numerically before normalization.

LOSS

Reference reconstruction loss:
full 12-lead L1.

Do not overwrite Lead I in the output.

For our deployment reporting, additionally calculate missing-lead
metrics excluding observed Lead I.

RUN CELLS

Stage R1, Lead I, seed 42:

RQ0_constant_l1
RQ1_theta_l1
RQ2_permuted_theta_l1
RQ3_learned12_l1
RQ4_random12_l1

Central contrast:
RQ1 - RQ2.

Stage R2:
run RQ1/RQ2/RQ3 with source=v3_minus_v2.

Do not run multitask QT until QT target provenance is resolved.

TRAINING

Use paper-level training regime:

SGD
initial LR=1e-3
weight_decay=1e-5
effective batch=128
CosineAnnealingWarmRestarts
max epochs=100
early stopping

Unknown:
momentum
T0
Tmult
patience

Do not silently claim those values are paper-exact.
Log them as assumptions.

If batch 128 does not fit, use gradient accumulation to preserve
effective batch 128.

VALIDATION

Never inspect PTB-XL fold 10 during model selection.
Never use RDB to select these cells.

PTB-XL:
train folds 1-8
validation fold 9
test fold 10 untouched.

REPORT

For every cell:
- all-12 mean Pearson
- missing-11 mean Pearson for Lead-I source
- patient-level r_p05
- V1-V6 mean Pearson
- precordial DeltaV progression correlation
- per-lead Pearson
- L1
- MSE
- Einthoven/Goldberger consistency residuals
- parameter count
- GPU memory
- convergence trajectory

For RQ1-RQ2 and RQ1-RQ3:
paired patient-level 10,000-sample bootstrap.

After seed-42 screen:
replicate RQ1/RQ2/RQ3 with seeds 42,43,44.

FILES TO CREATE

unified_latents/engineering/models/reconqt_reference/
    theta.py
    seresnext1d.py
    feature_fusion.py
    ecg_decoder.py
    qt_head.py
    model.py
    preprocessing.py

scripts/
    train_3dreconqt_reference.py
    evaluate_3dreconqt_reference.py
    compute_3dreconqt_bootstrap.py
    test_3dreconqt_reference.py
    run_3dreconqt_stage_r1.sh

protocols/
    3DRECONQT_REFERENCE_PROTOCOL.md

Every run must save architecture_provenance with each implementation
choice tagged as:
PAPER_CONFIRMED
FIGURE_CONFIRMED
PUBLIC_CODE_EXACT
PUBLIC_CODE_ADAPTED
ASSUMPTION

BEFORE TRAINING

Show me:
1. exact public ThetaEncoder import/source hash
2. exact lead angle table
3. tensor shape trace through all modules
4. RQ1 vs RQ2 module/state-dict parity
5. proof all 12 target leads use one shared decoder
6. proof theta cells never access learned lead IDs
7. proof strict normalization reads no hidden lead
8. parameter count
9. estimated memory at batch 128
10. provenance table of all paper-confirmed vs assumed components

Do not start full R1 training until the smoke tests pass.
```

## Scientific expectation

I would not assume this will rescue theta.

The previous permutation result is sufficiently close that my current prior remains that a substantial amount of the theta benefit will prove to be **target-conditioning structure rather than physical coordinate semantics**.

But this reference removes the strongest objection to that conclusion: that we tested the mechanism inside the wrong architecture.

If RQ1 still cannot separate from RQ2, we can say something considerably stronger.

If RQ1 does separate, we will know exactly where to look next: convolutional spatial synthesis, Z1/Z2 factorization, source-vector geometry, or the multitask physiologic constraint.

### References

Ansari, R. A., Trivedi, R. K., Brennan, K. A., et al. (2026). Deep learning–based continuous QT monitoring to identify high-risk prolongation events after Class III antiarrhythmic initiation. *Circulation, 153*(1), 35–46. [https://doi.org/10.1161/CIRCULATIONAHA.125.077494](https://doi.org/10.1161/CIRCULATIONAHA.125.077494)

Chen, J., Zheng, X., Yu, H., Chen, D. Z., & Wu, J. (2021). Electrocardio Panorama: Synthesizing new ECG views with self-supervision. *Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence*, 3597–3605. [https://doi.org/10.24963/ijcai.2021/495](https://doi.org/10.24963/ijcai.2021/495)

Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 7132–7141. [https://doi.org/10.1109/CVPR.2018.00745](https://doi.org/10.1109/CVPR.2018.00745)

Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2017). Aggregated residual transformations for deep neural networks. *2017 IEEE Conference on Computer Vision and Pattern Recognition*, 5987–5995. [https://doi.org/10.1109/CVPR.2017.634](https://doi.org/10.1109/CVPR.2017.634)

Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. *arXiv*. [https://doi.org/10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
