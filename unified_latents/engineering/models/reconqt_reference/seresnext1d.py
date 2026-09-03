import torch
import torch.nn as nn


class SE1D(nn.Module):
    """
    1D Squeeze-and-Excitation channel recalibration module (Hu et al., 2018).
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
    ):
        super().__init__()
        hidden = max(channels // reduction, 8)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class SEResNeXtBottleneck1D(nn.Module):
    """
    1D ResNeXt bottleneck block with grouped convolutions and SE recalibration.
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        groups: int = 32,
        width_per_group: int = 4,
        se_reduction: int = 16,
    ):
        super().__init__()
        width = int(planes * width_per_group / 64.0) * groups
        out_channels = planes * self.expansion

        self.conv1 = nn.Conv1d(
            in_channels,
            width,
            kernel_size=1,
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
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.se = SE1D(
            out_channels,
            reduction=se_reduction,
        )
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)

        out = out + identity
        return self.relu(out)


class SEResNeXt1D(nn.Module):
    """
    Canonical 1D SE-ResNeXt backbone (Ansari et al., 2026; Xie et al., 2017; Hu et al., 2018).
    Default configuration: 32x4d cardinality, layers=(3, 4, 6, 3).
    """

    def __init__(
        self,
        in_channels: int = 1,
        layers: tuple = (3, 4, 6, 3),
        groups: int = 32,
        width_per_group: int = 4,
        se_reduction: int = 16,
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
            se_reduction=se_reduction,
        )
        self.layer2 = self._make_layer(
            128,
            layers[1],
            stride=2,
            groups=groups,
            width_per_group=width_per_group,
            se_reduction=se_reduction,
        )
        self.layer3 = self._make_layer(
            256,
            layers[2],
            stride=2,
            groups=groups,
            width_per_group=width_per_group,
            se_reduction=se_reduction,
        )
        self.layer4 = self._make_layer(
            512,
            layers[3],
            stride=2,
            groups=groups,
            width_per_group=width_per_group,
            se_reduction=se_reduction,
        )

        self.out_channels = 512 * SEResNeXtBottleneck1D.expansion  # 2048
        self._init_weights()

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
        groups: int = 32,
        width_per_group: int = 4,
        se_reduction: int = 16,
    ) -> nn.Sequential:
        modules = [
            SEResNeXtBottleneck1D(
                self.inplanes,
                planes,
                stride=stride,
                groups=groups,
                width_per_group=width_per_group,
                se_reduction=se_reduction,
            )
        ]
        self.inplanes = planes * SEResNeXtBottleneck1D.expansion

        for _ in range(1, blocks):
            modules.append(
                SEResNeXtBottleneck1D(
                    self.inplanes,
                    planes,
                    stride=1,
                    groups=groups,
                    width_per_group=width_per_group,
                    se_reduction=se_reduction,
                )
            )

        return nn.Sequential(*modules)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, in_channels, T] -> [B, 64, T/4]
        x = self.stem(x)
        # [B, 256, T/4]
        x = self.layer1(x)
        # [B, 512, T/8]
        x = self.layer2(x)
        # [B, 1024, T/16]
        x = self.layer3(x)
        # [B, 2048, T/32]
        x = self.layer4(x)
        return x
