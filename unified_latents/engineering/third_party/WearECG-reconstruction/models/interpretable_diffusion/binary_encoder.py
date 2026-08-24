import torch
import torch.nn as nn


class BinaryActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播：直接传递梯度（直通估计器）
        return grad_output


class BinaryActivationModule(nn.Module):
    def __init__(self):
        super(BinaryActivationModule, self).__init__()

    def forward(self, x):
        return BinaryActivation.apply(x)


# 定义新的二值编码器
class BinaryEncoder(nn.Module):
    def __init__(self, input_shape=(12, 1000), latent_shape=(4, 125)):
        super(BinaryEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=input_shape[0], out_channels=16, kernel_size=4, stride=4
            ),
            nn.Conv1d(
                in_channels=16, out_channels=latent_shape[0], kernel_size=2, stride=2
            ),
            BinaryActivationModule(),
        )

    def forward(self, x):
        return self.encoder(x)


if __name__ == "__main__":

    print("输出张量的形状:", output.shape)  # 应输出 (bs, 4, 125)
    model = BinaryEncoder(input_shape=(12, 1000), latent_shape=(4, 125))
    x = torch.rand((32, 12, 1000))
    out = model(x)
    print("输出形状:", out.shape)
    print("部分输出值:", out[0, :, :10])
