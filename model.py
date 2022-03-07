import torch
from torch import nn

T = 1500


# Feature extractor network from https://arxiv.org/pdf/1707.03321.pdf (used as classifier here,
# the temporal classification is yet to be done)
class BasicModel(nn.Module):

    def __init__(self, in_channels=7):
        super().__init__()

        # First conv layer
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels,
            kernel_size=(in_channels, 1), stride=(1, 1),
            padding='same')  # VALID in paper, mistake ?

        self.conv2 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=(1, 25),
            stride=(1, 1),
            padding='same')
        self.relu2 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 6), stride=(1, 6))

        self.conv3 = nn.Conv2d(
            in_channels=8,
            out_channels=8,
            kernel_size=(1, 25),
            stride=(1, 1),
            padding='same')
        self.relu3 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 6), stride=(1, 6))

        self.dropout = nn.Dropout(p=0.5)
        self.layer_out = nn.Linear(in_channels * (T // 36) * 8, 5)

    def forward(self, x):
        # Start with (B, C, T)
        x = x[:, :, :, None]  # (B, C, T, 1)
        x = self.conv1(x)  # (B, C, T, 1)
        x = torch.permute(x, (0, 3, 1, 2))  # (B, 1, C, T)
        x = self.conv2(x)  # (B, 8, C, T)
        x = self.relu2(x)
        x = self.pool3(x)  # (B, 8, C, T//6)
        x = self.conv3(x)  # (B, 8, C, T//6)
        x = self.relu3(x)
        x = self.pool4(x)  # (B, 8, C, T//36)
        x = torch.flatten(x, start_dim=1)  # (B, 8*C*(T//36))
        x = self.dropout(x)
        x = self.layer_out(x)  # (B, 5)
        # No softmax since we are using CrossEntropyLoss which
        # expects logits as the model output not probabilities coming from softmax.
        return x
