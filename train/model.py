"""AlphaZero-style residual network for Quoridor.

Input/output layout must match the Kotlin side:
  engine/.../nn/FeatureEncoder.kt  (PLANES input planes, seen from the side to move)
  engine/.../nn/PolicyIndex.kt     (3 policy planes: step target, H wall, V wall)
"""
import torch
from torch import nn
import torch.nn.functional as F

ENCODER_VERSION = 1   # FeatureEncoder.VERSION
INPUT_PLANES = 10     # FeatureEncoder.PLANES
POLICY_PLANES = 3     # PolicyIndex.PLANES


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class QuoridorNet(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 6):
        super().__init__()
        self.channels, self.blocks = channels, blocks
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU())
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        # Fully convolutional policy: one logit per board square per policy plane.
        self.policy = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, POLICY_PLANES, 1))
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.value_fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        h = self.tower(self.stem(x))
        policy = self.policy(h).flatten(1)             # [B, 3*n*n], plane-major
        v = self.value_conv(h)
        pooled = torch.cat([v.mean(dim=(2, 3)), v.amax(dim=(2, 3))], dim=1)
        value = torch.tanh(self.value_fc(pooled)).squeeze(1)   # [B]
        return policy, value
