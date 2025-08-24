"""Reward functions for training policies with PG."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardFunction(ABC):
    """Abstract base class for reward functions"""

    name: str

    @abstractmethod
    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the reward given generation y, and groundtruth y_star"""
        pass


class Mahalanobis(RewardFunction):
    """Mahalanobis reward: - (y-y_star)^T M (y-y_star)"""

    name = "Mahalanobis"

    def __init__(self, matrix: torch.Tensor):
        self.matrix = matrix

    def __call__(self, y_hat, y):
        y_hat = torch.squeeze(y_hat)
        print(y_hat.shape)
        y = torch.squeeze(y)
        print(y.shape)
        diff = y_hat - y
        return -torch.einsum("gbi,ij,gbj->gb", diff, self.matrix, diff)

class OneHotMahalanobis:
    def __init__(self, U: torch.Tensor, num_classes: int):
        self.U = U         # (C, C), SPD
        self.C = num_classes

    def __call__(self, y_hat, y):
        #print(y_hat)
        # y_hat, y: (G,B) class ids
        yh = torch.nn.functional.one_hot(y_hat, num_classes=self.C).float()  # (G,B,C)
        #yh=F.softmax(y_hat, dim=-1)
        yt = torch.nn.functional.one_hot(y,num_classes=self.C).float()  # (G,B,C)
        diff = yh - yt                                                        # (G,B,C)
        # - (diff^T U diff) per (g,b)
        return -torch.einsum("gbc,cd,gbd->gb", diff, self.U, diff)
