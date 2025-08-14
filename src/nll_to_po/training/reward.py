"""Reward functions for training policies with PG."""

from abc import ABC, abstractmethod

import torch


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
        y = torch.squeeze(y)
        diff = y_hat - y
        return -torch.einsum("gbi,ij,gbj->gb", diff, self.matrix, diff)
