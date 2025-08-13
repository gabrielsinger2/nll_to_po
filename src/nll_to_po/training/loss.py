"""Loss functions for training policies in NLL to PO framework."""

from abc import ABC, abstractmethod
# from enum import Enum

from typing import Optional

import torch
import torch.nn as nn


# class LossType(Enum):
#     """Enum for different loss types used in training policies."""
#     MSE = "mse"
#     NLL = "nll"
#     PO = "policy_optimization"


class LossFunction(ABC):
    """Abstract base class for loss functions"""

    name: str

    @abstractmethod
    def compute_loss(self, policy, X, y):
        """Compute the loss given policy, inputs X, and targets y"""
        pass


class MSE(LossFunction):
    """MSE loss using only the mean prediction"""

    name = "MSE"

    def compute_loss(self, policy, X, y):
        mean, _ = policy(X)
        loss = nn.MSELoss()(mean, y)
        return loss, {"mean_error": loss.item()}


class NLL(LossFunction):
    """Negative log-likelihood loss"""

    name = "NLL"

    def compute_loss(self, policy, X, y):
        mean, std = policy(X)
        dist = torch.distributions.Normal(mean, std)
        metrics = {
            "mean_error": nn.MSELoss()(mean.mean(dim=0), y.mean(dim=0)).item(),
            "NLL": -dist.log_prob(y).mean().item(),
            "dist": torch.distributions.Normal(mean[0].clone(), std[0].clone()),
        }
        return -dist.log_prob(y).mean(), metrics


class PO(LossFunction):
    """Policy optimization loss with configurable reward transformation"""

    name = "PO"

    def __init__(
        self,
        n_generations: int = 5,
        use_rsample: bool = False,
        reward_transform: str = "normalize",  # "normalize", "rbf", "none"
        rbf_gamma: Optional[float] = None,
    ):
        self.n_generations = n_generations
        self.use_rsample = use_rsample
        self.reward_transform = reward_transform
        self.rbf_gamma = rbf_gamma

    def _transform_rewards(self, rewards):
        """Apply reward transformation"""
        if self.reward_transform == "rbf" and self.rbf_gamma is not None:
            return torch.exp(self.rbf_gamma * rewards)
        elif self.reward_transform == "normalize":
            rewards_min, _ = rewards.aminmax(dim=0, keepdim=True)
            return rewards - rewards_min
        else:  # "none"
            return rewards

    def compute_loss(self, policy, X, y):
        mean, std = policy(X)
        dist = torch.distributions.Normal(mean, std)

        if self.use_rsample:
            samples = dist.rsample((self.n_generations,))
            rewards = -nn.MSELoss(reduction="none")(samples, y).mean(dim=-1)
            loss = -rewards.mean()
        else:
            samples = dist.sample((self.n_generations,))
            neg_log_prob = -dist.log_prob(samples).mean(dim=-1)
            rewards = -nn.MSELoss(reduction="none")(samples, y).mean(dim=-1)
            rewards = self._transform_rewards(rewards)
            loss = (neg_log_prob * rewards).mean()
        metrics = {
            "mean_error": nn.MSELoss()(mean.mean(dim=0), y.mean(dim=0)).item(),
            "NLL": -dist.log_prob(y).mean().item(),
            "dist": torch.distributions.Normal(mean[0].clone(), std[0].clone()),
        }
        return loss, metrics


class PO_Entropy(LossFunction):
    """Policy optimization loss with configurable reward transformation"""

    name = "PO_Entropy"

    def __init__(
        self,
        n_generations: int = 5,
        use_rsample: bool = False,
        reward_transform: str = "normalize",  # "normalize", "rbf", "none"
        rbf_gamma: Optional[float] = None,
        entropy_weight: float = 0.01,
    ):
        self.n_generations = n_generations
        self.use_rsample = use_rsample
        self.reward_transform = reward_transform
        self.rbf_gamma = rbf_gamma
        self.entropy_weight = entropy_weight

    def _transform_rewards(self, rewards):
        """Apply reward transformation"""
        if self.reward_transform == "rbf" and self.rbf_gamma is not None:
            return torch.exp(self.rbf_gamma * rewards)
        elif self.reward_transform == "normalize":
            rewards_min, _ = rewards.aminmax(dim=0, keepdim=True)
            return rewards - rewards_min
        else:  # "none"
            return rewards

    def compute_loss(self, policy, X, y):
        mean, std = policy(X)
        dist = torch.distributions.Normal(mean, std)

        if self.use_rsample:
            samples = dist.rsample((self.n_generations,))
            rewards = -nn.MSELoss(reduction="none")(samples, y).mean(dim=-1)
            loss = -rewards.mean()
        else:
            samples = dist.sample((self.n_generations,))
            neg_log_prob = -dist.log_prob(samples).mean(dim=-1)
            rewards = -nn.MSELoss(reduction="none")(samples, y).mean(dim=-1)
            rewards = self._transform_rewards(rewards)
            loss = (neg_log_prob * rewards).mean()

        loss += self.entropy_weight * dist.entropy().mean()

        metrics = {
            "mean_error": nn.MSELoss()(mean.mean(dim=0), y.mean(dim=0)).item(),
            "NLL": -dist.log_prob(y).mean().item(),
            "dist": torch.distributions.Normal(mean[0].clone(), std[0].clone()),
            "entropy": dist.entropy().mean().item(),
        }
        return loss, metrics
