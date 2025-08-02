"""Train a single policy with the specified loss function.

This function creates a deep copy of the input policy and trains it using the provided
loss function and data. It supports both training and validation phases, with optional
Weights & Biases logging for metrics tracking.

Args:
    policy (MLPPolicy): The base policy to copy and train.
    train_dataloader (torch.utils.data.DataLoader): DataLoader containing training data
        with (X, y) batches.
    loss_function (L.LossFunction): Loss function implementing compute_loss method.
    n_updates (int, optional): Number of training epochs. Defaults to 1.
    learning_rate (float, optional): Learning rate for Adam optimizer. Defaults to 0.001.
    val_dataloader (Optional[torch.utils.data.DataLoader], optional): DataLoader for
        validation data. If None, validation is skipped. Defaults to None.
    wandb_run (optional): Weights & Biases run object for logging metrics. If None,
        no logging is performed. Defaults to None.

Returns:
    MLPPolicy: A trained copy of the input policy.

Notes:
    - The function applies gradient clipping with max_norm=1e9
    - Training and validation metrics are logged to wandb if wandb_run is provided
    - Only scalar metrics (int/float) are logged to wandb
    - The original policy is not modified (deep copy is used)
"""

import copy
from typing import Optional

import torch
from tqdm import tqdm

import nll_to_po.training.loss as L
from nll_to_po.models.dn_policy import MLPPolicy


def train_single_policy(
    policy: MLPPolicy,
    train_dataloader: torch.utils.data.DataLoader,
    loss_function: L.LossFunction,
    n_updates: int = 1,
    learning_rate: float = 0.001,
    val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    wandb_run=None,
):
    """Train a single policy with the specified loss function"""

    trained_policy = copy.deepcopy(policy).train()
    optimizer = torch.optim.Adam(trained_policy.parameters(), lr=learning_rate)

    # Training loop
    for epoch in tqdm(range(n_updates), desc="Training epochs"):
        # Training phase
        trained_policy.train()
        epoch_loss = 0
        epoch_grad_norm = 0

        batch_count = 0
        for X, y in train_dataloader:
            loss, metrics = loss_function.compute_loss(trained_policy, X, y)

            if wandb_run is not None:
                # Filter to only log scalar metrics
                scalar_metrics = {
                    f"train/{k}": v
                    for k, v in metrics.items()
                    if isinstance(v, (int, float))
                }
                wandb_run.log(scalar_metrics)

            optimizer.zero_grad()
            loss.backward()

            # Compute gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trained_policy.parameters(), max_norm=1e9
            )

            optimizer.step()

            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_grad_norm += grad_norm.item()

            batch_count += 1

        # Store averaged training metrics
        avg_loss = epoch_loss / batch_count
        avg_grad_norm = epoch_grad_norm / batch_count

        # Track training metrics
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": avg_loss,
                    "train/grad_norm": avg_grad_norm,
                }
            )

        # Validation phase
        if val_dataloader is not None:
            trained_policy.eval()
            val_epoch_loss = 0
            val_batch_count = 0

            with torch.no_grad():
                for X, y in val_dataloader:
                    val_loss, metrics = loss_function.compute_loss(trained_policy, X, y)
                    val_epoch_loss += val_loss.item()

                    if wandb_run is not None:
                        # Log validation metrics
                        scalar_metrics = {
                            f"val/{k}": v
                            for k, v in metrics.items()
                            if isinstance(v, (int, float))
                        }
                        wandb_run.log(scalar_metrics)

                    val_batch_count += 1

            avg_val_loss = val_epoch_loss / val_batch_count

            # Track validation metrics
            if wandb_run is not None:
                wandb_run.log({"val/loss": avg_val_loss})

    return trained_policy


# def compare_dist_unified(
#     policy: MLPPolicy,
#     X: torch.tensor,
#     y: torch.tensor,
#     loss_types: List[L.LossType],
#     n_updates: int = 1,
#     learning_rate: float = 0.001,
#     # Policy optimization parameters
#     n_generations_grpo: int = 5,
#     rsample_for_grpo: bool = False,
#     reward_transform: str = "normalize",  # "normalize", "rbf", "none"
#     RBF_gamma: Optional[float] = None,
# ):
#     """
#     Unified comparison function that trains policies with specified loss types

#     Args:
#         policy: Base policy to copy and train
#         X: Input tensor
#         y: Target tensor
#         loss_types: List of LossType enums specifying which losses to use
#         n_updates: Number of training updates
#         learning_rate: Learning rate for all optimizers
#         n_generations_grpo: Number of generations for policy optimization
#         rsample_for_grpo: Whether to use rsample for policy optimization
#         reward_transform: Type of reward transformation ("normalize", "rbf", "none")
#         RBF_gamma: Gamma parameter for RBF kernel (if using RBF transform)

#     Returns:
#         Dictionary mapping loss type names to their results
#     """

#     results = {}

#     for loss_type in loss_types:
#         if loss_type == LossType.DETERMINISTIC:
#             loss_fn = DeterministicLoss()
#         elif loss_type == LossType.SUPERVISED_NLL:
#             loss_fn = SupervisedNLLLoss()
#         elif loss_type == LossType.SUPERVISED_MSE:
#             loss_fn = SupervisedMSELoss()
#         elif loss_type == LossType.POLICY_OPTIMIZATION:
#             loss_fn = PolicyOptimizationLoss(
#                 n_generations=n_generations_grpo,
#                 use_rsample=rsample_for_grpo,
#                 reward_transform=reward_transform,
#                 rbf_gamma=RBF_gamma
#             )
#         else:
#             raise ValueError(f"Unknown loss type: {loss_type}")

#         # Train policy with this loss function
#         policy_results = train_single_policy(
#             policy=policy,
#             X=X,
#             y=y,
#             loss_function=loss_fn,
#             n_updates=n_updates,
#             learning_rate=learning_rate
#         )

#         results[loss_type.value] = policy_results

#     return results

# # Backward compatibility function
# def compare_dist(
#     policy: MLPPolicy,
#     X: torch.tensor,
#     y: torch.tensor,
#     n_updates: int = 1,
#     n_generations_grpo: int = 5,
#     learning_rate: float = 0.001,
#     sup_log_prob: bool = True,
#     rsample_for_grpo: bool = False,
#     RBF_gamma: Optional[float] = None,
# ):
#     """
#     Backward compatible version of compare_dist that maintains the original interface
#     """

#     # Determine loss types based on original parameters
#     loss_types = [LossType.DETERMINISTIC]

#     if sup_log_prob:
#         loss_types.append(LossType.SUPERVISED_NLL)
#     else:
#         loss_types.append(LossType.SUPERVISED_MSE)

#     loss_types.append(LossType.POLICY_OPTIMIZATION)

#     # Determine reward transform
#     reward_transform = "rbf" if RBF_gamma is not None else "normalize"

#     # Call unified function
#     results = compare_dist_unified(
#         policy=policy,
#         X=X,
#         y=y,
#         loss_types=loss_types,
#         n_updates=n_updates,
#         learning_rate=learning_rate,
#         n_generations_grpo=n_generations_grpo,
#         rsample_for_grpo=rsample_for_grpo,
#         reward_transform=reward_transform,
#         RBF_gamma=RBF_gamma
#     )

#     # Return in original format for backward compatibility
#     det = results.get("deterministic", {})
#     supervised = results.get("supervised_nll" if sup_log_prob else "supervised_mse", {})
#     grpo = results.get("policy_optimization", {})

#     return (det, supervised, grpo)
