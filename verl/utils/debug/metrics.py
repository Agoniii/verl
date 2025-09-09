# Copyright 2025 Individual Contributor: TomQunChaoA
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch

from verl.protocol import DataProto

logger = logging.getLogger(__file__)


def calculate_token_list_diff(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # verify inputs
    if tensor1.numel() == 0 or tensor2.numel() == 0:
        return torch.zeros(tensor1.shape[0], dtype=torch.long, device=tensor1.device)
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        print(
            f"<WARN> dim of tensor1, tensor2, mask is not equal, {(tensor1.shape)=},{(tensor2.shape)=}, {(mask.shape)=}"
        )
        return torch.ones_like(tensor1)
    # transfer to same device
    if tensor2.device != tensor1.device:
        tensor2 = tensor2.to(tensor1.device)
    if mask.device != tensor1.device:
        mask = mask.to(tensor1.device)

    # calculate diff
    diff_mask = tensor1 != tensor2

    valid_diff_mask = diff_mask & (mask == 1)

    diff_counts = valid_diff_mask.sum(dim=1)

    return diff_counts


def pearson_correlation_coefficient(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # implemention of https://arxiv.org/pdf/2506.13585
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        return 0
    mt1 = torch.masked_select(tensor1, mask)
    mt2 = torch.masked_select(tensor2, mask)
    result = torch.corrcoef(torch.stack([mt1, mt2], dim=0))
    return result[0][1].detach().item()


def calculate_log_prob_diff(log_probs1: torch.Tensor, log_probs2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    full_diff = torch.abs(log_probs1 - log_probs2)
    return torch.masked_select(full_diff, mask)


def calculate_debug_metrics(data: DataProto, tis_imp_ratio_cap: float = -1.0) -> dict:
    """
    Calculate rollout vs actor logprobs diff and TIS-related metrics for debugging and monitoring purpose

    Args:
        data: DataProto
            the data batch to calculate
            rollout_log_probs: log_probs record when rollout forward tokens
            old_log_probs(actor log probs): log_probs record when actor forward tokens
            loss_mask or attention_mask: to mask unrelated token
            responses: the response tokens, for calculating size
        tis_imp_ratio_cap: float
            Truncated Importance Sampling ratio cap. If > 0, enables TIS metrics calculation
    Returns:
        dict: metrics
            "training/rollout_probs_diff_valid": 1->input is valid, 0->input is invalid
            "training/rollout_probs_diff_max": max value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_mean": mean value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_std": std value of logprob diff of rollout vs. actor
            "training/rollout_actor_probs_pearson_corr": logprob's pearson corrcoef of rollout vs. actor
            "training/tis_imp_ratio_mean": mean of TIS importance sampling ratios (if TIS enabled)
            "training/tis_imp_ratio_max": max of TIS importance sampling ratios (if TIS enabled)
            "training/tis_imp_ratio_std": std of TIS importance sampling ratios (if TIS enabled)
            "training/tis_truncation_ratio": ratio of samples that were truncated by TIS cap (if TIS enabled)
            "training/kl_divergence_mean": mean KL divergence between rollout and actor policies
            "training/kl_divergence_max": max KL divergence between rollout and actor policies
            "training/policy_ratio_mean": mean policy ratio (exp(actor_log_prob - rollout_log_prob))
            "training/policy_ratio_std": std policy ratio
    """

    rollout_old_log_probs = data.batch["rollout_log_probs"]
    actor_old_log_probs = data.batch["old_log_probs"]
    if "response_mask" in data.batch:
        logger.debug("response mask found, use it to mask log probs")
        log_prob_mask = data.batch["response_mask"]
    elif "attention_mask" in data.batch:
        log_prob_mask = data.batch["attention_mask"]
    else:
        logger.warning(f"no mask info found, use all log probs, {(data.batch.keys())=}")
        log_prob_mask = torch.ones_like(rollout_old_log_probs)
    responses = data.batch["responses"]
    response_length = responses.size(1)

    response_mask = log_prob_mask[:, -response_length:]
    response_mask_bool = response_mask.bool()
    
    # Calculate basic probability differences
    actor_probs = torch.exp(actor_old_log_probs)
    rollout_probs = torch.exp(rollout_old_log_probs)
    pearson_corrcoef = pearson_correlation_coefficient(actor_probs, rollout_probs, response_mask_bool)
    rollout_probs_diff = calculate_log_prob_diff(actor_probs, rollout_probs, response_mask_bool)
    
    # Calculate KL divergence between rollout and actor policies
    kl_divergence = actor_old_log_probs - rollout_old_log_probs  # KL(actor || rollout)
    kl_divergence_masked = torch.masked_select(kl_divergence, response_mask_bool)
    
    # Calculate policy ratio (importance sampling ratio)
    policy_ratio = torch.exp(actor_old_log_probs - rollout_old_log_probs)
    policy_ratio_masked = torch.masked_select(policy_ratio, response_mask_bool)
    
    # Base metrics
    metrics = {
        "training/rollout_probs_diff_valid": 1,
        "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
        "training/rollout_actor_probs_pearson_corr": pearson_corrcoef,
        "training/kl_divergence_mean": torch.mean(kl_divergence_masked).detach().item(),
        "training/kl_divergence_max": torch.max(kl_divergence_masked).detach().item(),
        "training/policy_ratio_mean": torch.mean(policy_ratio_masked).detach().item(),
        "training/policy_ratio_std": torch.std(policy_ratio_masked).detach().item(),
    }
    
    # TIS-specific metrics if enabled
    if tis_imp_ratio_cap > 0:
        # Calculate TIS importance sampling ratios (same as policy_ratio but with different semantic meaning)
        tis_imp_ratio = policy_ratio_masked
        tis_imp_ratio_capped = torch.clamp(tis_imp_ratio, max=tis_imp_ratio_cap)
        
        # Calculate truncation statistics
        truncation_mask = tis_imp_ratio > tis_imp_ratio_cap
        truncation_ratio = torch.mean(truncation_mask.float()).detach().item()
        
        # Add TIS-specific metrics
        metrics.update({
            "training/tis_imp_ratio_mean": torch.mean(tis_imp_ratio).detach().item(),
            "training/tis_imp_ratio_max": torch.max(tis_imp_ratio).detach().item(),
            "training/tis_imp_ratio_std": torch.std(tis_imp_ratio).detach().item(),
            "training/tis_imp_ratio_capped_mean": torch.mean(tis_imp_ratio_capped).detach().item(),
            "training/tis_truncation_ratio": truncation_ratio,
            "training/tis_imp_ratio_cap_value": tis_imp_ratio_cap,
        })
        
        logger.debug(f"TIS metrics: cap={tis_imp_ratio_cap}, truncation_ratio={truncation_ratio:.4f}, "
                    f"mean_ratio={torch.mean(tis_imp_ratio).item():.4f}")
    
    return metrics
