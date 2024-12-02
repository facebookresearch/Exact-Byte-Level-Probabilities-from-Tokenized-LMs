# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from transformers.cache_utils import HybridCache


def sample_top_p(probs, p, return_probs=False):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.
        return_probs (bool): Return all indexes and probabilities instead of sampling one.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    if return_probs:
        return probs_idx[~mask], probs_sort[~mask]
    else:
        return next_token


def temp_scale_prob(probs, temperature):
    """
    Perform temperature scaling.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        temperature (float): scaling temperature.

    Returns:
        torch.Tensor: temperature-scaled probability.

    Note: this is on probability (not logit) space.
    """
    eps = 1e-7
    scale_probs = torch.pow(probs + eps, 1.0 / temperature)
    return scale_probs / scale_probs.sum()


def log2prob_rescale(cvrs_logprobs):
    """
    Rescale log-probabilities of cover encodings
    such that the max(cvrs_logprobs) = 0.0.

    Args:
        cvrs_logprobs (torch.Tensor): cover encoding log-probabilities.

    Returns:
        cvrs_scaled_probs (torch.Tensor): rescaled probablities.
        cvrs_logprobs (torch.Tensor): rescaled log-probablities.
    """
    max_log = torch.max(cvrs_logprobs)
    cvrs_logprobs -= max_log
    cvrs_scaled_probs = torch.exp(cvrs_logprobs)
    return cvrs_scaled_probs, cvrs_logprobs


def gemma_init_cache(llm, inp_enc):
    # Only for gemma which use HybridCache.
    cache = HybridCache(
        config=llm.config,
        max_batch_size=1,
        max_cache_len=len(inp_enc) + 500,
        device=llm.device,
        dtype=llm.dtype,
    )
    return cache
