# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import List, Tuple
from dataclasses import dataclass
from .trie import TrieNode


@dataclass
class SamplerState:
    P_tT: torch.Tensor
    mask: torch.Tensor
    utf8_seq: List[int]
    cvrs_logprobs: torch.Tensor
    covers: List[str]
    cover_encs: List[List[int]]
    prev_context_enc: List[int]
    div_enc: List[int]
    old_cache: Tuple
    old_cache_enc: List[int]
    new_cache: Tuple
    new_cache_enc: List[int]


def extract_sampler_state(sampler_state):
    """
    Simply return values of sampler state.
    """
    P_tT = sampler_state.P_tT
    mask = sampler_state.mask
    cvrs_logprobs = sampler_state.cvrs_logprobs
    covers = sampler_state.covers
    prev_covers_encs = sampler_state.cover_encs
    prev_context_enc = sampler_state.prev_context_enc
    div_enc = sampler_state.div_enc
    assert prev_context_enc[: len(div_enc)] == div_enc

    return (
        P_tT,
        mask,
        cvrs_logprobs,
        covers,
        prev_covers_encs,
        prev_context_enc,
        div_enc,
    )


def utf8trie():
    """
    Return trie of all valid utf-8 representation.
    """
    root = TrieNode()
    for i in range(0x110000):
        try:
            char = chr(i)
            utf8_bytes = list(char.encode("utf-8"))
            root.insert(utf8_bytes)
        except ValueError:
            pass
    return root


def str2bytes(white_space, inp_str):
    """
    Convert string -> bytes.
    """
    if white_space == "▁":
        inp_str = inp_str.replace(" ", "▁")
    return list(inp_str.encode("utf-8"))


def whitespace_split(white_space, raw_str):
    """
    Check if the string contains a whitespace
    appearing after a word. Then split raw_str
    into 2 parts where cond_str is the prefix
    before the last white space.
    """
    break_point = -1
    for i in range(1, len(raw_str)):
        if raw_str[i] == white_space and raw_str[i - 1] != white_space:
            break_point = i

    assert break_point > 0, "invalid string"
    cond_str = raw_str[:break_point]
    query_str = raw_str[break_point:]
    return cond_str, query_str
