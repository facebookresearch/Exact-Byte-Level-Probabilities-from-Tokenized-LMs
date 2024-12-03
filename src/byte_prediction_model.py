# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple
import logging
import torch
import numpy as np
import copy
from byte_gen_utils import (
    sample_top_p,
    log2prob_rescale,
    temp_scale_prob,
    gemma_init_cache,
)
from custom_base_model import BaseModel
from string_processor.tokenizer_wrapper import ByteTokenizer
from string_processor.string_helper import *


class BytePredLLM(BaseModel):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        device: "cpu",
    ):

        super().__init__(pretrained_model_name_or_path, device)
        del self.t_wrapper
        self.t_wrapper = ByteTokenizer(pretrained_model_name_or_path)
        self.t_wrapper.sum_prefix_idx = self.t_wrapper.sum_prefix_idx.to(self.device)

        logging.info("Warning: Recheck the case when prompt ends with special token.")

    def run_TR(
        self,
        inp_enc,
        fast_mode=False,
        cache=None,
        cache_enc=[],
        use_cache=True,
    ):
        """
        Next-token probabilities, i.e. compute P(.|t^k_1).
        Truncate invalid encoding probabilities to 0.0 if fast_mode=False.
        Set fast_mode=True for speed-up.

        Args:
            inp_enc (List): input encodings.
            fast_mode (bool): False for disable truncation.
            cache (Tuple): KV cache from previous computation, as return from hf models.
            cache_enc (List): encodings correspond to cache.
            use_cache (bool): True/False if we are using KV caching or not.

        Returns:
            logprobs (torch.Tensor): next-token log probabilities
            torch.Tensor: mask of invalid (0.0) /valid (1.0) tokens.
            cache (Tuple): return cache from hf models.
        """
        with torch.no_grad():
            if self.llm.__class__.__name__ == "Gemma2ForCausalLM" and cache == None:
                cache = gemma_init_cache(self.llm, inp_enc)
            logprobs, cache = self.logprobs_next_token(
                input_ids=inp_enc,
                past_key_values=copy.deepcopy(cache),
                use_cache=use_cache,
                cache_enc=cache_enc,
            )
        if fast_mode:
            return logprobs, torch.ones_like(logprobs).to(self.device), cache

        return logprobs, self.t_wrapper.isvalid(inp_enc).to(self.device), cache

    def _compute_tokens_likelihood(
        self,
        query_toks,
        ctx_toks,
    ):
        """
        Computes logP(query_toks|ctx_toks)

        Args:
            query_toks (List): query encodings.
            ctx_toks (List): contex encodings.

        Returns:
            torch.Tensor: logP(query_toks|ctx_toks)

        Note:
            encodings must be valid.
        """
        logp = 0.0

        for tok in query_toks:
            logprobs, mask, _ = self.run_TR(ctx_toks, use_cache=False)
            assert mask[tok] == 1
            logp += logprobs[tok]
            ctx_toks.append(tok)
        return logp

    def cover_token_likelihoods(
        self,
        left_enc,
        div_enc,
    ):
        """
        Compute logP(t_{m+1}=t, t^m_n|t^n_1). See diagram.
        |  t^n_1       |       t^m_n      |   t_{m+1}    |
        |  div_enc     |    left tokens   | right tokens |
        |          left_enc               | right tokens |

        Args:
            left_enc (List): left encodings.
            div_enc (List): encodings ends before the last whitespace.

        Returns:
            torch.Tensor: logP(t_{m+1}=t, t^m_n|t^n_1)
            mask (torch.Tensor): invalid map (1.0 = valid, 0.0 = invalid).
        """

        assert left_enc[: len(div_enc)] == div_enc
        logprobs, _mask, _ = self.run_TR(left_enc, use_cache=False)
        logpval = self._compute_tokens_likelihood(
            query_toks=left_enc[len(div_enc) :],
            ctx_toks=copy.deepcopy(div_enc),
        )
        likelihood_per_encs = logpval + logprobs
        return likelihood_per_encs, _mask

    def extract_cover_prob_bytes(
        self,
        raw_str,
    ):
        """
        Compute P(t | div_enc) for all cover encs of raw_str.
        div_enc: encoding corresponds to the prefix of raw_str + stops before the last white space.
        Step 1: Preprocess String --> Bytes.
        Step 2: Extract cover encodings + compute prob.

        Args:
            raw_str (str): unprocessed string. must end with a valid utf-8 characters.

        Returns:
            cover_strings (List[str]): list of all strings represented by cover encodings (in bytes)
            torch.Tensor : log probability of all cover encodings.
            covers_encs (List[List[Int]]): list of all cover encodings.
            div_enc (List[Int]): Encoding that stops at the last white space in raw_str.

        Note:
        Step 1: For simplification, we force the string to end with a valid utf-8 character.
        It must also contain a white space (exclude the dummy prefix one as in llama2)
        since some models such as Yi does not provide P(t_1).
        """
        assert isinstance(raw_str, str), "Input must be string."
        assert len(raw_str) != 0, "String length must be larger than 0."

        # Step 1.
        raw_str = raw_str.replace(" ", self.t_wrapper.white_space)
        cond_str, query_str = whitespace_split(self.t_wrapper.white_space, raw_str)
        div_enc = self.t_wrapper.encode_str(cond_str)  # with bos if required.
        query_bytes = str2bytes(self.t_wrapper.white_space, query_str)

        # Step 2.
        cover_strings, cover_logprobs, covers_encs = [], [], []

        for i in range(len(query_bytes)):
            left_byte_ids = query_bytes[:i]
            right_tks = self.t_wrapper.supertoken(query_bytes[i:])
            if len(right_tks) == 0:
                continue

            try:  # make sure that they are valid utf-8 splits.
                left_str = cond_str + bytes(left_byte_ids).decode("utf-8")
                left_enc = self.t_wrapper.encode_str(left_str)
            except UnicodeDecodeError:
                continue

            propososal_encs = [left_enc + [last_tok] for last_tok in right_tks]
            mask = self.t_wrapper._cover_enc_check(propososal_encs).to(self.device)
            valids = []

            for m in range(len(mask)):
                if mask[m] == 1:
                    valids.append(propososal_encs[m])

            likelihood_per_encs, _mask = self.cover_token_likelihoods(left_enc, div_enc)

            for encs in valids:
                assert _mask[encs[-1]] == 1
                cover_strings.append(self.t_wrapper._enc_to_byte(encs))
                covers_encs.append(encs[len(div_enc) :])
                cover_logprobs.append(likelihood_per_encs[encs[-1]])

            assert len(cover_strings) == len(
                set(cover_strings)
            ), "Warning! Duplicated Encoding!"

        return cover_strings, torch.stack(cover_logprobs), covers_encs, div_enc

    def prob_next_byte(
        self,
        raw_utf8_seq,
        fast_mode=False,
        sampler_state: Optional[SamplerState] = None,
        last_byte: Optional[int] = None,
    ) -> Tuple[torch.Tensor, SamplerState]:
        """
        Consists of two steps:
        1) Extract the cover encodings of the input utf8-seq, i.e. obtain cover(x^n_1).
        2) Compute P(x|T) where T = enc(utf8-seq) and compute P(x^{n+1}|x_{n}_1).

        Args:
            raw_utf8_seq (List[Int]): prompt as bytes input (x^n_1).
            fast_mode (bool): False for disable truncation.
            sampler_state (SamplerState): meta data from previous runs, contains cover(x^{n-1}) and KV caches.
            last_byte (int): previously sampled byte, i.e. x_n.

        Returns:
            predictions (torch.Tensor): next-byte probability P(x_{n+1}|x^n_1).
            sampler_state (SamplerState): new meta data, i.e cover(x_n).
        """
        assert raw_utf8_seq is not None, "Raw input must be specified."

        if self.t_wrapper.add_prefix_space:
            utf8_seq = list(self.t_wrapper.white_space.encode("utf-8")) + raw_utf8_seq
        else:
            utf8_seq = copy.deepcopy(raw_utf8_seq)

        # Step 1
        if sampler_state is None:  # Compute P(x^n_1)
            inp_str = bytes(utf8_seq).decode("utf-8")

            if self.t_wrapper.add_prefix_space:
                assert inp_str[0] == self.t_wrapper.white_space
                ctx_enc = self.t_wrapper.encode_str(inp_str[1:])
                (
                    raw_covers,
                    cvrs_logprobs,
                    cover_encs,
                    div_enc,
                ) = self.extract_cover_prob_bytes(inp_str[1:])
            else:
                ctx_enc = self.t_wrapper.encode_str(inp_str)
                (
                    raw_covers,
                    cvrs_logprobs,
                    cover_encs,
                    div_enc,
                ) = self.extract_cover_prob_bytes(inp_str)
            assert ctx_enc[len(div_enc) :] in cover_encs, "recheck the encoding."
            covers = [cover[len(utf8_seq) :] for cover in raw_covers]
            assert (
                ctx_enc == div_enc + cover_encs[covers.index(b"")]
            ), "encoding(string) from encode str and one computed from extract cover must be the same"
        else:  # Update P(x^n_1) from previous sampler.
            (
                cvrs_logprobs,
                ctx_enc,
                covers,
                cover_encs,
                utf8_seq,
                div_enc,
            ) = self.update_sampler_state(sampler_state, utf8_seq, last_byte, fast_mode)

        # Step 2
        predictions, sampler_state = self.next_byte_update(
            cvrs_logprobs,
            ctx_enc,
            covers,
            cover_encs,
            utf8_seq,
            fast_mode,
            div_enc,
            sampler_state,
        )

        return predictions, sampler_state

    def update_sampler_state(self, sampler_state, utf8_seq, last_byte, fast_mode):
        """
        Extract the cover encodings of the input utf8-seq, i.e. obtain cover(x^n_1) from cover(x^{n-1}_1).
        Step 1: Load the sampler state + encode new context.
        Step 2: Extract cover encodings of x^{n+1}_1.
              a) P(t, T): filter t where x_{n+1} t_1.
              b) Remove t = cover(x^n_1) and do not contain x_{n+1}.

        Args:
            raw_utf8_seq (List[Int]): prompt as bytes input (x^n_1).
            fast_mode (bool): False for disable truncation.
            sampler_state (SamplerState): meta data from previous runs, contains cover(x^{n-1}) and KV caches.
            last_byte (int): previously sampled byte, i.e. x_n.

        Returns:
            cvrs_logprobs: log probability of cover of x^n_1
            ctx_enc: encode of (x^n_1)
            covers: cover string of x^n_1
            new_cover_encs: cover encodings of x^n_1.
            utf8_seq: x^n_1
            new_div_enc: Encoding that stops at the last white space in x^n_1.
        """
        assert (
            last_byte == utf8_seq[-1] and last_byte < 256
        ), "Make sure last byte < 256 and consistent with utf8_seq."
        assert (
            sampler_state.utf8_seq == utf8_seq[:-1]
        ), "utf-8 seq is not consistent with sampler_state.utf8_seq"

        # Step 1.
        (
            P_tT,
            mask,
            cvrs_logprobs,
            covers,
            prev_covers_encs,
            prev_context_enc,
            div_enc,
        ) = extract_sampler_state(sampler_state)
        ctx_enc, _ = self.t_wrapper._encode_bytes(utf8_seq, prev_context_enc)

        # Step 2a.
        log_PtT = torch.log(P_tT)
        new_covers, new_encs, idx_temp = [], [], []
        mask_np = mask.detach().cpu().numpy()

        #################################################################
        # Note: This block function is very specific for sentencepiece.
        # In some cases, bytes such as \n are not duplicated and used.
        # In most cases, character 'a' has two representation (byte and piece)
        # sentencepiece will use the piece representation most of the time.
        for tok_id in self.t_wrapper.map_byte2id[last_byte]["list_toks"]:
            if mask_np[tok_id] != 0:
                if self.t_wrapper.is_byte(tok_id):
                    byte_check = self.t_wrapper.map_id2byte[tok_id]
                    assert byte_check == last_byte
                    to_add = bytes([byte_check])
                else:
                    to_add = self.t_wrapper.id2token(tok_id).encode("utf-8")
                enc_to_add = prev_context_enc[len(div_enc) :] + [tok_id]
                if not fast_mode:
                    assert (
                        not to_add in covers
                    ), "Collision happened. Please recheck utf-8 encoding"
                new_covers.append(to_add)
                new_encs.append(enc_to_add)
                idx_temp.append(tok_id)
        ###################################################################

        new_logprobs = log_PtT[idx_temp]
        if len(new_logprobs) > 0:
            cvrs_logprobs = torch.cat((cvrs_logprobs, new_logprobs))
            covers = covers + new_covers
            cover_encs = prev_covers_encs + new_encs
        else:
            cover_encs = prev_covers_encs

        # Step 2b.
        tmp_covers, tmp_cvrs_logprobs, tmp_cover_encs = [], [], []
        for k, cover_str in enumerate(covers):
            if len(cover_str) == 0:
                continue
            if cover_str[0] == last_byte:
                tmp_covers.append(cover_str[1:])
                tmp_cvrs_logprobs.append(cvrs_logprobs[k])
                tmp_cover_encs.append(cover_encs[k])
        covers = tmp_covers
        cvrs_logprobs = torch.stack(tmp_cvrs_logprobs)
        cover_encs = tmp_cover_encs

        # Readjust the conver encs.
        new_loc = -1
        for i in range(1, len(ctx_enc)):
            if (
                self.t_wrapper.id2token(ctx_enc[i])[0] == self.t_wrapper.white_space
                and self.t_wrapper.id2token(ctx_enc[i - 1])[-1]
                != self.t_wrapper.white_space
            ):
                new_loc = i
        new_div_enc = ctx_enc[:new_loc] if new_loc > 0 else div_enc

        if len(new_div_enc) != len(div_enc):
            len_diff = len(new_div_enc) - len(div_enc)
            new_cover_encs = [c[len_diff:] for c in cover_encs]
        else:
            new_cover_encs = cover_encs

        return (
            cvrs_logprobs,
            ctx_enc,
            covers,
            new_cover_encs,
            utf8_seq,
            new_div_enc,
        )

    def next_byte_update(
        self,
        cvrs_logprobs,
        ctx_enc,
        covers,
        cover_encs,
        utf8_seq,
        fast_mode,
        div_enc,
        sampler_state,
    ):
        """
        Compute P(x_{n+1}|x^n_1).

        Step 1: Compute logP(t|T), T = ctx_enc, also rescale.
        Step 2: Compute P(x, T) through marginalization.
        Step 3: Compute P(x, T') for other cover encodings T'.
        Step 4: Compute P(x|prev_x).

        Conditioning separated by _, e.g. log_pt_T = log(p_t|T)

        Args:
            cvrs_logprobs: log probability of cover of x^n_1
            ctx_enc: encode of (x^n_1)
            covers: cover string of x^n_1
            cover_encs: cover encodings of x^n_1.
            utf8_seq: x^n_1
            fast_mode (bool): False for disable truncation.
            div_enc (List[Int]): Encoding that stops at the last white space in x^n_1
            sampler_state: contains cover(x^{n-1}_1) and KV caches.

        Returns:
            predictions: P(x_{n+1}|x^n_1)
            sampler_state: new meta data, contains cover(x^n_1).
        """

        # Step 1:
        log_pt_T, mask, cache = self._run_kv(ctx_enc, sampler_state, fast_mode, div_enc)
        cvrs_scaled_probs, cvrs_logprobs = log2prob_rescale(cvrs_logprobs)

        # Step 2:
        log_pT = cvrs_logprobs[cover_encs.index(ctx_enc[len(div_enc) :])]
        ptT = torch.exp(log_pT + log_pt_T) * mask
        pxT = torch.matmul(self.t_wrapper.sum_prefix_idx, ptT)

        # Step 3:
        mat_sum = torch.zeros(len(pxT), len(cvrs_scaled_probs), dtype=torch.float32)
        temp_list = []

        for k in range(len(covers)):
            if len(covers[k]) == 0:
                continue
            byte_id = covers[k][0]
            temp_list.append([byte_id, k])

        temp_list = np.asarray(temp_list)
        if len(temp_list) > 0:
            mat_sum[temp_list[:, 0], temp_list[:, 1]] = 1.0

        # Step 4:
        P_unorm = torch.matmul(mat_sum.to(self.device), cvrs_scaled_probs) + pxT
        predictions = P_unorm / P_unorm.sum()

        # Update the kv cache.
        if sampler_state is None:
            old_cache_enc = div_enc
            _, _, old_cache = self.run_TR(
                ctx_enc[: len(old_cache_enc)], fast_mode, cache=None, use_cache=True
            )
        else:
            old_cache = sampler_state.old_cache
            old_cache_enc = sampler_state.old_cache_enc

        try:  # see if the string is a valid utf-8
            bytes(utf8_seq).decode("utf-8")
            new_cache = cache
            new_cache_enc = ctx_enc
        except UnicodeError:  # if it's not, keep the old one.
            new_cache = sampler_state.new_cache
            new_cache_enc = sampler_state.new_cache_enc

        sampler_state = SamplerState(
            P_tT=ptT,
            mask=mask,
            utf8_seq=utf8_seq,
            cvrs_logprobs=cvrs_logprobs,
            covers=covers,
            cover_encs=cover_encs,
            prev_context_enc=ctx_enc,
            div_enc=div_enc,
            old_cache=old_cache,
            old_cache_enc=old_cache_enc,
            new_cache=new_cache,
            new_cache_enc=new_cache_enc,
        )
        return predictions, sampler_state

    def _run_kv(
        self,
        ctx_enc,
        sampler_state,
        fast_mode,
        div_enc,
    ):
        """
        Next-token prediction, deciding which KV cache to use.
        Compute log P(t_{i+1}| t^i_1).

        Args:
            ctx_enc (List[int]): context encoding.
            sampler_state (SamplerState): meta data from previous runs, contains cover(x^{n-1}) and KV caches.
            temperature (float): temperature to scale probability.
            fast_mode (bool): False for disable truncation.
            div_enc (List[int]): encodings end before the last whitespace.

        Returns:
            log_pt_T (torch.Tensor): log P(t_{i+1}| t^i_1) for all t_{i+1}
            mask (torch.Tensor): invalid encoding masks.
            cache : KV cache of t^i.
        """
        if sampler_state != None:
            new_cache_enc = sampler_state.new_cache_enc
            if new_cache_enc == div_enc:
                sampler_state.old_cache = sampler_state.new_cache
                sampler_state.old_cache_enc = sampler_state.new_cache_enc
            else:
                assert sampler_state.old_cache_enc == div_enc
                assert sampler_state.old_cache != None
            cache = sampler_state.old_cache
            cache_enc = sampler_state.old_cache_enc
        else:
            cache = None
            cache_enc = []

        log_pt_T, mask, cache = self.run_TR(
            ctx_enc, fast_mode, cache, cache_enc=cache_enc, use_cache=True
        )

        return log_pt_T, mask, cache

    def generate(
        self,
        input_str: str,
        fast_mode=False,
        temperature=1.0,
        top_p=1.0,
        max_new_bytes: int = 100,
    ):
        """Main call for the wrapper, API similar to [1].

        [1] https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/generation/configuration_utils.py#L71

        Args:
            input_str (str): unprocessed string. must end with a valid utf-8 characters.
            fast_mode (bool): whether to truncate invalid encodings (False) or not (True). This is for speed-up generations.
            temperature (float): temperature to scale probability.
            top_p (float): top_p value for predictions.
            max_new_tokens (int): number to bytes to generate.

        Returns:
            inp_bytes: output generated bytes.
        """

        # input string --> utf-8 bytes
        input_str = input_str.replace(" ", self.t_wrapper.white_space)
        inp_bytes = list(input_str.encode("utf-8"))
        sampler_state = None
        last_byte = None

        for _ in range(max_new_bytes):
            predictions, sampler_state = self.prob_next_byte(
                inp_bytes,
                fast_mode=fast_mode,
                sampler_state=sampler_state,
                last_byte=last_byte,
            )

            if temperature < 0.1:
                last_byte = predictions.argmax().item()
            else:
                predictions = temp_scale_prob(predictions, temperature)
                last_byte = sample_top_p(predictions, top_p, return_probs=False).item()

            inp_bytes = inp_bytes + [last_byte]
            try:
                print(bytes(inp_bytes).decode("utf-8"))
            except:
                pass

        return inp_bytes
