# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from transformers import AutoTokenizer
from sentencepiece import SentencePieceProcessor
import os
import numpy as np
import torch
from .string_helper import *
import copy


class BaseTokenizer:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
    ):
        """
        Initialize the Base Tokenizer (SentencePiece).

        Args:
            pretrained_model_name_or_path (str): Path to the HuggingFace model.

        Attributes:
            sp_model (SentencePieceProcessor): sentencepiece module for encoding/decoding.
            white_space: by sentencepiece default, it is "▁".
            add_bos_token: whether tokenizer includes bos.
            bos_token_id: corresponding bos token id
            eos_token_id: corresponding eos token id
            unknown_token_id: unknown token id
            vocab_size: number of tokens in vocabulary
            vocab: mapping from token to id
            inv_vocab: mapping from id to token
            add_prefix_space: whether it add a dummy prefix "▁" before the first token.
        """
        self.name_or_path = pretrained_model_name_or_path
        # get tokenizer
        sp_model_path_v1 = os.path.join(
            pretrained_model_name_or_path, "tokenizer.model"
        )
        sp_model_path_v3 = os.path.join(
            pretrained_model_name_or_path, "tokenizer.model.v3"
        )

        self.sp_model = SentencePieceProcessor()
        self.sentence_piece = False
        if os.path.exists(sp_model_path_v3):
            print(f"Loading model from {sp_model_path_v3}")
            self.sp_model.Load(sp_model_path_v3)
            self.white_space = "▁"
            self.sentence_piece = True
            self.white_space_id = self.token2id("▁")
        elif os.path.exists(sp_model_path_v1):
            print(f"Loading model from {sp_model_path_v1}")
            self.sp_model.Load(sp_model_path_v1)
            self.white_space = "▁"
            self.sentence_piece = True
            self.white_space_id = self.token2id("▁")
        else:
            raise FileNotFoundError(
                "Neither tokenizer.model nor tokenizer.model.v3 was found in the specified directory. Only support sentencepiece."
            )

        self._tokenizer_metadata()
        self.add_prefix_space = self.is_add_dummy_prefix()

    def _tokenizer_metadata(self):
        """
        Extract vocab details.
        """
        autokenizer = AutoTokenizer.from_pretrained(self.name_or_path, legacy=False)
        try:
            self.add_bos_token = autokenizer.add_bos_token
        except:
            self.add_bos_token = self.sp_model._add_bos
        self.bos_token_id = autokenizer.bos_token_id
        self.eos_token_id = autokenizer.eos_token_id
        self.unknown_token_id = autokenizer.unk_token_id

        # get vocab size
        self.vocab_size = self.sp_model.vocab_size()
        # token2id
        self.vocab = autokenizer.vocab
        # id2token
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def is_add_dummy_prefix(self):
        """
        Check if tokenizer add dummy prefix or not.
        """
        if self.sentence_piece == False:
            return False
        dummy_text = "a"
        encs = self.sp_model.encode(dummy_text)
        assert encs[0] != self.bos_token_id

        if self.id2token(encs)[0][0] == self.white_space:
            return True

        return False

    def encode_str(self, inp_str, post_process=True):
        """
        String -> encodings, includes bos and dummy_prefix
        """
        if self.sentence_piece:
            encs = self.sp_model.encode(inp_str)

            if not post_process:
                return encs

            if self.add_bos_token:
                encs = [self.bos_token_id] + encs
        else:
            encs = self.tokenizer.encode(inp_str)

            if not post_process:
                if self.add_bos_token and encs[0] == self.bos_token_id:
                    return encs[1:]
                else:
                    return encs

            if self.add_bos_token:
                if encs[0] == self.bos_token_id:
                    pass
                else:
                    encs = [self.bos_token_id] + encs
        return encs

    def decode(self, inp_enc):
        """
        Encoding -> String, includes removing bos and dummy_prefix
        """

        inp_str = self.sp_model.decode(inp_enc)
        return inp_str

    def id2token(self, token_id):
        """
        Returns the token using the Hugging Face tokenizer.

        Args:
        token_id (int): The ID for which to find the token (piece).

        Returns:
        str: The token of the id.
        """
        # return self.tokenizer.convert_ids_to_tokens(token_id)
        if self.sentence_piece:
            return self.sp_model.IdToPiece(token_id)
        else:
            return self.tokenizer.convert_ids_to_tokens(token_id)

    def token2id(self, token):
        """
        Returns the ID of the given token using the Hugging Face tokenizer.

        Args:
        token (str): The token (piece) for which to find the ID.

        Returns:
        int: The ID of the piece.
        """
        if self.sentence_piece:
            return self.sp_model.PieceToId(token)
        else:
            return self.tokenizer.convert_tokens_to_ids(token)

    def is_byte(self, token_id):
        """
        Checks if the given token is a byte, i.e. <0xFF>.
        This is a special case for sentencepiece, which has
        duplicates bytes versus character. ('a' appears twice in
        the vocabulary).

        Args:
        token_id (int): A single token id to check.

        Returns:
        bool: True if a single byte, False otherwise.
        """
        if self.sentence_piece:
            return self.sp_model.IsByte(token_id)
        return False

    def is_control(self, token_id):
        """
        Checks if the given token is a control token.

        Args:
        token_id (int): A single token id to check.

        Returns:
        bool: True if a control token, False otherwise.
        """
        if self.sentence_piece:
            return self.sp_model.IsControl(token_id)

    def is_unknown(self, token_id):
        """
        Checks if the given token is a control token.

        Args:
        token_id (int): A single token id to check.

        Returns:
        bool: True if unknown, False otherwise.
        """
        if self.sentence_piece:
            return self.sp_model.IsUnknown(token_id)
        else:
            return token_id == self.tokenizer.unk_token_id

    def is_unused(self, token_id):
        """
        Checks if the given token is an unused token.

        Args:
        token_id (int): A single token id to check.

        Returns:
        bool: True if unused, False otherwise.
        """
        if self.sentence_piece:
            return self.sp_model.IsUnused(token_id)
        else:
            return False

    def _cover_enc_check(self, potential_encs):
        """
        Check if encoding is valid. Only use for cover encodings.

        Args:
            potential_encs (List[List[int]]): list encodings.

        Returns:
            mask (torch.Tensor): 1.0/0.0 for valid/invalid encoding.
        """

        mask = np.ones(len(potential_encs))
        correct_enc = self.encode_str(self.decode(potential_encs), post_process=False)
        start_ptr = 1 if self.add_bos_token else 0

        for i in range(len(mask)):
            if correct_enc[i] != potential_encs[i][start_ptr:]:
                mask[i] = 0.0

        return torch.Tensor(mask)


class ByteTokenizer(BaseTokenizer):
    def __init__(self, pretrained_model_name_or_path: str):
        """
        Attributes:
        map_byte2id: mapping from byte x (0-255 + eos + bos + unk) to its corresponding token id. Also store tokens that has prefix x.
        map_id2byte: map from id to byte
        sum_prefix_idx: matrix (num_bytes x num_tokens) where 1.0 at position i,j indicates that byte i is a prefix of token j.
        """
        super().__init__(pretrained_model_name_or_path)

        # get utf8trie, alphabet and cover tokens from vocab
        self.utf8trie = utf8trie()
        self.set_alphabet()
        self._get_default_byte_masking()

    def isvalid(self, enc):
        """
        Check invalid next-token t_{k+1} for context encoding t^k_1. Assume white space pretokenization.
        This is a bit different from _cover_enc_check, where we need to take into account the case
        where the encoding ends with a byte that can potentially does not lead to a valid utf8 character.

        - Case 1: If the encoding ends with a byte token,
                  use byte token for the rest.
        - Case 2: If the encoding does not, use the one
                  from encode(str + 'byte'). check invalid

        Args:
            enc (List[int]): encoding (t^k_1).

        Returns:
            mask (torch.Tensor): 1.0/0.0 for valid/invalid encoding t^{k+1}_1.
        """

        is_end_utf8, byte_info = self._is_end_of_utf8(enc)

        if is_end_utf8:
            mask = self._process_end_utf8(enc)
        else:
            mask = self._process_not_end_utf8(byte_info)

        return mask

    def _process_end_utf8(self, inp_encs):
        """
        Masking when inp_encs ends with valid utf-8 encoding.
        Args:
            enc (List[int]): encoding (t^k_1).

        Returns:
            mask (torch.Tensor): 1.0/0.0 for valid/invalid encoding.
        """
        # Reduce inp_encs til the last whitespace.
        break_point = -1

        for i in range(len(inp_encs) - 1, 0, -1):
            t = self.id2token(inp_encs[i])
            prev_t = self.id2token(inp_encs[i - 1])

            if t[0] == self.white_space and prev_t[-1] != self.white_space:
                break_point = i
                break

        inp_encs = inp_encs[break_point:]
        assert break_point > 0

        # Construct potential and almost correct enc
        # Almost correct due to the special/byte tokens.
        potential_encs = [inp_encs + [i] for i in range(self.vocab_size)]
        almost_correct_enc = self.encode_str(
            self.decode(potential_encs), post_process=False
        )  # if add prefix, decode will remove the first one so no worries!

        mask = copy.deepcopy(self.default_mask)

        for i in range(len(mask)):
            if self.is_control(i) or self.is_unknown(i) or self.is_unused(i):
                mask[i] = 1.0  # Allow all control & unknown tokens.
                continue

            if self.is_byte(i) and self.map_id2byte[i] > 127:
                continue

            if almost_correct_enc[i] != potential_encs[i]:
                mask[i] = 0.0
                continue

            mask[i] = 1.0

        return torch.Tensor(mask)

    def _process_not_end_utf8(self, byte_info):
        """
        If enc ends with unfinished byte, the next token
        must be a byte token within byte_info. It also must not
        appear in the vocab.

        Args:
            byte_info (Tuple(List, List)): contains list of valid next bytes and the previous incomplete
            utf-8 byte sequence.

        Returns:
            mask (torch.Tensor): 1.0/0.0 for valid/invalid encoding.
        """
        allowed_next_byte = byte_info[0]
        suffix_byte = byte_info[1]

        mask = np.zeros(self.vocab_size)
        for _byte in allowed_next_byte:
            mask[self.map_byte2id[_byte]["vocab_index"]] = 1.0

            # If suffix merge + new byte = valid unicode character in vocab
            new_byte = suffix_byte + [_byte]
            if len(self.utf8trie.prefix_list(new_byte)) == 0:
                char = bytes(new_byte).decode("utf-8")
                if not self.is_unknown(self.token2id(char)):
                    mask[self.map_byte2id[_byte]["vocab_index"]] = 0.0
                    continue

        return torch.Tensor(mask)

    def _is_end_of_utf8(self, enc):
        """
        Check if enc ends with valid utf-8 character.
        Args:
            enc (List[int]): encodings
        Returns:
            bool: True if it ends with valid utf-8 character, False otherwise
            byte_info (Tuple()): None if valid utf-8 character. Else contains meta data for allowed next byte.
                For details, see _process_not_end_utf8(byte_info)
        """

        if not self.is_byte(enc[-1]):  # if last token is not byte -> end of utf-8
            return True, None

        suffix_byte = []
        for token in enc[::-1]:
            if not self.is_byte(token):
                break
            suffix_byte.append(self.map_id2byte[token])
        suffix_byte = suffix_byte[::-1]

        ptr = 0
        for i in range(len(suffix_byte) + 1):
            if len(self.utf8trie.prefix_list(suffix_byte[ptr:i])) == 0:
                ptr = i
        suffix_byte = suffix_byte[ptr:]

        if len(suffix_byte) == 0:
            return True, None
        else:
            return False, (self.utf8trie.prefix_list(suffix_byte), suffix_byte)

    def _get_default_byte_masking(self):
        """
        Set default mask for (token id of bytes > 127), following the allowed prefix byte
        of utf-8 characters.
        """
        mask = np.zeros(self.vocab_size)
        allowed_prefix = self.utf8trie.prefix_list([])

        for i in range(256):
            if i > 127 and i in allowed_prefix:
                mask[self.map_byte2id[i]["vocab_index"]] = 1.0
        self.default_mask = mask

    def supertoken(self, byte_ids):
        """
        Find tokens with prefix = bytes_id.

        Args:
            byte_ids (List[int]): list of input bytes (int).
        Returns:
            list_toks (List[int]): list of token ids that starts with byte_ids.
        """
        assert all(_ids < 256 for _ids in byte_ids)

        bytes_rep = bytes(byte_ids)
        list_toks = []

        for tok in self.vocab.keys():
            tok_id = self.vocab[tok]

            if self.is_byte(tok_id):
                # This is for the situation where the string ends with a byte
                byte_rep_tok_id = int(
                    self.id2token(tok_id)[1:-1], 0
                )  # int(tok[1:-1], 0)
                if len(byte_ids) == 1 and byte_ids[0] == byte_rep_tok_id:
                    list_toks.append(tok_id)
                continue

            if (
                tok_id >= self.vocab_size
                or self.is_control(tok_id)
                or self.is_unknown(tok_id)
                or self.is_unused(tok_id)
            ):  # if the last token
                continue

            tok_bytes = tok.encode("utf-8")
            if tok_bytes.startswith(bytes_rep):
                list_toks.append(tok_id)

        return list_toks

    def _enc_to_byte(self, enc):
        """
        Turn Encodings into Bytes. We need to take into account
        the scenario where encoding ends with invalid utf-8 byte token id.
        Args:
            enc (List[int]): input encoding.
        Returns:
            bytes : decode(enc) but takes into account utf-8 byte.
        """
        byte_list = []
        for token_id in enc:
            if (
                self.is_control(token_id)
                or self.is_unknown(token_id)
                or self.is_unused(token_id)
            ):
                continue
            if self.is_byte(token_id):
                byte_mapped = self.map_id2byte[token_id]
                byte_list = byte_list + [byte_mapped]
                continue

            byte_list = byte_list + list(self.id2token(token_id).encode("utf-8"))

        return bytes(byte_list)

    def _encode_bytes(self, utf8_seq, prev_context_enc):
        """
        Encode the input bytes --> token ids.
        Args:
            utf8_seq (List[int]): input byte id (0-255 + bos + eos + unk).
            pre_context_enc (List[int]): encoding of prefix(utf8_seq) that ends with valid utf8-seq.
        Returns:
            ctx_enc (List[int]): encoding (token ids).
            bytes_seq: associate byte sequence.
        """

        last_byte = utf8_seq[-1]
        bytes_seq = bytes(utf8_seq)

        try:
            inp_str = bytes_seq.decode("utf-8")
            if self.add_prefix_space:
                assert inp_str[0] == self.white_space, "Dummy Prefix is not added"
                ctx_enc = self.encode_str(inp_str[1:])
            else:
                ctx_enc = self.encode_str(inp_str)
        except UnicodeError:
            ctx_enc = prev_context_enc + [self.map_byte2id[last_byte]["vocab_index"]]

        return ctx_enc, bytes_seq

    def set_alphabet(self):
        """
        Extract the alphabet from vocab.
        """

        # Forward map: byte id -> vocab id
        fmap_alphabet = {}
        # Reverse map: (vocab_id): byteid.
        rmap_alphabet = {}

        # Map bytes \x00 to \xFF
        for i in range(self.vocab_size):
            if self.is_byte(i):
                piece = self.id2token(i)
                base_10_id = int(piece[1:-1], 0)
                fmap_alphabet[base_10_id] = {"vocab_index": i, "piece": piece}
                rmap_alphabet[i] = base_10_id

        assert max(fmap_alphabet.keys()) == 255, "Warning! Bytes mapping exceeds 255."

        # Map unk, bos, eos
        default_special_tokens = [
            self.bos_token_id,
            self.eos_token_id,
            self.unknown_token_id,
        ]
        fmap_alphabet[256] = {
            "vocab_index": self.unknown_token_id,
            "piece": self.id2token(self.unknown_token_id),
        }
        fmap_alphabet[257] = {
            "vocab_index": self.bos_token_id,
            "piece": self.id2token(self.bos_token_id),
        }
        fmap_alphabet[258] = {
            "vocab_index": self.eos_token_id,
            "piece": self.id2token(self.eos_token_id),
        }

        rmap_alphabet[self.unknown_token_id] = 256
        rmap_alphabet[self.bos_token_id] = 257
        rmap_alphabet[self.eos_token_id] = 258

        # Map other special tokens
        special_tokens_ids = 255 + 4
        for i in range(self.vocab_size):
            if i in default_special_tokens:
                continue

            if self.is_control(i) or self.is_unknown(i) or self.is_unused(i):
                piece = self.id2token(i)
                fmap_alphabet[special_tokens_ids] = {"vocab_index": i, "piece": piece}
                rmap_alphabet[i] = special_tokens_ids
                special_tokens_ids += 1
                continue

        # Extract cover tokens.
        for byte_base10 in range(len(fmap_alphabet.keys())):
            cover_tok = [fmap_alphabet[byte_base10]["vocab_index"]]

            # special tokens arent meant to be part of any merged.
            if byte_base10 > 255:
                fmap_alphabet[byte_base10]["list_toks"] = cover_tok
                continue
            for i in range(self.vocab_size):
                if (
                    self.is_control(i)
                    or self.is_unknown(i)
                    or self.is_unused(i)
                    or self.is_byte(i)
                ):
                    continue

                piece = self.id2token(i)
                b_piece = list(piece.encode("utf-8"))
                if b_piece[0] == byte_base10:
                    cover_tok.append(i)

            fmap_alphabet[byte_base10]["list_toks"] = cover_tok

        sum_idx = torch.zeros(
            len(fmap_alphabet.keys()), self.vocab_size, dtype=torch.float32
        )
        for k in fmap_alphabet.keys():
            sum_idx[k, fmap_alphabet[k]["list_toks"]] = 1.0
        sum_idx = sum_idx

        self.map_byte2id = fmap_alphabet
        self.map_id2byte = rmap_alphabet
        self.sum_prefix_idx = sum_idx
