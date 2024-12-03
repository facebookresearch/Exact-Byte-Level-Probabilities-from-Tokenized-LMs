# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

sys.path.append("./src")

from src.byte_ensemble_model import ByteEnsemble
import argparse

parser = argparse.ArgumentParser(description="Input a list of paths")
parser.add_argument("--paths", nargs="+", help="List of paths", required=True)
parser.add_argument("--prompt", type=str, help="Prompt", required=True)
parser.add_argument("--device", type=str, help="device", default='cuda')
parser.add_argument("--temp", type=float, help="List of paths", default=0.6)
parser.add_argument(
    "--max_new_bytes", type=int, help="max bytes to generate", default=500
)

"""
Example checkpoint: /whatever/Yi-1.5-6B, contains:
     - README.md
     - tokenizer.json
     - tokenizer.model
     - model-00001-of-00003.safetensors
     ...
"""


def main():
    args = parser.parse_args()

    temp = args.temp
    device = args.device
    # Load model to different devices
    mixture = ByteEnsemble([path for path in args.paths], device=device)

    # generate text
    prompt = args.prompt
    mixture.generate(
        prompt, fast_mode=True, max_new_bytes=args.max_new_bytes, temperature=temp
    )


if __name__ == "__main__":
    main()
