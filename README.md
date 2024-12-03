# Exact Byte-Level Probabilities from Tokenized Language Models for FIM-Tasks and Model Ensembles

This repo is an official implementation for converting tokenized LLM to byte-level LLM ([arxiv:2410.09303](https://arxiv.org/abs/2410.09303)).

Please 🌟star🌟 this repo and cite our paper 📜 if you like (and/or use) our work, thank you! 

We support Llama2, Mistral and Yi (i.e. sentencepiece tokenizer).

## 0. Installation

-  Install python3.10 and requirements.txt manually:
```bash
  conda create -n byte_conversion python=3.10.14
  conda activate byte_conversion
  pip install -r requirements.txt
```

## 1. Byte Generation Example. 
To generate bytes with HuggingFace models from $path1 and $path2, run the following command.
```bash
  path1=... # path to Huggingface model, e.g. /whatever/Yi-1.5-6B
  
  path2=... # additional models for ensemble.
  
  prompt_string=... # Example "def add5(aa): return 5 + a"
  
  max_bytes=... # Max number of bytes to generate.
  
  temp=... # Temperature scaling.
  
  device=... #'gpu' or 'cpu'
  
  python3 generate.py --paths $path1 $path2 --prompt $prompt --max_new_bytes $max_bytes --temp $temp --device $device
```

## 2. Citation

If you use our work in your research please cite [our paper](https://arxiv.org/abs/2410.09303):

```
@inproceedings{phan2024exactbytelevelprobabilitiestokenized,
      title={Exact Byte-Level Probabilities from Tokenized Language Models for FIM-Tasks and Model Ensembles}, 
      author={Buu Phan and Brandon Amos and Itai Gat and Marton Havasi and Matthew Muckley and Karen Ullrich},
      year={2024},
      eprint={2410.09303},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2410.09303}, 
}
```

## 3. Legal

Our work is licenced under CC-BY-NC, please refer to the [LICENSE](LICENSE) file in the top level directory.

Copyright © Meta Platforms, Inc. See the [Terms of Use](https://opensource.fb.com/legal/terms/) and [Privacy Policy](https://opensource.fb.com/legal/privacy/) for this project.