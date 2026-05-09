Chinese NER Using Lattice LSTM
====

Lattice LSTM for Chinese NER. Character based LSTM with Lattice embeddings as input.

中文版本： [readme_zh.md](README_ZH.md)

Models and results can be found at our ACL 2018 paper [Chinese NER Using Lattice LSTM](https://arxiv.org/pdf/1805.02023.pdf). It achieves 93.18% F1-value on MSRA dataset, which is the state-of-the-art result on Chinese NER task.

Details will be updated soon.

Requirement
======

This repository's declared environment is defined in [pyproject.toml](pyproject.toml):

- Python: 3.11
- PyTorch: >= 2.2, < 3
- NumPy: >= 1.26
- Package manager: uv

The source code still contains some legacy implementation details from the original project, so additional compatibility fixes may be needed depending on your local environment.

Input format
======

CoNLL format (prefer BIOES tag scheme), with each character its label for one line. Sentences are splited with a null line.

 美 B-LOC
 国 E-LOC
 的 O
 华 B-PER
 莱 I-PER
 士 E-PER

 我 O
 跟 O
 他 O
 谈 O
 笑 O
 风 O
 生 O

Pretrained Embeddings
====

The pretrained character and word embeddings are the same with the embeddings in the baseline of [RichWordSegmentor](https://github.com/jiesutd/RichWordSegmentor)

Character embeddings (gigaword_chn.all.a2b.uni.ite50.vec): [Google Drive](https://drive.google.com/file/d/1_Zlf0OAZKVdydk7loUpkzD2KPEotUE8u/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D)

Word(Lattice) embeddings (ctb.50d.vec): [Google Drive](https://drive.google.com/file/d/1K_lG3FlXTgOOf8aQ4brR9g3R40qi1Chv/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D)

How to run the code?
====

The current runnable paths in this repository are:

- Resume NER training data: `./ResumeNER/train.char.bmes`, `./ResumeNER/dev.char.bmes`, `./ResumeNER/test.char.bmes`
- Demo data: `./data/demo.train.char`, `./data/demo.dev.char`, `./data/demo.test.char`
- Pretrained embeddings: `./data/gigaword_chn.all.a2b.uni.ite50.vec` and `./data/ctb.50d.vec`

To run the repo:

1. Make sure the pretrained embeddings are placed in the `data` folder.
2. Use `run_main.sh` for the ResumeNER data or `run_demo.sh` for the demo data.
3. You can also run `main.py` directly with the same paths if you want to override the defaults.

Examples:

```bash
bash run_main.sh
bash run_demo.sh
```

If you prefer to call the entry point directly, `main.py` now defaults to the ResumeNER files above.

uv environment setup
====

This repository is configured for Python 3.11 and the dependencies declared in [pyproject.toml](pyproject.toml).

1. Install uv.
2. Create or select a Python 3.11 interpreter.
3. From the repository root, run `uv sync` to install the declared dependencies from `pyproject.toml`.
4. Run `uv run main.py` or one of the shell scripts once the environment is ready.

Resume NER data
====

Crawled from the Sina Finance, it includes the resumes of senior executives from listed companies in the Chinese stock market. Details can be found in our paper.

Cite
========

Please cite our ACL 2018 paper:

    @article{zhang2018chinese,  
     title={Chinese NER Using Lattice LSTM},  
     author={Yue Zhang and Jie Yang},  
     booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL)},
     year={2018}  
    }
