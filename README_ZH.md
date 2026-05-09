# Chinese NER Using Lattice LSTM

Lattice LSTM 是一个用于中文命名实体识别（NER）的模型。它将字符级 LSTM 与词格（Lattice）嵌入结合，用于建模中文序列标注任务。

模型和结果可参考 ACL 2018 论文 [Chinese NER Using Lattice LSTM](https://arxiv.org/pdf/1805.02023.pdf)。在 MSRA 数据集上，该方法取得了 93.18% 的 F1 值。

## 技术栈

本仓库当前的声明环境以 `pyproject.toml` 为准：

- Python: 3.11
- PyTorch: >= 2.2, < 3
- NumPy: >= 1.26
- 包管理: uv

说明：仓库中的部分源码仍保留了较早版本的实现风格，实际运行时如果出现兼容性问题，可能需要继续做少量适配。

## 输入格式

输入采用 CoNLL 格式，建议使用 BIOES 标注方案。每一行包含一个字及其标签，句子之间用空行分隔。

```text
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
```

## 预训练嵌入

原始 README 中提到的预训练字符向量和词向量如下：

- 字符向量：`gigaword_chn.all.a2b.uni.ite50.vec`
- 词格向量：`ctb.50d.vec`

下载链接：

- 字符向量：[Google Drive](https://drive.google.com/file/d/1_Zlf0OAZKVdydk7loUpkzD2KPEotUE8u/view?usp=sharing) 或 [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D)
- 词格向量：[Google Drive](https://drive.google.com/file/d/1K_lG3FlXTgOOf8aQ4brR9g3R40qi1Chv/view?usp=sharing) 或 [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D)

这两个文件都需要放到 `data` 目录下。

## 如何运行

1. 下载字符向量和词格向量，并放入 `data` 目录。
2. 按需修改 `main.py`、`run_main.sh` 或 `run_demo.sh` 中的训练、验证、测试文件路径。
3. 运行训练脚本，例如：

```bash
uv run main.py
```

如果你想跑示例数据，也可以使用仓库里现成的 `run_demo.sh`。

## Resume NER 数据

`ResumeNER` 目录中的数据来源于新浪财经爬取的上市公司高管简历数据，具体说明可参考论文。

## 引用

如果你使用了本项目，请引用以下论文：

```bibtex
@article{zhang2018chinese,
  title={Chinese NER Using Lattice LSTM},
  author={Yue Zhang and Jie Yang},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2018}
}
```