# char-rnn (PyTorch)

一个现代 PyTorch 字符级语言模型项目。它从单个 UTF-8 文本文件学习预测下一个字符，并支持普通 RNN、GRU 与 LSTM。

原始 Lua/Torch7 代码保留在仓库中作为历史参考；`train.py` 与 `sample.py` 是新的实现入口。

## 特性

- Python 3 + PyTorch；CPU、CUDA 与 Apple Silicon MPS 均可用。
- 真正按 Unicode 字符切分文本，因此中文不会被 UTF-8 字节拆散。
- 可选 `rnn`、`gru`、`lstm`，使用 Embedding、AdamW 和梯度范数裁剪。
- 自动划分训练/验证集，保存 `checkpoints/best.pt` 和 `checkpoints/last.pt`。
- 支持提示文本、temperature、top-k 和贪心生成。

## 安装

建议先创建虚拟环境，再安装 PyTorch：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如需针对 CUDA 的特定 PyTorch wheel，请按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 的命令安装。

## 训练

仓库自带莎士比亚语料：

```bash
python train.py --data data/tinyshakespeare/input.txt --epochs 30
```

常用配置：

```bash
python train.py \
  --data data/my_corpus/input.txt \
  --model lstm \
  --embedding-dim 128 \
  --hidden-size 256 \
  --num-layers 2 \
  --dropout 0.2 \
  --batch-size 64 \
  --seq-length 128 \
  --device auto
```

数据文件必须是 UTF-8 编码。模型使用 `--train-fraction`（默认 `0.95`）之前的时间块训练，其余时间块验证。

## 生成

选择验证损失最低的 checkpoint：

```bash
python sample.py checkpoints/best.pt --prompt "ROMEO:" --length 1000 --temperature 0.8
```

更保守的生成：

```bash
python sample.py checkpoints/best.pt --prompt "第一章" --length 500 --top-k 20 --temperature 0.7
```

确定性地始终选择概率最高的字符：

```bash
python sample.py checkpoints/best.pt --prompt "ROMEO:" --length 500 --greedy
```

## 主要文件

| 文件 | 作用 |
| --- | --- |
| `train.py` | 训练、验证与 checkpoint 保存 |
| `sample.py` | 从 checkpoint 生成文本 |
| `char_rnn/data.py` | Unicode tokenizer 与 BPTT 批处理 |
| `char_rnn/model.py` | RNN/GRU/LSTM 网络 |
| `char_rnn/checkpoints.py` | checkpoint 读写 |

## 与旧实现的对应

旧版 `train.lua` 的 `rnn_size` 对应新版 `--hidden-size`；新版另加 `--embedding-dim`，用可学习的嵌入替代 one-hot 输入。旧版 RMSProp 换为 AdamW，checkpoint 从 `.t7` 换为 `.pt`，并移除了对已弃用 Torch7/CUDA/OpenCL Lua 扩展的依赖。
