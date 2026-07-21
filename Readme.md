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

## 唐及唐以前语料

仓库自带 `data/pre_tang/input.txt`，内容严格限定为唐代及以前成书或创作的文本。它不是单一诗歌集，而是覆盖经、史、子、集和字书的综合语料：

- 经与字书：完整十三经相关正文、《四书》和《说文解字》，包括《周易》《尚书》《礼记》《周礼》《仪礼》《左传》等。
- 史：前十五部正史，从《史记》《汉书》《三国志》到《隋书》《南史》《北史》，另有《国语》《战国策》《竹书纪年》等。
- 子：包括《老子》《庄子》《墨子》《荀子》《韩非子》《管子》《孙子兵法》《吕氏春秋》《淮南子》《论衡》等。
- 医、算及杂类：包括《黄帝内经》《九章算术》《周髀算经》《山海经》等。
- 集：完整全唐诗 57,607 首，以及《楚辞》和曹操诗集。

当前训练集约 2,008 万字符、58.9 MB，涵盖 81 部作品或作品集合。构建脚本采用年代白名单，明确排除宋代《资治通鉴》、宋诗、五代诗词、宋词和元曲。完整书目、固定版本、去重规则和许可证见 [`data/pre_tang/SOURCES.md`](data/pre_tang/SOURCES.md)。

如需从上游 JSON 重新生成：

```bash
git clone --depth 1 https://github.com/gujilab/chinese-classical-corpus.git /tmp/chinese-classical-corpus
git clone --depth 1 https://github.com/direct-phonology/ect-krp.git /tmp/ect-krp
git clone --depth 1 https://github.com/chinese-poetry/chinese-poetry.git /tmp/chinese-poetry

python scripts/build_pre_tang_corpus.py \
  --classical-source /tmp/chinese-classical-corpus \
  --early-source /tmp/ect-krp \
  --poetry-source /tmp/chinese-poetry
```

构建脚本按书名去除三个来源之间的重复版本，保留作品标题、篇章、作者、正文和上游标点，删除 JSON 元数据与 UUID，并用空行分隔记录。上游数据同时包含简体与繁体字，本项目不做自动转换，以免转换过程改动古籍用字。

## 训练

如果你没有模型训练经验，请先用浏览器打开 [`training_guide.html`](training_guide.html)。它用费曼式比喻逐步说明语料、字符表、批次、RNN/LSTM、损失、反向传播、验证、checkpoint 和生成，并提供可交互的训练命令生成器。训练完成后，可打开 [`time_capsule_lab.html`](time_capsule_lab.html)，了解“唐前惊奇度实验”与“古文续写器”的严谨做法和实施路线。

默认使用唐及唐以前语料训练 LSTM：

```bash
python train.py --epochs 30
```

显式选择普通 RNN：

```bash
python train.py --data data/pre_tang/input.txt --model rnn --epochs 30
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

### 从 checkpoint 精修

`--resume` 会恢复 checkpoint 中的模型参数，并继续训练 `--epochs` 个额外轮次。新 checkpoint 会保存 AdamW 优化器状态；旧 checkpoint 若没有该状态，也可以恢复模型参数并以指定的学习率开始精修。建议将输出写到新目录，保留原始最佳模型：

```bash
python -u train.py \
  --resume checkpoints/full/best.pt \
  --data data/pre_tang/input_normalized.txt \
  --output-dir checkpoints/refined \
  --batch-size 8 \
  --seq-length 128 \
  --epochs 5 \
  --learning-rate 0.0003 \
  --device cuda
```

恢复训练必须使用与原 checkpoint 相同的训练文本；程序会检查字符表是否一致。这里的 `--epochs 5` 表示额外训练 5 轮，因此输出会从第 11 轮显示到第 15 轮。

数据文件必须是 UTF-8 编码。模型使用 `--train-fraction`（默认 `0.95`）之前的时间块训练，其余时间块验证。

## 生成

选择验证损失最低的 checkpoint：

```bash
python sample.py checkpoints/best.pt --prompt "《秋日》" --length 300 --temperature 0.8
```

更保守的生成：

```bash
python sample.py checkpoints/best.pt --prompt "山中" --length 300 --top-k 20 --temperature 0.7
```

确定性地始终选择概率最高的字符：

```bash
python sample.py checkpoints/best.pt --prompt "明月" --length 300 --greedy
```

## 主要文件

| 文件 | 作用 |
| --- | --- |
| `train.py` | 训练、验证与 checkpoint 保存 |
| `sample.py` | 从 checkpoint 生成文本 |
| `char_rnn/data.py` | Unicode tokenizer 与 BPTT 批处理 |
| `char_rnn/model.py` | RNN/GRU/LSTM 网络 |
| `char_rnn/checkpoints.py` | checkpoint 读写 |
| `scripts/build_pre_tang_corpus.py` | 从上游 JSON 重建唐及唐以前训练集 |
| `data/pre_tang/input.txt` | 默认中文古典语料 |
| `training_guide.html` | 零基础费曼式交互训练教程 |
| `time_capsule_lab.html` | 唐前惊奇度实验与古文续写器说明页 |

## 与旧实现的对应

旧版 `train.lua` 的 `rnn_size` 对应新版 `--hidden-size`；新版另加 `--embedding-dim`，用可学习的嵌入替代 one-hot 输入。旧版 RMSProp 换为 AdamW，checkpoint 从 `.t7` 换为 `.pt`，并移除了对已弃用 Torch7/CUDA/OpenCL Lua 扩展的依赖。
