# 唐及唐以前综合语料：来源与书目

`input.txt` 是供字符级语言模型使用的 UTF-8 纯文本语料。筛选标准是作品成书或作者年代不晚于唐代；后世整理本可以作为数字底本，但晚于唐代创作的正文、注疏和译文不纳入。

## 当前规模

- 20,082,131 个 Unicode 字符，58,885,132 UTF-8 字节
- 645,070 行，字符表大小 17,507
- 81 部作品或作品集合，70,101 条原始记录
- SHA-256：`97ed61897be785974f3c0468977d3965b13ca89ef6ee860a02a631a3c8839a74`

## 来源一：chinese-classical-corpus

- 仓库：[`gujilab/chinese-classical-corpus`](https://github.com/gujilab/chinese-classical-corpus)
- 固定版本：`09a6873e46760f20027e20bf48814f5c5cc05d66`
- 许可证：仓库 `output/` 数据为 [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
- 纳入 29 部、11,715 条记录、约 1,266 万字符
- 经与字书：《周易》《尚书》《诗经》《礼记》《春秋左传》《春秋公羊传》《春秋穀梁传》《孝经》《尔雅》《论语》《孟子》《大学》《中庸》《说文解字》
- 正史：《史记》《汉书》《后汉书》《三国志》《晋书》《宋书》《南齐书》《梁书》《陈书》《魏书》《北齐书》《周书》《隋书》《南史》《北史》
- 明确排除：同一数据集中的宋代《资治通鉴》全部 294 卷、4,697,829 字

构建脚本还会检查每条记录的 `era`；只有周、春秋、战国、汉、晋、南朝宋、南朝梁、北齐和唐能够通过白名单。

## 来源二：Early Chinese Text Corpus

- 仓库：[`direct-phonology/ect-krp`](https://github.com/direct-phonology/ect-krp)
- 固定版本：`b28fab8f54b0e3ec3ca97cc2baa8caecfe71259f`
- 许可证：[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- 修改说明：保留上游已经移除注疏、标点、空白和非汉字字符的正文；本项目增加书名和换行，并排除与来源一及诗歌来源重复的版本
- 去重后纳入 49 部、688 条记录、约 258 万字符

具体书目：

《韩诗外传》《周礼》《仪礼》《大戴礼记》《春秋繁露》《释名》《竹书纪年》《汉记》《逸周书》《东观汉记》《国语》《战国策》《晏子春秋》《吴越春秋》《越绝书》《孔子家语》《荀子》《新语》《新书》《盐铁论》《说苑》《新序》《法言》《潜夫论》《申鉴》《中论》《孙子兵法》《管子》《商君书》《韩非子》《黄帝内经》《周髀算经》《九章算术》《太玄经》《墨子》《鹖冠子》《公孙龙子》《吕氏春秋》《淮南子》《白虎通》《独断》《论衡》《风俗通义》《新论》《山海经》《穆天子传》《老子道德经》《列子》《庄子》。

## 来源三：chinese-poetry

- 仓库：[`chinese-poetry/chinese-poetry`](https://github.com/chinese-poetry/chinese-poetry)
- 固定版本：`b8594f81a89752241442f2ce267d6f66f96704ee`
- 许可证：MIT，副本见 [`THIRD_PARTY_LICENSE`](THIRD_PARTY_LICENSE)
- 纳入：《楚辞》65 篇、曹操诗集 26 篇、`全唐诗/poet.tang.*.json` 共 57,607 首
- 明确排除：`poet.song.*.json`、五代诗词、宋词、元曲和其他晚期目录

## 去重与文本处理

- 《周易》《尚书》《诗经》《礼记》《左传》《公羊传》《穀梁传》《孝经》《尔雅》《论语》《孟子》《史记》《汉书》《说文解字》优先采用带结构和标点的 `chinese-classical-corpus` 版本，ECT 中的对应版本不重复加入。
- 《楚辞》优先采用带篇章和作者信息的 `chinese-poetry` 版本。
- 保留繁简混合状态，不做自动繁简转换，以免改变古籍用字。
- 保留原有标点；ECT 本身为无标点版本，本项目不自动补标点。
- 去掉 JSON 字段和 UUID，保留书名、篇章名、作者与正文，不同记录以空行分隔。

## 重新构建

```bash
git clone --depth 1 https://github.com/gujilab/chinese-classical-corpus.git /tmp/chinese-classical-corpus
git clone --depth 1 https://github.com/direct-phonology/ect-krp.git /tmp/ect-krp
git clone --depth 1 https://github.com/chinese-poetry/chinese-poetry.git /tmp/chinese-poetry

python scripts/build_pre_tang_corpus.py \
  --classical-source /tmp/chinese-classical-corpus \
  --early-source /tmp/ect-krp \
  --poetry-source /tmp/chinese-poetry
```

三个上游都可能继续更新；若要严格复现当前文件，请将各仓库检出到上面列出的固定提交。
