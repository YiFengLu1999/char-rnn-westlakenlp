#!/usr/bin/env python3
"""Build a broad Tang-and-earlier corpus from three open text collections."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator


ALLOWED_ERAS = {"周", "春秋", "战国", "汉", "晋", "南朝宋", "南朝梁", "北齐", "唐"}

# These works already have structured, punctuated versions in chinese-classical-corpus
# or chinese-poetry. Excluding their ECT copies prevents large duplicate weights.
EARLY_TEXT_DUPLICATES = {
    "易經",
    "尚書",
    "詩經",
    "禮記",
    "春秋左傳",
    "春秋公羊傳",
    "春秋穀梁傳",
    "孝經",
    "爾雅",
    "論語",
    "孟子",
    "史記",
    "漢書",
    "說文解字",
    "楚辭",
}


def load_json(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source_file:
        value = json.load(source_file)
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return value


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected a JSON object at {path}:{line_number}")
                yield value


def clean_lines(lines: Iterable[str]) -> list[str]:
    return [line.strip() for line in lines if isinstance(line, str) and line.strip()]


def poetry_block(item: dict, *, default_author: str = "") -> str:
    lines: list[str] = []
    title = item.get("title")
    author = item.get("author", default_author)
    if isinstance(title, str) and title.strip():
        lines.append(f"《{title.strip()}》")
    if isinstance(author, str) and author.strip():
        lines.append(author.strip())
    content = item.get("paragraphs", item.get("content", []))
    lines.extend(clean_lines(content))
    return "\n".join(lines)


def build_classical_blocks(source_root: Path) -> tuple[list[str], Counter[str]]:
    corpus_path = source_root / "output" / "corpus.jsonl"
    blocks: list[str] = []
    counts: Counter[str] = Counter()
    for item in iter_jsonl(corpus_path):
        era = item.get("era")
        source = item.get("source", "")
        if era not in ALLOWED_ERAS:
            continue
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Missing source name in record {item.get('id')}")
        content = item.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        chapter = item.get("chapter") or item.get("title") or item.get("section")
        header = chapter.strip() if isinstance(chapter, str) and chapter.strip() else source.strip()
        blocks.append(f"《{header}》\n{content.strip()}")
        counts[source.strip()] += 1
    if not blocks:
        raise ValueError(f"No Tang-or-earlier records found in {corpus_path}")
    return blocks, counts


def build_early_text_blocks(source_root: Path) -> tuple[list[str], Counter[str]]:
    metadata_path = source_root / "metadata.json"
    with metadata_path.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    blocks: list[str] = []
    counts: Counter[str] = Counter()
    for path in sorted((source_root / "jsonl").glob("*.jsonl")):
        title = metadata.get(path.stem)
        if not isinstance(title, str):
            raise ValueError(f"Missing title for {path.stem} in {metadata_path}")
        if title in EARLY_TEXT_DUPLICATES:
            continue
        texts = [item.get("text", "").strip() for item in iter_jsonl(path)]
        texts = [text for text in texts if text]
        if texts:
            blocks.append(f"《{title}》\n" + "\n".join(texts))
            counts[title] = len(texts)
    if not blocks:
        raise ValueError(f"No supplementary early texts found under {source_root / 'jsonl'}")
    return blocks, counts


def build_poetry_blocks(source_root: Path) -> tuple[list[str], Counter[str]]:
    blocks: list[str] = []
    counts: Counter[str] = Counter()

    chuci = load_json(source_root / "楚辞" / "chuci.json")
    blocks.extend(poetry_block(item) for item in chuci)
    counts["楚辞"] = len(chuci)

    caocao = load_json(source_root / "曹操诗集" / "caocao.json")
    blocks.extend(poetry_block(item, default_author="曹操") for item in caocao)
    counts["曹操诗集"] = len(caocao)

    tang_files = sorted(
        (source_root / "全唐诗").glob("poet.tang.*.json"),
        key=lambda path: int(path.stem.split(".")[-1]),
    )
    if not tang_files:
        raise FileNotFoundError(f"No poet.tang.*.json files under {source_root / '全唐诗'}")
    for path in tang_files:
        poems = load_json(path)
        blocks.extend(poetry_block(item) for item in poems)
        counts["全唐诗"] += len(poems)
    return blocks, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classical-source", required=True, type=Path, help="chinese-classical-corpus checkout")
    parser.add_argument("--early-source", required=True, type=Path, help="direct-phonology/ect-krp checkout")
    parser.add_argument("--poetry-source", required=True, type=Path, help="chinese-poetry checkout")
    parser.add_argument("--output", type=Path, default=Path("data/pre_tang/input.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blocks: list[str] = []
    all_counts: Counter[str] = Counter()
    for builder, source in (
        (build_classical_blocks, args.classical_source),
        (build_early_text_blocks, args.early_source),
        (build_poetry_blocks, args.poetry_source),
    ):
        new_blocks, counts = builder(source)
        blocks.extend(new_blocks)
        all_counts.update(counts)

    text = "\n\n".join(block for block in blocks if block.strip()) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8", newline="\n")
    print(
        f"wrote {args.output}: {len(text):,} characters; "
        f"{len(all_counts):,} works/collections; {sum(all_counts.values()):,} records"
    )


if __name__ == "__main__":
    main()
