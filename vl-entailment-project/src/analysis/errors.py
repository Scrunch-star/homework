"""
错误分析：找出各方法预测错误的典型样本并可视化。
用法: python -m src.analysis.errors --num_examples 4
"""
import argparse
import json
import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import SNLIVEDataset
from src.utils.config import ANALYSIS_DIR, PRED_DIR

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

LABEL_COLORS = {
    "entailment": "#4CAF50",
    "neutral": "#FF9800",
    "contradiction": "#F44336",
}


def load_errors(pred_path: Path) -> list[dict]:
    """加载预测文件，返回预测错误的样本列表。"""
    with open(pred_path, encoding="utf-8") as f:
        data = json.load(f)
    return [record for record in data.get("results", []) if record["pred_label"] != record["gold_label"]]


def summarize_response(record: dict, limit: int = 80) -> str:
    raw_response = record.get("raw_response", "")
    if not raw_response:
        return ""
    compact = " ".join(raw_response.split())
    return compact[:limit] + ("..." if len(compact) > limit else "")


def visualize_errors(
    errors: list[dict],
    dataset: SNLIVEDataset,
    method: str,
    num_examples: int,
    output_dir: Path,
    seed: int = 42,
) -> None:
    id2sample = {sample["image_id"]: sample for sample in (dataset[i] for i in range(len(dataset)))}

    sampled = random.Random(seed).sample(errors, min(num_examples, len(errors)))

    cols = min(4, len(sampled))
    rows = (len(sampled) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 5.5))
    axes = [axes] if rows * cols == 1 else list(np.array(axes).flatten()) if rows > 1 else list(axes)

    for ax, err in zip(axes, sampled):
        sample = id2sample.get(err["image_id"])
        if sample is None or sample["image"] is None:
            ax.axis("off")
            continue

        ax.imshow(sample["image"])
        ax.axis("off")

        gold = err["gold_label"]
        pred = err["pred_label"]
        hypothesis = err["hypothesis"][:50]
        response_summary = summarize_response(err)
        title = f"真实: {gold}  预测: {pred}\nH: {hypothesis}"
        if response_summary:
            title += f"\n输出: {response_summary}"
        ax.set_title(title, fontsize=8, color=LABEL_COLORS.get(pred, "gray"))

    for ax in axes[len(sampled):]:
        ax.axis("off")

    plt.suptitle(f"错误样本 - {method}", fontsize=13)
    plt.tight_layout()

    out_path = output_dir / f"errors_{method}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"错误分析图已保存: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="错误分析可视化")
    parser.add_argument("--pred_dir", type=str, default=str(PRED_DIR))
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--num_examples", type=int, default=4)
    parser.add_argument("--output", type=str, default=str(ANALYSIS_DIR))
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SNLIVEDataset(split=args.split, load_images=True)

    for pred_path in sorted(pred_dir.glob(f"*_{args.split}.json")):
        with open(pred_path, encoding="utf-8") as f:
            data = json.load(f)
        method = data.get("method", pred_path.stem)
        errors = [record for record in data.get("results", []) if record["pred_label"] != record["gold_label"]]

        print(f"\n{method}: {len(errors)} 个错误样本 / {len(data.get('results', []))} 总样本")
        if errors:
            visualize_errors(errors, dataset, method, args.num_examples, output_dir)


if __name__ == "__main__":
    main()
