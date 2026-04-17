"""
对比各方法的预测结果，输出准确率表格与混淆矩阵。
用法: python -m src.analysis.compare
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.utils.config import ANALYSIS_DIR, LABELS, PRED_DIR


def load_results(pred_dir: Path) -> dict[str, dict]:
    """加载所有预测结果文件，返回 {method: data} 字典。"""
    results = {}
    for json_path in sorted(pred_dir.glob("*.json")):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        method = data.get("method", json_path.stem)
        results[method] = data
    return results


def compute_confusion_matrix(results: list[dict]) -> np.ndarray:
    """计算混淆矩阵，行=真实标签，列=预测标签。"""
    n = len(LABELS)
    label2id = {label: i for i, label in enumerate(LABELS)}
    matrix = np.zeros((n, n), dtype=int)

    for record in results:
        gold_id = label2id.get(record["gold_label"], 1)
        pred_id = label2id.get(record["pred_label"], 1)
        matrix[gold_id][pred_id] += 1

    return matrix


def per_class_accuracy(matrix: np.ndarray) -> list[float]:
    """计算每类的准确率（召回率）。"""
    return [matrix[i, i] / matrix[i].sum() if matrix[i].sum() > 0 else 0.0 for i in range(len(LABELS))]


def compute_prediction_distribution(results: list[dict]) -> dict[str, int]:
    distribution = {label: 0 for label in LABELS}
    for record in results:
        distribution[record["pred_label"]] = distribution.get(record["pred_label"], 0) + 1
    return distribution


def plot_confusion_matrix(matrix: np.ndarray, method: str, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=ax,
    )
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title(f"混淆矩阵 - {method}")
    plt.tight_layout()

    out_path = output_dir / f"confusion_{method}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"混淆矩阵已保存: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="对比各方法的预测结果")
    parser.add_argument("--pred_dir", type=str, default=str(PRED_DIR))
    parser.add_argument("--output", type=str, default=str(ANALYSIS_DIR))
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = load_results(pred_dir)
    if not all_results:
        print(f"未找到预测结果，请先运行各评估脚本。(目录: {pred_dir})")
        return

    print("\n" + "=" * 100)
    print(f"{'方法':<24} {'总准确率':>10} {'entail':>10} {'neutral':>10} {'contra':>10} {'预测分布':>30}")
    print("=" * 100)

    summary = []
    for method, data in all_results.items():
        records = data.get("results", [])
        if not records:
            continue

        overall_acc = data.get("accuracy", 0.0)
        matrix = compute_confusion_matrix(records)
        class_acc = per_class_accuracy(matrix)
        distribution = data.get("prediction_distribution", compute_prediction_distribution(records))

        print(
            f"{method:<24} {overall_acc:>10.2%} {class_acc[0]:>10.2%} {class_acc[1]:>10.2%} {class_acc[2]:>10.2%} "
            f"{str(distribution):>30}"
        )
        summary.append({
            "method": method,
            "accuracy": overall_acc,
            "per_class": class_acc,
            "prediction_distribution": distribution,
        })

        plot_confusion_matrix(matrix, method, output_dir)

    print("=" * 100)

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总结果已保存: {summary_path}")


if __name__ == "__main__":
    main()
