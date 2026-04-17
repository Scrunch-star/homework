"""
CLIP zero-shot 评估：基于图像-文本相似度判断蕴含关系。

策略：
1. 计算 image 与 hypothesis 的相似度 sim_h
2. 计算 image 与 premise 的相似度 sim_p（作为基准）
3. 根据相似度差异判断：
   - sim_h > sim_p + threshold_high  → entailment
   - sim_h < sim_p - threshold_low   → contradiction
   - 其他                            → neutral
"""
import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from huggingface_hub import snapshot_download

from src.data.dataset import SNLIVEDataset
from src.utils.config import LABELS, PRED_DIR, PROJECT_ROOT


def prepare_local_model(model_name: str, model_cache_dir: Path) -> Path:
    model_dir = model_cache_dir / model_name.replace("/", "--")
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_name,
        repo_type="model",
        local_dir=str(model_dir),
    )
    return model_dir


class CLIPEvaluator:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        model_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"加载 CLIP 模型: {model_name} (device: {self.device})")
        model_source = str(model_dir) if model_dir else model_name

        self.processor = CLIPProcessor.from_pretrained(model_source, local_files_only=bool(model_dir))
        self.model = CLIPModel.from_pretrained(model_source, local_files_only=bool(model_dir)).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def compute_similarity(self, image: Image.Image, text: str) -> float:
        """计算单张图像与单个文本的相似度。"""
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)
        return outputs.logits_per_image[0, 0].item()

    def score_sample(self, image: Image.Image, premise: str, hypothesis: str) -> dict:
        sim_h = self.compute_similarity(image, hypothesis)
        sim_p = self.compute_similarity(image, premise)
        diff = sim_h - sim_p
        return {
            "sim_hypothesis": sim_h,
            "sim_premise": sim_p,
            "diff": diff,
        }

    def predict_from_diff(
        self,
        diff: float,
        threshold_high: float = 0.05,
        threshold_low: float = 0.05,
    ) -> str:
        if diff > threshold_high:
            return "entailment"
        if diff < -threshold_low:
            return "contradiction"
        return "neutral"

    def evaluate_dataset(
        self,
        dataset: SNLIVEDataset,
        batch_size: int = 32,
        threshold_high: float = 0.05,
        threshold_low: float = 0.05,
    ) -> list[dict]:
        """批量评估数据集。"""
        results = []

        for sample in tqdm(dataset, desc="CLIP 推理"):
            scores = self.score_sample(sample["image"], sample["premise"], sample["hypothesis"])
            pred_label = self.predict_from_diff(scores["diff"], threshold_high, threshold_low)
            results.append({
                "image_id": sample["image_id"],
                "premise": sample["premise"],
                "hypothesis": sample["hypothesis"],
                "gold_label": sample["label"],
                "pred_label": pred_label,
                **scores,
            })

        return results


def compute_accuracy(results: list[dict]) -> float:
    """计算准确率。"""
    correct = sum(1 for r in results if r["pred_label"] == r["gold_label"])
    return correct / len(results) if results else 0.0


def compute_label_distribution(results: list[dict]) -> dict[str, int]:
    distribution = {label: 0 for label in LABELS}
    for result in results:
        distribution[result["pred_label"]] = distribution.get(result["pred_label"], 0) + 1
    return distribution


def search_thresholds(evaluator: CLIPEvaluator, dataset: SNLIVEDataset, grid_values: list[float]) -> dict:
    base_results = []
    for sample in tqdm(dataset, desc="CLIP 阈值搜索打分"):
        scores = evaluator.score_sample(sample["image"], sample["premise"], sample["hypothesis"])
        base_results.append({
            "gold_label": sample["label"],
            "diff": scores["diff"],
        })

    best = {
        "accuracy": -1.0,
        "threshold_high": None,
        "threshold_low": None,
        "distribution": {},
    }
    search_results = []

    for threshold_high in grid_values:
        for threshold_low in grid_values:
            trial = []
            for item in base_results:
                pred = evaluator.predict_from_diff(item["diff"], threshold_high, threshold_low)
                trial.append({"gold_label": item["gold_label"], "pred_label": pred})
            accuracy = compute_accuracy(trial)
            distribution = compute_label_distribution(trial)
            result = {
                "threshold_high": threshold_high,
                "threshold_low": threshold_low,
                "accuracy": accuracy,
                "distribution": distribution,
            }
            search_results.append(result)
            if accuracy > best["accuracy"]:
                best = result

    return {"best": best, "trials": search_results}


def main() -> None:
    parser = argparse.ArgumentParser(description="CLIP zero-shot 评估")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threshold_high", type=float, default=0.05)
    parser.add_argument("--threshold_low", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--search_thresholds", action="store_true")
    parser.add_argument("--search_values", type=str, default="0.01,0.03,0.05,0.07,0.1")
    parser.add_argument("--model_cache_dir", type=str, default=str(PROJECT_ROOT / "data" / "hf-models"))
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    dataset = SNLIVEDataset(
        split=args.split,
        load_images=True,
        max_samples=args.max_samples,
    )
    print(f"数据集大小: {len(dataset)}")

    model_name = "openai/clip-vit-base-patch32"
    model_dir = prepare_local_model(model_name, Path(args.model_cache_dir))
    evaluator = CLIPEvaluator(model_name=model_name, model_dir=model_dir)

    output_path = Path(args.output) if args.output else PRED_DIR / f"clip_{args.split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.search_thresholds:
        grid_values = [float(value.strip()) for value in args.search_values.split(",") if value.strip()]
        search_summary = search_thresholds(evaluator, dataset, grid_values)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "method": "clip_threshold_search",
                "split": args.split,
                "max_samples": args.max_samples,
                "search_values": grid_values,
                **search_summary,
            }, f, ensure_ascii=False, indent=2)
        print(f"最佳阈值: high={search_summary['best']['threshold_high']}, low={search_summary['best']['threshold_low']}")
        print(f"最佳准确率: {search_summary['best']['accuracy']:.2%}")
        print(f"搜索结果已保存: {output_path}")
        return

    results = evaluator.evaluate_dataset(
        dataset,
        args.batch_size,
        args.threshold_high,
        args.threshold_low,
    )

    accuracy = compute_accuracy(results)
    distribution = compute_label_distribution(results)
    print(f"\n准确率: {accuracy:.2%}")
    print(f"预测分布: {distribution}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "clip",
            "split": args.split,
            "accuracy": accuracy,
            "threshold_high": args.threshold_high,
            "threshold_low": args.threshold_low,
            "prediction_distribution": distribution,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"结果已保存: {output_path}")


if __name__ == "__main__":
    main()
