"""
BLIP-2 / LLaVA VQA 风格评估。

策略：构造 prompt 让模型直接回答 A/B/C 三选一：
  A = entailment
  B = neutral
  C = contradiction
"""
import argparse
import json
import re
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm

from src.data.dataset import SNLIVEDataset
from src.utils.config import PRED_DIR, PROJECT_ROOT

# ==================== Prompt 模板 ====================
PROMPT_TEMPLATE = (
    "You are solving a 3-way visual entailment task.\n\n"
    "Label definitions:\n"
    "- entailment: the image and premise clearly support the hypothesis.\n"
    "- neutral: the hypothesis adds details that are not clearly supported.\n"
    "- contradiction: the hypothesis conflicts with the image or premise.\n\n"
    "Examples:\n"
    "Premise: 'A child sits in front of a computer.'\n"
    "Hypothesis: 'A child is near a computer.'\n"
    "Answer: entailment\n\n"
    "Premise: 'A child sits in front of a computer.'\n"
    "Hypothesis: 'The child loves the computer.'\n"
    "Answer: neutral\n\n"
    "Premise: 'A child sits in front of a computer.'\n"
    "Hypothesis: 'An old man is outdoors.'\n"
    "Answer: contradiction\n\n"
    "Now solve this example.\n"
    'Premise: "{premise}"\n'
    'Hypothesis: "{hypothesis}"\n\n'
    "Return exactly one word: entailment, neutral, or contradiction."
)

# 答案到标签的映射
ANSWER2LABEL = {"A": "entailment", "B": "neutral", "C": "contradiction"}
WORD2LABEL = {
    "ENTAILMENT": "entailment",
    "NEUTRAL": "neutral",
    "CONTRADICTION": "contradiction",
}


def parse_response(response: str) -> str:
    """从模型生成文本中提取标签，失败时回退到 neutral。"""
    cleaned = response.strip()
    if not cleaned:
        return "neutral"

    first_line = cleaned.splitlines()[0].strip().upper()
    word_match = re.search(r"\b(ENTAILMENT|NEUTRAL|CONTRADICTION)\b", first_line)
    if word_match:
        return WORD2LABEL[word_match.group(1)]

    patterns = [
        r"^(?:ANSWER\s*[:：-]?\s*)?\(?([ABC])\)?\b",
        r"^(?:LABEL\s*[:：-]?\s*)?\(?([ABC])\)?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, first_line)
        if match:
            return ANSWER2LABEL[match.group(1)]

    fallback_word_match = re.search(r"\b(ENTAILMENT|NEUTRAL|CONTRADICTION)\b", cleaned.upper())
    if fallback_word_match:
        return WORD2LABEL[fallback_word_match.group(1)]

    fallback_match = re.search(r"\b([ABC])\b", cleaned.upper())
    if fallback_match:
        return ANSWER2LABEL[fallback_match.group(1)]

    return "neutral"


def prepare_local_model(model_name: str, model_cache_dir: Path) -> Path:
    model_dir = model_cache_dir / model_name.replace("/", "--")
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_name,
        repo_type="model",
        local_dir=str(model_dir),
    )
    return model_dir


def decode_generated_text(processor, output_ids: torch.Tensor, input_ids: Optional[torch.Tensor]) -> str:
    """只解码模型新生成的 token，避免把输入 prompt 一起解码。"""
    generated_ids = output_ids
    if input_ids is not None and input_ids.shape[1] < output_ids.shape[1]:
        generated_ids = output_ids[:, input_ids.shape[1]:]
    return processor.decode(generated_ids[0], skip_special_tokens=True).strip()


# ==================== BLIP-2 ====================
class BLIP2Evaluator:
    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        model_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"加载 BLIP-2: {model_name} (device: {self.device})")
        model_source = str(model_dir) if model_dir else model_name

        self.processor = Blip2Processor.from_pretrained(model_source, local_files_only=bool(model_dir))
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_source,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            local_files_only=bool(model_dir),
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_label(self, image: Image.Image, premise: str, hypothesis: str) -> tuple[str, str]:
        prompt = PROMPT_TEMPLATE.format(premise=premise, hypothesis=hypothesis)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )
        response = decode_generated_text(self.processor, output_ids, inputs.get("input_ids"))
        return parse_response(response), response


# ==================== LLaVA ====================
class LLaVAEvaluator:
    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        model_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"加载 LLaVA: {model_name} (device: {self.device})")
        model_source = str(model_dir) if model_dir else model_name

        self.processor = AutoProcessor.from_pretrained(model_source, local_files_only=bool(model_dir))
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_source,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            local_files_only=bool(model_dir),
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_label(self, image: Image.Image, premise: str, hypothesis: str) -> tuple[str, str]:
        prompt = (
            "USER: <image>\n"
            f"{PROMPT_TEMPLATE.format(premise=premise, hypothesis=hypothesis)}\n"
            "ASSISTANT:"
        )
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )
        response = decode_generated_text(self.processor, output_ids, inputs.get("input_ids"))
        return parse_response(response), response


def build_evaluator(model_type: str, model_cache_dir: Path):
    """工厂函数，根据模型类型返回对应的 Evaluator。"""
    if model_type == "blip":
        model_name = "Salesforce/blip2-opt-2.7b"
        model_dir = prepare_local_model(model_name, model_cache_dir)
        return BLIP2Evaluator(model_name=model_name, model_dir=model_dir)
    if model_type == "llava":
        model_name = "llava-hf/llava-1.5-7b-hf"
        model_dir = prepare_local_model(model_name, model_cache_dir)
        return LLaVAEvaluator(model_name=model_name, model_dir=model_dir)
    raise ValueError(f"未知模型类型: {model_type}，支持: blip / llava")


def evaluate_dataset(evaluator, dataset: SNLIVEDataset) -> list[dict]:
    """批量评估数据集，返回预测结果列表。"""
    results = []
    for sample in tqdm(dataset, desc="LLM 推理"):
        pred_label, raw_response = evaluator.predict_label(
            sample["image"],
            sample["premise"],
            sample["hypothesis"],
        )
        results.append({
            "image_id": sample["image_id"],
            "premise": sample["premise"],
            "hypothesis": sample["hypothesis"],
            "gold_label": sample["label"],
            "pred_label": pred_label,
            "raw_response": raw_response,
        })
    return results


def compute_accuracy(results: list[dict]) -> float:
    correct = sum(1 for r in results if r["pred_label"] == r["gold_label"])
    return correct / len(results) if results else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="BLIP/LLaVA 评估")
    parser.add_argument("--model", type=str, default="blip", choices=["blip", "llava"])
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_cache_dir", type=str, default=str(PROJECT_ROOT / "data" / "hf-models"))
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    dataset = SNLIVEDataset(split=args.split, load_images=True, max_samples=args.max_samples)
    print(f"数据集大小: {len(dataset)}")

    evaluator = build_evaluator(args.model, Path(args.model_cache_dir))
    results = evaluate_dataset(evaluator, dataset)

    accuracy = compute_accuracy(results)
    print(f"\n准确率: {accuracy:.2%}")

    output_path = Path(args.output) if args.output else PRED_DIR / f"{args.model}_{args.split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": args.model,
            "split": args.split,
            "accuracy": accuracy,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"结果已保存: {output_path}")


if __name__ == "__main__":
    main()
