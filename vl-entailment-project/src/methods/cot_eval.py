"""
Chain-of-Thought 推理：要求模型输出结构化 JSON，包含推理过程。

输出格式：
{
  "image_entities": ["..."],
  "statement_entities": ["..."],
  "conflict_detected": true/false,
  "reasoning": "...",
  "label": "A" / "B" / "C"
}
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

# ==================== CoT Prompt ====================
COT_PROMPT = (
    "You are solving a visual entailment task with conservative reasoning.\n\n"
    "Label definitions:\n"
    "- entailment: clearly supported by image and premise.\n"
    "- neutral: hypothesis adds unsupported details or cannot be confirmed.\n"
    "- contradiction: hypothesis conflicts with image or premise.\n\n"
    "Examples:\n"
    "Premise: 'A child sits in front of a computer.' Hypothesis: 'A child is near a computer.' Label: entailment\n"
    "Premise: 'A child sits in front of a computer.' Hypothesis: 'The child loves the computer.' Label: neutral\n"
    "Premise: 'A child sits in front of a computer.' Hypothesis: 'An old man is outdoors.' Label: contradiction\n\n"
    'Premise: "{premise}"\n'
    'Hypothesis: "{hypothesis}"\n\n'
    "Return valid JSON only, without markdown fences, using this schema:\n"
    "{{\n"
    '  "image_entities": ["..."],\n'
    '  "statement_entities": ["..."],\n'
    '  "conflict_detected": true,\n'
    '  "reasoning": "...",\n'
    '  "label": "neutral"\n'
    "}}"
)

ANSWER2LABEL = {"A": "entailment", "B": "neutral", "C": "contradiction"}
WORD2LABEL = {
    "A": "entailment",
    "B": "neutral",
    "C": "contradiction",
    "ENTAILMENT": "entailment",
    "NEUTRAL": "neutral",
    "CONTRADICTION": "contradiction",
}


def parse_cot_response(response: str) -> tuple[str, dict]:
    """解析 CoT 输出，返回 (label, parsed_json)。"""
    cleaned = response.strip()
    if not cleaned:
        return "neutral", {}

    normalized = cleaned.replace("\\_", "_")
    json_match = re.search(r"\{.*\}", normalized, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            label_raw = str(parsed.get("label", "B")).strip().upper()
            if label_raw:
                if label_raw[0] in ANSWER2LABEL:
                    return ANSWER2LABEL[label_raw[0]], parsed
                if label_raw in WORD2LABEL:
                    return WORD2LABEL[label_raw], parsed
            return "neutral", parsed
        except json.JSONDecodeError:
            pass

    first_line = normalized.splitlines()[0].strip().upper()
    match = re.search(r"^(?:LABEL\s*[:：-]?\s*)?\(?([ABC])\)?\b", first_line)
    if match:
        return ANSWER2LABEL[match.group(1)], {}

    word_match = re.search(r"\b(ENTAILMENT|NEUTRAL|CONTRADICTION)\b", first_line)
    if word_match:
        return WORD2LABEL[word_match.group(1)], {}

    label_match = re.search(r'"label"\s*:\s*"?([ABC])"?', normalized, re.IGNORECASE)
    if label_match:
        return ANSWER2LABEL[label_match.group(1).upper()], {}

    fallback_match = re.search(r"\b([ABC])\b", normalized.upper())
    if fallback_match:
        return ANSWER2LABEL[fallback_match.group(1)], {}

    return "neutral", {}


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


# ==================== LLaVA CoT ====================
class LLaVACotEvaluator:
    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        model_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"加载 LLaVA (CoT): {model_name} (device: {self.device})")
        model_source = str(model_dir) if model_dir else model_name

        self.processor = AutoProcessor.from_pretrained(model_source, local_files_only=bool(model_dir))
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_source,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            local_files_only=bool(model_dir),
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: Image.Image, premise: str, hypothesis: str) -> tuple[str, dict, str]:
        prompt = f"USER: <image>\n{COT_PROMPT.format(premise=premise, hypothesis=hypothesis)}\nASSISTANT:"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )
        response = decode_generated_text(self.processor, output_ids, inputs.get("input_ids"))
        label, parsed = parse_cot_response(response)
        return label, parsed, response


def evaluate_dataset(evaluator, dataset: SNLIVEDataset) -> list[dict]:
    results = []
    for sample in tqdm(dataset, desc="CoT 推理"):
        pred_label, cot_output, raw_response = evaluator.predict(
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
            "cot": cot_output,
            "raw_response": raw_response,
        })
    return results


def compute_accuracy(results: list[dict]) -> float:
    correct = sum(1 for r in results if r["pred_label"] == r["gold_label"])
    return correct / len(results) if results else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Chain-of-Thought 推理评估")
    parser.add_argument("--model", type=str, default="llava", choices=["llava"])
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_cache_dir", type=str, default=str(PROJECT_ROOT / "data" / "hf-models"))
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    dataset = SNLIVEDataset(split=args.split, load_images=True, max_samples=args.max_samples)
    print(f"数据集大小: {len(dataset)}")

    model_name = "llava-hf/llava-1.5-7b-hf"
    model_dir = prepare_local_model(model_name, Path(args.model_cache_dir))
    evaluator = LLaVACotEvaluator(model_name=model_name, model_dir=model_dir)
    results = evaluate_dataset(evaluator, dataset)

    accuracy = compute_accuracy(results)
    print(f"\n准确率: {accuracy:.2%}")

    output_path = Path(args.output) if args.output else PRED_DIR / f"cot_{args.model}_{args.split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": f"cot_{args.model}",
            "split": args.split,
            "accuracy": accuracy,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"结果已保存: {output_path}")


if __name__ == "__main__":
    main()
