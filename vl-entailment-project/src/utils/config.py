"""
项目全局配置。
"""
from pathlib import Path

# ==================== 路径配置 ====================
# 项目根目录（从本文件向上两级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "snli-ve"
IMAGE_DIR = DATA_DIR / "flickr30k-images"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PRED_DIR = OUTPUT_DIR / "predictions"
VIZ_DIR = OUTPUT_DIR / "visualizations"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"

# ==================== 标签配置 ====================
LABELS = ["entailment", "neutral", "contradiction"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
