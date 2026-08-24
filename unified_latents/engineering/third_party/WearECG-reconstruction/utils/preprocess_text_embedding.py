import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from tqdm import tqdm
import os

project_root_dir = "/data1/1shared/jinjiarui/run/MLA-Diffusion"
os.chdir(project_root_dir)
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# 配置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 初始化模型和分词器
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, trust_remote_code=True, cache_dir="models/text_encoder")
model = AutoModel.from_pretrained(
    MODEL_NAME, trust_remote_code=True, cache_dir="models/text_encoder").to(device)
model.eval()


def process_json_file(input_path, batch_size=32):
    """处理单个JSON文件生成[CLS]特征"""
    with open(input_path, 'r') as f:
        data = json.load(f)
    reports = [item["report"] for item in data]

    # 分批处理
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(reports), batch_size), desc=f"Processing {Path(input_path).name}"):
            batch_texts = reports[i:i+batch_size]

            encoded = tokenizer(
                text=batch_texts,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=2048
            ).to(device)

            outputs = model(**encoded)
            text_embs = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(text_embs.cpu().numpy())

    output_path = Path(input_path)
    output_name = output_path.stem.replace("reports", "text_embed") + ".npy"
    np.save(output_path.parent / output_name, np.concatenate(all_embeddings))
    print(f"Embeddings saved to {output_path.parent / output_name}")


files_to_process = [
    "/data1/1shared/jinjiarui/run/MLA-Diffusion/data/datasets/MIMIC/test_reports_0.json",
    "/data1/1shared/jinjiarui/run/MLA-Diffusion/data/datasets/MIMIC/train_reports_0.json",
    "/data1/1shared/jinjiarui/run/MLA-Diffusion/data/datasets/MIMIC/train_reports_1.json"
]

for file_path in files_to_process:
    process_json_file(file_path)

print("All processing completed!")
