import re
import pandas as pd
import json
from tqdm import tqdm

# 1. 读取数据
file_path = "/data/0shared/jinjiarui/datasets/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv"
df = pd.read_csv(file_path)

# 2. 初始化结果字典
results = {}

# 3. 获取所有唯一的 subject_id
subject_ids = df["subject_id"].unique()

# 4. 处理每个 subject_id，并显示进度条
for subject_id in tqdm(subject_ids, desc="Processing subjects"):
    df_subject = df[df["subject_id"] == subject_id]
    min_row = df_subject.loc[df_subject["note_seq"].idxmin()]
    text_content = min_row["text"]

    # 提取性别
    match_sex = re.search(r"Sex:\s*(\S+)", text_content)
    sex = match_sex.group(1) if match_sex else None

    # 提取过往病史
    match_pmh = re.search(
        r"Past Medical History:\s*(.*?)(?:\n\s*\n|$)", text_content, re.DOTALL
    )
    past_medical_history = match_pmh.group(1).strip() if match_pmh else None

    # 过滤无效的病史内容
    invalid_keywords = {"none", "none.", "unknown", "unknown,", ".", ""}
    if past_medical_history:
        pmh_lower = past_medical_history.lower()
        if pmh_lower in invalid_keywords:
            past_medical_history = None

    # 如果过往病史为空或无效，则跳过该条记录，不保存此 subject 的数据
    if past_medical_history is None or past_medical_history == "":
        continue

    # 构建结果字典
    subject_entry = {"sex": sex, "past_medical_history": past_medical_history}
    results[int(subject_id)] = subject_entry

# 5. 保存为 JSON
output_file = "experiments/debug/files/subject_info.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"结果已保存到: {output_file}")
