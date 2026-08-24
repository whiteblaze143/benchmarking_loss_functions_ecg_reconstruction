import pandas as pd
from pathlib import Path
import wfdb
from tqdm import tqdm

# 1. 读取 CSV 文件
record_list_path = "/data1/1shared/jinjiarui/datasets/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/record_list.csv"
admission_path = "/data1/1shared/jinjiarui/datasets/physionet.org/files/mimiciv/3.1/hosp/admissions.csv"

df_records = pd.read_csv(record_list_path)
df_adm = pd.read_csv(admission_path, parse_dates=["admittime", "dischtime"])

# 2. 匹配 ECG 到住院记录
matched_records = []

base_dir = "/data1/1shared/jinjiarui/datasets/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"

for idx, row in tqdm(df_records.iterrows(), total=len(df_records)):
    subject_id = row["subject_id"]
    study_id = row["study_id"]
    ecg_path = row["path"]
    hea_file = Path(base_dir) / f"{ecg_path}"

    try:
        header = wfdb.rdheader(str(hea_file))
        base_date = header.base_date
        base_time = header.base_time

        if not base_date or not base_time:
            continue  # 跳过时间缺失记录

        ecg_datetime = pd.to_datetime(f"{base_date} {base_time}")
    except Exception as e:
        print(f"无法读取 {hea_file}: {e}")
        continue

    # 找到对应 subject_id 的所有住院记录
    adm_sub = df_adm[df_adm["subject_id"] == subject_id]
    # print(f"admittime time: {adm_sub['admittime']}  dischtime: {adm_sub['dischtime']}")
    # print(f"ecg_datetime: {ecg_datetime}")
    matched = adm_sub[
        (adm_sub["admittime"] <= ecg_datetime) & (adm_sub["dischtime"] >= ecg_datetime)
    ]
    # print(matched)
    if not matched.empty:
        for _, adm_row in matched.iterrows():
            matched_records.append(
                {
                    "subject_id": subject_id,
                    "hadm_id": adm_row["hadm_id"],
                    "study_id": study_id,
                    "ecg_datetime": ecg_datetime,
                    "admittime": adm_row["admittime"],
                    "dischtime": adm_row["dischtime"],
                }
            )

# 3. 保存匹配结果
df_matched = pd.DataFrame(matched_records)
output_path = "experiments/debug/files/ecg_to_admission_mapping.csv"
df_matched.to_csv(output_path, index=False)

print(f"完成匹配，共找到 {len(df_matched)} 条匹配记录，结果已保存至: {output_path}")
