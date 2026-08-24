import pandas as pd

# 读取病史数据（唯一subject_id）
info_path = "/data/0shared/jinjiarui/run/MLA-Diffusion/experiments/debug/files/merged_subject_info_with_error.csv"
df_info = pd.read_csv(
    info_path,
    usecols=["subject_id", "past_medical_history_format"],
    dtype={"subject_id": str},
).drop_duplicates(
    "subject_id"
)  # 确保每个subject_id唯一

# 读取ECG记录数据（允许重复subject_id）
record_path = "/data/0shared/jinjiarui/datasets/physionet.org/files/mimic-iv-ecg/1.0/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/record_list.csv"
df_record = pd.read_csv(record_path, dtype={"subject_id": str})

# 左连接合并数据（保留所有ECG记录，包括NaN）
merged_df = df_record.merge(df_info, on="subject_id", how="left")

# 仅保留有病史的记录
filtered_df = merged_df.dropna(subset=["past_medical_history_format"])

# 验证输出
print(f"总ECG记录数（包含NaN）: {len(merged_df)}")
print(f"匹配到病史的记录数（无NaN）: {len(filtered_df)}")

# 保存两份结果
output_path_all = "experiments/debug/files/mimic_ecg_with_report.csv"
output_path_filtered = "experiments/debug/files/mimic_ecg_with_report_filtered.csv"

merged_df.to_csv(output_path_all, index=False)
filtered_df.to_csv(output_path_filtered, index=False)

print(f"\n包含所有ECG记录的结果已保存至：{output_path_all}")
print(f"仅包含有病史的结果已保存至：{output_path_filtered}")
