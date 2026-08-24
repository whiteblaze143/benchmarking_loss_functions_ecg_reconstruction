import pandas as pd

# 文件路径
error_info_path = "/data/0shared/jinjiarui/run/MLA-Diffusion/experiments/debug/files/error_subject_info_merged.csv"
merged_info_path = "/data/0shared/jinjiarui/run/MLA-Diffusion/experiments/debug/files/merged_subject_info.csv"
output_path = "/data/0shared/jinjiarui/run/MLA-Diffusion/experiments/debug/files/merged_subject_info_with_error.csv"

# 读取 CSV 文件时尝试使用制表符分隔
df_error = pd.read_csv(error_info_path, dtype=str, sep="\t")  # 指定分隔符
df_merged = pd.read_csv(merged_info_path, dtype=str)  # 指定分隔符

# 确保两者的列顺序一致
df_merged = df_merged[
    df_error.columns.tolist()
    + [col for col in df_merged.columns if col not in df_error.columns]
]

# 确保 error_subject_info_merged 的数据在前
df_final = pd.concat([df_error, df_merged], axis=0, ignore_index=True)


df_final.to_csv(output_path, index=False)

print(f"数据合并完成，已保存至: {output_path}")
