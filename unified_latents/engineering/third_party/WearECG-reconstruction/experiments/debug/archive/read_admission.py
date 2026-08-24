import pandas as pd

# 读取原始 admissions.csv 文件
csv_path = "/data1/1shared/jinjiarui/datasets/physionet.org/files/mimiciv/3.1/hosp/admissions.csv"
df = pd.read_csv(csv_path)

# 获取前10行
df_head = df.head(10)

# 保存为新的 CSV 文件
output_path = "experiments/debug/files/admissions_head.csv"
df_head.to_csv(output_path, index=False)

print(f"已成功将前10行保存至：{output_path}")
