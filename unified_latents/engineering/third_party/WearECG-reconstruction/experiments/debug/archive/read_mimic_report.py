import pandas as pd

# 文件路径
csv_path = "/data/0shared/jinjiarui/run/MLA-Diffusion/experiments/debug/files/mimic_ecg_with_report_filtered.csv"

# 读取 CSV 文件
df = pd.read_csv(csv_path, dtype=str)

pd.set_option("display.max_rows", None)  # 显示所有行
pd.set_option("display.max_columns", None)  # 显示所有列
pd.set_option("display.max_colwidth", None)  # 不截断列内容

print(df.columns)

# 打印前30行
print(df.head(10))
