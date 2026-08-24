import pandas as pd

# 指定原始 CSV 文件路径
file_path = "/data1/1shared/jinjiarui/datasets/physionet.org/files/labelled-notes-hospital-course/1.2.0/mimic-iv-bhc.csv"
# 指定保存前100行数据的 CSV 文件路径
small_file_path = "mimic_iv_bhc_first20.csv"

# 读取 CSV 文件，生成 DataFrame 对象
df = pd.read_csv(file_path)

print(df)

# 提取前 100 行数据
df_first100 = df.head(20)
# 将提取的数据保存为新的 CSV 文件，不包含索引信息
df_first100.to_csv(small_file_path, index=False)

print("已成功保存前 20 行数据到:", small_file_path)
