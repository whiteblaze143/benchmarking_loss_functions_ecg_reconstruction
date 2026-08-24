import json
import csv


def merge_jsonl_to_csv(subject_info_file, output_jsonl_files, csv_output_file):
    """
    该函数将 LLM 返回的 jsonl 文件进行解析，并根据其中的 request_id 在 subject_info.json 中查找对应记录，
    最终创建一个 CSV 文件。CSV 文件的格式如下：

      第一列：subject id
      第二列：sex
      第三列：past_medical_history（原始数据）
      第四列：past_medical_history format（LLM 格式化后的内容）

    参数：
      subject_info_file (str): subject_info.json 文件路径。
      output_jsonl_files (list): 包含多个 LLM 返回的 jsonl 文件路径的列表。
      csv_output_file (str): 最终输出的 CSV 文件路径。
    """
    # 读取 subject_info.json 文件
    with open(subject_info_file, "r", encoding="utf-8") as f:
        subject_info = json.load(f)

    records = []  # 用于存储最终待写入 CSV 的记录，每个元素为字典

    # 遍历所有输出的 jsonl 文件
    for jsonl_file in output_jsonl_files:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                # 提取 request_id 以及格式化后的过往病史（存储在 choices[0].message.content）
                request_id = data.get("response", {}).get("body", {}).get("request_id")
                choices = data.get("response", {}).get("body", {}).get("choices", [])
                if not request_id or not choices:
                    continue
                formatted_history = choices[0].get("message", {}).get("content", "")
                # print(formatted_history)
                # 根据 request_id 在 subject_info 中查找对应记录
                subject_record = subject_info.get(request_id, {})
                sex = subject_record.get("sex", "")
                past_medical_history = subject_record.get("past_medical_history", "")

                record = {
                    "subject_id": request_id,
                    "sex": sex,
                    "past_medical_history": past_medical_history,
                    "past_medical_history_format": formatted_history,
                }
                records.append(record)

    # 写入 CSV 文件，指定列顺序
    with open(csv_output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "subject_id",
            "sex",
            "past_medical_history",
            "past_medical_history_format",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    subject_info_file = "experiments/debug/files/subject_info.json"
    output_jsonl_files = [
        "experiments/debug/files/output_batch_1.jsonl",
        "experiments/debug/files/output_batch_2.jsonl",
        "experiments/debug/files/output_batch_3.jsonl",
  
    ]
    csv_output_file = "experiments/debug/files/merged_subject_info.csv"
    merge_jsonl_to_csv(subject_info_file, output_jsonl_files, csv_output_file)
