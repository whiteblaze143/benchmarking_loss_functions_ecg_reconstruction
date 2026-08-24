import json


def convert_json_to_jsonl_in_chunks(input_file, output_file_base, chunk_size=50000):
    """
    该函数用于读取输入的 JSON 文件，并将其中每个 subject 的数据转换为指定格式，
    最终输出为 JSON Lines 格式（.jsonl），且每 chunk_size 条记录保存为一个文件。

    转换规则如下：
      - custom_id 使用 subject id；
      - user 消息中的 content 使用 past_medical_history 字段；
      - 系统消息内容为固定模板，不进行修改。

    参数：
      input_file (str): 输入 JSON 文件路径。
      output_file_base (str): 输出文件基名，最终生成的文件名将为 {output_file_base}_1.jsonl, {output_file_base}_2.jsonl, …。
      chunk_size (int): 每个输出文件中包含的记录数，默认设定为 50000。
    """
    # 读取输入 JSON 文件
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 将数据转换为列表（若需要对 subject id 排序，可在此处进行排序，例如按数值大小）
    items = list(data.items())
    # 如 subject id 为纯数字字符串，可按数值排序：
    # items.sort(key=lambda x: int(x[0]))

    total_items = len(items)
    file_count = 0

    # 分批次写入，每个文件包含 chunk_size 条记录
    for i in range(0, total_items, chunk_size):
        file_count += 1
        chunk_items = items[i : i + chunk_size]
        # 构造输出文件名，后缀按顺序排序
        output_file = f"{output_file_base}_{file_count}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f_out:
            for subject_id, record in chunk_items:
                past_medical_history = record.get("past_medical_history", "")
                entry = {
                    "custom_id": subject_id,
                    "method": "POST",
                    "url": "/v4/chat/completions",
                    "body": {
                        "model": "glm-4-flash",
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "Role: Medical Record Organizer\n"
                                    "Objective: Convert medical data fragments into grammatically correct English paragraphs with strict verbatim retention of all original terms, including potential typos, unknown abbreviations, and identifiers.\n\n"
                                    "Non-Negotiable Rules:\n\n"
                                    "100% verbatim fidelity:\n\n"
                                    'Retain all original terms (e.g., "HED" → "HED", never expand or correct).\n\n'
                                    "Preserve identifiers ___, symbols, and punctuation exactly as written.\n\n"
                                    'Never flag, question, or explain potential errors (e.g., "HED" remains "HED").\n\n'
                                    "Subject standardization:\n\n"
                                    'Replace "Pt" with "the patient" once per sentence (e.g., "Pt HED" → "The patient has HED").\n\n'
                                    'Retain all other abbreviations (e.g., "DM" → "DM", "SOB" → "SOB").\n\n'
                                    "Special cases:\n\n"
                                    'Inputs like "None", ".", "—", or standalone symbols → output verbatim.\n\n'
                                    'Sentences with no actionable data (e.g., "HED.") → "The patient has HED."\n\n'
                                    "Output Format:\n\n"
                                    "Single plain-text paragraph.\n\n"
                                    "No markdown, bullet points, or line breaks.\n\n"
                                    "Examples:\n\n"
                                    "Input: Pt HED.\n"
                                    "Output: The patient has HED.\n\n"
                                    "Input: None.\n"
                                    "Output: None.\n\n"
                                    "Input: Pt c/o SOB. HLD___.\n"
                                    "Output: The patient c/o SOB. HLD___.\n\n"
                                    "Input: HR.\n"
                                    "Output: HR.\n\n"
                                    "Key Clarifications:\n\n"
                                    'No error handling: Treat "HED" as a valid term even if unrecognized.\n\n'
                                    'No contextualization: Avoid phrases like "diagnosed with" unless explicitly stated.\n\n'
                                    'Grammar limited to subject standardization: Only modify "Pt" → "the patient"; all other terms remain untouched.'
                                ),
                            },
                            {"role": "user", "content": past_medical_history},
                        ],
                        "temperature": 0.1,
                    },
                }
                # 将当前 entry 序列化为 JSON 字符串，并写入文件，每个 JSON 对象占一行
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # 定义输入与输出文件路径
    input_file = "experiments/debug/files/subject_info.json"
    # 输出文件基名，最终将生成诸如 batch_api_subject_info_1.jsonl, batch_api_subject_info_2.jsonl, … 文件
    output_file_base = "experiments/debug/files/batch_api_subject_info.jsonl"
    convert_json_to_jsonl_in_chunks(input_file, output_file_base, chunk_size=50000)
