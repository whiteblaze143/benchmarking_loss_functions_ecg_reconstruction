import json
import logging
from logging.handlers import RotatingFileHandler
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures
import sys
from datetime import datetime


# 配置日志系统
def setup_logger():
    # 创建日志对象
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 统一日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 文件日志（自动轮转，最大5个文件，每个10MB）
    file_handler = RotatingFileHandler(
        "experiments/debug/files/processing.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    # 控制台日志
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # 添加过滤器，屏蔽包含 "HTTP Request" 的日志
    def http_request_filter(record):
        return "HTTP Request" not in record.getMessage()

    file_handler.addFilter(http_request_filter)
    console_handler.addFilter(http_request_filter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# 初始化日志
logger = setup_logger()


def log_summary(results):
    """记录处理结果统计"""
    total = len(results)
    success = sum(
        1 for v in results.values() if v["past_medical_history_formatted"] is not None
    )
    failed = total - success

    logger.info("\n处理结果汇总:")
    logger.info(f"总处理数量: {total}")
    logger.info(f"成功数量: {success}")
    logger.info(f"失败数量: {failed}")
    if failed > 0:
        logger.warning("以下ID处理失败:")
        for sid, data in results.items():
            if data["past_medical_history_formatted"] is None and data.get(
                "past_medical_history"
            ):
                logger.warning(f"失败ID: {sid}")


# 读取 JSON 文件
input_file = "experiments/debug/files/subject_info.json"
logger.info(f"开始读取输入文件: {input_file}")
with open(input_file, "r", encoding="utf-8") as f:
    subject_info = json.load(f)
logger.info(f"成功加载 {len(subject_info)} 条患者数据")

# 用于存储处理后的结果
updated_results = {}


def process_subject(subject_id, record):
    """处理单个患者的函数"""
    logger.debug(f"开始处理患者ID: {subject_id}")
    pmh = record.get("past_medical_history")

    # 如果没有过往病史，直接返回
    if not pmh:
        logger.debug(f"患者ID {subject_id} 无病史记录，跳过处理")
        return subject_id, record

    # 构造请求提示
    prompt = f"Please organize the following medical record into a paragraph in English, where ___ represents a de-identified identifier that should be preserved. Do not explain any abbreviations. \n{pmh}"
    logger.debug(f"患者ID {subject_id} 构造提示完成")

    try:
        # 每个线程使用独立的客户端实例
        client = OpenAI(
            api_key="sk-e1c7a1ba4f784283a7e93d5d8325b1b5",
            base_url="https://api.deepseek.com",
        )

        logger.debug(f"患者ID {subject_id} 开始API请求")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            stream=False,
        )

        # 提取格式化后的文本
        if response.choices and response.choices[0].message.content:
            formatted_text = response.choices[0].message.content.strip()
            record["past_medical_history_formatted"] = formatted_text
            logger.info(f"患者ID {subject_id} 处理成功")
        else:
            record["past_medical_history_formatted"] = None
            logger.warning(f"患者ID {subject_id} 返回空响应")

    except Exception as e:
        record["past_medical_history_formatted"] = None
        logger.error(f"患者ID {subject_id} 处理失败 - {str(e)}", exc_info=True)
        # 仅记录错误堆栈前500个字符
        logger.debug(f"完整错误信息: {str(e)[:500]}")

    return subject_id, record


# 使用线程池并发处理
try:
    logger.info("开始并发处理...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        futures = []

        # 提交所有任务到线程池
        for subject_id in subject_info:
            futures.append(
                executor.submit(process_subject, subject_id, subject_info[subject_id])
            )

        # 使用进度条跟踪处理进度
        progress_bar = tqdm(total=len(futures), desc="Processing subjects")

        # 获取并处理完成的任务
        for future in concurrent.futures.as_completed(futures):
            subject_id, processed_record = future.result()
            updated_results[subject_id] = processed_record
            progress_bar.update(1)
        progress_bar.close()

except Exception as e:
    logger.critical(f"主程序异常终止: {str(e)}", exc_info=True)
    sys.exit(1)

# 保存处理结果
try:
    logger.info("开始保存处理结果...")
    output_file = "experiments/debug/files/subject_info_formatted.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(updated_results, f, ensure_ascii=False, indent=4)
    logger.info(f"结果已保存到: {output_file}")

except Exception as e:
    logger.error(f"保存结果失败: {str(e)}", exc_info=True)

# 记录处理摘要
log_summary(updated_results)
logger.info("处理流程完成")
