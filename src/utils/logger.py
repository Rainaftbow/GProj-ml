import logging
import os
from datetime import datetime

# 全局 logger 实例
logger = logging.getLogger("ml_module")

def setup_logging(log_dir="logs", log_level=logging.INFO):
    """配置日志系统"""
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"ml_module_{timestamp}.log")
    
    # 配置日志
    logger.setLevel(log_level)
    
    # 创建格式化器
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("日志系统已配置")
    return logger