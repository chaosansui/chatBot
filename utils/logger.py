import sys
from loguru import logger
from core.config import settings

# 配置日志
def setup_logger():
    """配置日志系统"""
    logger.remove()  # 移除默认处理器
    
    # 添加控制台处理器
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # 可选：添加文件处理器
    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

# 初始化日志
setup_logger()