import httpx
import os
from loguru import logger

class OCRService:
    def __init__(self):
        # 指向你刚才写的 OCR 服务端口
        self.ocr_api_url = "http://localhost:8010/ocr" 

    async def file_to_markdown(self, file_path: str):
        """
        调用独立部署的 DeepSeek OCR 服务
        返回: (markdown_content, md_file_path)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件未找到: {file_path}")

        logger.info(f"📤 [OCR] 发送文件至 DeepSeek 服务 (Port 8010): {file_path}")
        
        # OCR 比较慢，设置 5 分钟超时
        timeout = httpx.Timeout(300.0, connect=10.0) 
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                files = {'file': open(file_path, 'rb')}
                resp = await client.post(self.ocr_api_url, files=files)
                
                if resp.status_code == 200:
                    result = resp.json()
                    
                    if result.get("code") != 200:
                        raise Exception(f"OCR 内部错误: {result}")

                    md_file_path = result.get("md_file_path")
                    
                    # 关键步骤：OCR 服务已经把文件写到了磁盘上，我们直接读取它
                    if os.path.exists(md_file_path):
                        with open(md_file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        logger.success(f"✅ [OCR] 转换成功，读取到 Markdown 文件: {md_file_path}")
                        return content, md_file_path
                    else:
                        raise FileNotFoundError(f"OCR 声称生成了文件但未找到: {md_file_path}")

                else:
                    logger.error(f"❌ [OCR] 服务报错: {resp.text}")
                    raise Exception(f"OCR HTTP Error: {resp.status_code}")
                    
            except Exception as e:
                logger.error(f"❌ [OCR] 调用失败: {e}")
                raise e

ocr_service = OCRService()