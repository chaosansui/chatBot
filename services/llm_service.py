import time
import httpx
from loguru import logger
from typing import Dict, Any, List, Optional
from core.config import settings
from models.api_models import SimpleChatResponse

class LLMService:
    """LLM服务 - 专门适配vLLM的Qwen模型"""
    
    def __init__(self):
        self.model_url = settings.LOCAL_MODEL_URL
        self.timeout = settings.MODEL_TIMEOUT
        self.max_tokens = settings.MODEL_MAX_TOKENS
        self.temperature = settings.MODEL_TEMPERATURE
        
        # 创建HTTP客户端
        self.client = httpx.AsyncClient(timeout=self.timeout)
        self.is_ready = False
    
    async def health_check(self) -> bool:
        """检查模型服务是否就绪"""
        try:
            health_url = f"http://{settings.LOCAL_MODEL_HOST}:{settings.LOCAL_MODEL_PORT}/health"
            response = await self.client.get(health_url)
            self.is_ready = response.status_code == 200
            logger.info(f"🤖 模型服务状态: {'正常' if self.is_ready else '异常'}")
            return self.is_ready
        except Exception as e:
            logger.warning(f"模型服务健康检查失败: {e}")
            self.is_ready = False
            return False
    
    async def chat(
        self, 
        message: str, 
        session_id: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> SimpleChatResponse:
        """调用vLLM的Qwen模型进行聊天"""
        start_time = time.time()
        
        try:
            # 构建消息历史 - vLLM使用OpenAI格式
            messages = self._build_messages(message, conversation_history)
            
            # vLLM OpenAI兼容格式的请求体
            payload = {
                "model": "qwen",  # 模型名称，vLLM会忽略这个但需要提供
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False
            }
            
            logger.info(f"调用vLLM模型 - Session: {session_id}, 消息: {message[:50]}...")
            logger.debug(f"请求体: {payload}")
            
            # 调用vLLM API
            response = await self.client.post(
                self.model_url,
                json=payload,
                timeout=self.timeout
            )
            
            # 打印详细响应信息用于调试
            logger.debug(f"响应状态码: {response.status_code}")
            logger.debug(f"响应头: {dict(response.headers)}")
            
            response.raise_for_status()
            result = response.json()
            
            logger.debug(f"完整响应: {result}")
            
            # 解析vLLM响应
            answer = self._parse_vllm_response(result)
            processing_time = time.time() - start_time
            
            logger.info(f"✅ 模型响应成功 - 耗时: {processing_time:.2f}s, 字符数: {len(answer)}")
            
            return SimpleChatResponse(
                answer=answer,
                session_id=session_id,
                processing_time=processing_time
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP错误 - 状态码: {e.response.status_code}, 响应: {e.response.text}")
            return SimpleChatResponse(
                answer=f"模型服务返回错误: {e.response.status_code}",
                session_id=session_id,
                processing_time=time.time() - start_time
            )
        except httpx.TimeoutException:
            logger.error(f"❌ 模型调用超时 - 超时设置: {self.timeout}s")
            return SimpleChatResponse(
                answer="抱歉，模型响应超时，请稍后重试。",
                session_id=session_id,
                processing_time=time.time() - start_time
            )
        except httpx.ConnectError:
            logger.error(f"❌ 无法连接到模型服务: {self.model_url}")
            return SimpleChatResponse(
                answer="抱歉，模型服务暂时不可用，请检查服务状态。",
                session_id=session_id,
                processing_time=time.time() - start_time
            )
        except Exception as e:
            logger.error(f"❌ 模型调用失败: {e}")
            return SimpleChatResponse(
                answer="抱歉，服务处理出现异常，请稍后重试。",
                session_id=session_id,
                processing_time=time.time() - start_time
            )
    
    def _build_messages(
        self, 
        current_message: str, 
        conversation_history: Optional[List[Dict]] = None
    ) -> List[Dict[str, str]]:
        """构建OpenAI格式的消息历史"""
        messages = []
        
        # 添加系统消息（可选）
        system_message = {
            "role": "system",
            "content": "你是一个有用的AI助手。请用中文回答用户的问题。"
        }
        messages.append(system_message)
        
        # 添加历史消息（如果存在）
        if conversation_history:
            for msg in conversation_history[-6:]:  # 只保留最近6轮对话
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        # 添加当前消息
        messages.append({
            "role": "user",
            "content": current_message
        })
        
        return messages
    
    def _parse_vllm_response(self, result: Dict[str, Any]) -> str:
        """解析vLLM的响应格式"""
        try:
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"].strip()
            
            # 如果标准格式不匹配，尝试其他可能格式
            logger.warning(f"非标准响应格式: {result}")
            
            if "text" in result:
                return result["text"].strip()
            elif "generated_text" in result:
                return result["generated_text"].strip()
            else:
                # 返回原始响应用于调试
                return f"[调试] 响应格式异常: {str(result)[:200]}"
                
        except Exception as e:
            logger.error(f"解析模型响应失败: {e}, 原始响应: {result}")
            return "抱歉，模型返回了无法解析的响应。"
    
    async def close(self):
        """关闭HTTP客户端"""
        await self.client.aclose()

# 全局LLM服务实例
llm_service = LLMService()