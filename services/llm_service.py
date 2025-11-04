import time
import json
import httpx
from loguru import logger
from typing import Dict, Any, List, Optional, AsyncIterator
from core.config import settings
from models.api_models import SimpleChatResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult



class VLLMChatModel(BaseChatModel):

    llm_service: Any # 接收 LLMService 实例，用于调用 chat_stream

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        """同步生成方法，不推荐在流式应用中使用。"""
        # 强制要求使用异步流式方法
        raise NotImplementedError("请使用异步方法 _agenerate 或流式方法 _astream")

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """异步流式生成方法，调用底层 LLMService 的流式 API。"""
        
        # 提取用户消息和历史记录，适应 LLMService 的 _build_messages 格式
        current_message = messages[-1].content if messages else ""
        conversation_history = [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in messages[:-1]
        ]
        
        # 调用 LLMService 的流式生成器
        stream_generator = self.llm_service.chat_stream(
            message=current_message,
            conversation_history=conversation_history,
            session_id=kwargs.get("session_id") 
        )
        
        # 迭代 LLM 产生的增量，并包装成 LangChain Chunk
        async for chunk_content in stream_generator:
            # 创建 ChatGenerationChunk
            chunk = ChatGenerationChunk(
                text=chunk_content,
                message=AIMessage(content=chunk_content),
                # 可选：如果需要 LangChain Callback，可以在这里调用 run_manager.on_llm_new_token(chunk_content)
            )
            yield chunk

    @property
    def _llm_type(self) -> str:
        return "vllm-qwen-chat-model"

# ----------------------------------------------------
# 2. 核心 LLMService 类
# ----------------------------------------------------
class LLMService:
    """LLM服务 - 专门适配vLLM的Qwen模型"""
    
    def __init__(self):
        self.model_url = settings.LOCAL_MODEL_URL
        self.timeout = settings.MODEL_TIMEOUT
        self.max_tokens = settings.MODEL_MAX_TOKENS
        self.temperature = settings.MODEL_TEMPERATURE
        self.client = httpx.AsyncClient(timeout=self.timeout)
        self.is_ready = False
        self.langchain_llm = VLLMChatModel(llm_service=self)
    
    async def health_check(self) -> bool:
        
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
        
    async def chat(self, message: str, session_id: Optional[str] = None, conversation_history: Optional[List[Dict]] = None) -> SimpleChatResponse:
  
        full_answer = ""
        start_time = time.time()
        try:
            async for chunk in self.chat_stream(message, session_id, conversation_history):
                full_answer += chunk
            
            processing_time = time.time() - start_time
            logger.info(f"✅ 模型响应成功 (同步聚合) - 耗时: {processing_time:.2f}s, 字符数: {len(full_answer)}")
            return SimpleChatResponse(answer=full_answer, session_id=session_id, processing_time=processing_time)
        except Exception as e:
            logger.error(f"❌ chat 同步聚合失败: {e}")
            return SimpleChatResponse(answer="同步聊天失败。", session_id=session_id, processing_time=time.time() - start_time)


    async def chat_stream(
        self, 
        message: str, 
        session_id: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> AsyncIterator[str]:
        
        try:
            # 构建消息历史
            messages = self._build_messages(message, conversation_history)
            
    
            payload = {
                "model": "qwen",
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": True  
            }
            
            logger.debug(f"调用vLLM模型 (STREAM) - Session: {session_id}")
            
           
            async with self.client.stream(
                "POST",
                self.model_url,
                json=payload,
                timeout=self.timeout
            ) as response:
                
                response.raise_for_status() # 检查初始 HTTP 状态码

                # 异步迭代响应行（vLLM/OpenAI 兼容 SSE 格式）
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:].strip()
                        
                        if data == "[DONE]":
                            break # 流结束标记
                        
                        try:
                            # 解析 JSON 数据块
                            chunk = json.loads(data)
                            
                            # 提取增量文本: choices[0].delta.content
                            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                            
                            if content:
                                # 生成 (yield) 文本增量
                                yield content
                                
                        except json.JSONDecodeError:
                            logger.warning(f"无法解析的流式数据块: {data}")
                        except Exception as e:
                            logger.error(f"处理流式数据时发生错误: {e}")
                            
        except httpx.HTTPStatusError as e:
            error_msg = f"模型服务返回 HTTP 错误: {e.response.status_code}"
            logger.error(f"❌ {error_msg}, 响应: {e.response.text[:100]}...")
            yield error_msg
        except httpx.TimeoutException:
            error_msg = f"模型调用超时 - 超时设置: {self.timeout}s"
            logger.error(f"❌ {error_msg}")
            yield error_msg
        except Exception as e:
            error_msg = f"模型调用失败: {e}"
            logger.error(f"❌ {error_msg}")
            yield error_msg


    def _build_messages(
        self, 
        current_message: str, 
        conversation_history: Optional[List[Dict]] = None
    ) -> List[Dict[str, str]]:
        """构建OpenAI格式的消息历史 - 保持不变"""
        messages = []
        
        system_message = {
            "role": "system",
            "content": "你是一个有用的AI助手。请用中文回答用户的问题。"
        }
        messages.append(system_message)
        
        if conversation_history:
            for msg in conversation_history[-6:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        messages.append({
            "role": "user",
            "content": current_message
        })
        
        return messages
    
    def _parse_vllm_response(self, result: Dict[str, Any]) -> str:
        """解析vLLM的响应格式 - 保持不变 (主要用于同步 chat 方法的兼容)"""
        try:
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"].strip()
            
            logger.warning(f"非标准响应格式: {result}")
            
            if "text" in result:
                return result["text"].strip()
            elif "generated_text" in result:
                return result["generated_text"].strip()
            else:
                return f"[调试] 响应格式异常: {str(result)[:200]}"
                
        except Exception as e:
            logger.error(f"解析模型响应失败: {e}, 原始响应: {result}")
            return "抱歉，模型返回了无法解析的响应。"
    
    async def close(self):
        """关闭HTTP客户端 - 保持不变"""
        await self.client.aclose()

# 全局LLM服务实例
llm_service = LLMService()