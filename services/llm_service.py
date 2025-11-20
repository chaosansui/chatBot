# services/llm_service.py

import time
import json
import httpx
from loguru import logger
from typing import Dict, Any, List, Optional, AsyncIterator

from core.config import settings
from models.api_models import SimpleChatResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult 


class VLLMChatModel(BaseChatModel):
    """LangChain wrapper for vLLM API integration."""

    llm_service: Any

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        raise NotImplementedError("Use the async streaming method: _astream")

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]: 
        
        current_message = messages[-1].content if messages else ""
        conversation_history = [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in messages[:-1]
        ]
        
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")
        
        stream_generator = self.llm_service.chat_stream(
            message=current_message,
            conversation_history=conversation_history,
            session_id=kwargs.get("session_id"),
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        
        async for chunk_content in stream_generator:
            chunk = ChatGenerationChunk(
                text=chunk_content,
                message=AIMessageChunk(content=chunk_content), 
            )
            yield chunk

    @property
    def _llm_type(self) -> str:
        return "vllm-qwen-chat-model"

# ----------------------------------------------------

class LLMService:
    """Core LLM Service layer handling model API calls."""
    
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
            logger.info(f"✅ 模型响应成功 (同步聚合) - 耗时: {processing_time:.2f}s")
            return SimpleChatResponse(answer=full_answer, session_id=session_id, processing_time=processing_time)
        except Exception as e:
            logger.error(f"❌ chat 同步聚合失败: {e}")
            return SimpleChatResponse(answer="同步聊天失败。", session_id=session_id, processing_time=time.time() - start_time)


    async def chat_stream(
        self, 
        message: str, 
        session_id: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None, 
    ) -> AsyncIterator[str]:
        
        try:
            messages = self._build_messages(message, conversation_history)
            
            payload = {
                "model": "qwen",
                "messages": messages,
                "temperature": temperature if temperature is not None else self.temperature,
                "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
                "stream": True,
                "stop": stop if stop is not None else []
            }
            
            async with self.client.stream(
                "POST",
                self.model_url,
                json=payload,
                timeout=self.timeout
            ) as response:
                
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:].strip()
                        
                        if data == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(data)
                            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                            
                            if content:
                                # ⭐️ 关键修复：过滤 Qwen 的思考标签 ⭐️
                                if "<think>" in content or "</think>" in content:
                                    continue
                                    
                                yield content
                                
                        except json.JSONDecodeError:
                            logger.warning(f"无法解析的流式数据块: {data}")
                        except Exception as e:
                            logger.error(f"处理流式数据时发生错误: {e}")
                            
        except httpx.HTTPStatusError as e:
            error_msg = f"模型服务返回 HTTP 错误: {e.response.status_code}"
            logger.error(f"❌ {error_msg}")
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
        return ""
    
    async def close(self):
        await self.client.aclose()

# 全局LLM服务实例
llm_service = LLMService()