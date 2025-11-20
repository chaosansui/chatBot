import json
from typing import List, Optional
from datetime import datetime
import redis.asyncio as redis
from redis.asyncio import Redis
from loguru import logger

# --- 新增导入 ---
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
# ----------------

from models.api_models import SessionInfo, ChatMessage
from core.config import settings

SESSION_KEY_PREFIX = "session:info:"
MESSAGE_KEY_PREFIX = "session:msgs:"

class SessionStore:
    """会话存储管理 - 基于异步 Redis 的高性能实现"""
    
    def __init__(self):
        self.client: Optional[Redis] = None
        # 会话信息和消息列表分开存储
        self.session_ttl = getattr(settings, 'SESSION_TTL', 24 * 3600)

    async def connect(self):
        """连接到 Redis 实例"""
        try:
            # 兼容逻辑：如果 settings 里没有完整的 REDIS_URL，则自动拼接
            if hasattr(settings, "REDIS_URL") and settings.REDIS_URL:
                redis_url = settings.REDIS_URL
            else:
                # 这里的字段名请根据你的 config.py 实际情况调整
                password_part = f":{settings.REDIS_PASSWORD}@" if settings.REDIS_PASSWORD else ""
                redis_url = f"redis://{password_part}{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"

            self.client = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
            
            # 尝试执行一个操作，确认连接成功
            await self.client.ping()
            logger.success("✅ Redis 连接成功")
        except Exception as e:
            logger.error(f"❌ Redis 连接失败: {e}")
            self.client = None
            # 这里的 raise 很重要，能让 main.py 捕获到启动失败
            raise

    async def close(self):
        """关闭 Redis 连接"""
        if self.client:
            await self.client.close()
            logger.info("👋 Redis 连接已关闭")

    # --- 核心操作方法 ---

    def _get_info_key(self, session_id: str) -> str:
        return SESSION_KEY_PREFIX + session_id

    def _get_messages_key(self, session_id: str) -> str:
        return MESSAGE_KEY_PREFIX + session_id

    async def create_session(self, session_id: str, user_id: Optional[str] = None) -> SessionInfo:
        """创建新会话"""
        if not self.client: raise ConnectionError("Redis 客户端未连接")
        
        current_time = datetime.now()
        
        session_info = SessionInfo(
            session_id=session_id,
            created_at=current_time,
            last_activity=current_time,
            message_count=0,
            metadata={"user_id": user_id} if user_id else {}
        )
        
        # model_dump_json 是 Pydantic v2 的写法，如果是 v1 请用 .json()
        session_data = session_info.model_dump_json()

        info_key = self._get_info_key(session_id)
        pipe = self.client.pipeline()
        pipe.set(info_key, session_data)
        pipe.expire(info_key, self.session_ttl)
        pipe.delete(self._get_messages_key(session_id))
        await pipe.execute()
        
        logger.info(f"创建新会话: {session_id}")
        return session_info
    
    async def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """获取会话信息"""
        if not self.client: return None

        info_key = self._get_info_key(session_id)
        session_data_json = await self.client.get(info_key)
        
        if session_data_json:
            try:
                session_data = json.loads(session_data_json)
                return SessionInfo(**session_data)
            except Exception as e:
                logger.error(f"解析会话数据失败: {e}")
                return None
        return None
    
    async def update_session_activity(self, session_id: str):
        """更新会话活动时间并重置 TTL"""
        if not self.client: return

        info_key = self._get_info_key(session_id)
        pipe = self.client.pipeline()

        # 1. 事务性更新 last_activity 和 message_count
        session_data_json = await self.client.get(info_key)
        if session_data_json:
            try:
                session_data = json.loads(session_data_json)
                session_data["last_activity"] = datetime.now().isoformat()
                session_data["message_count"] = session_data.get("message_count", 0) + 1
                pipe.set(info_key, json.dumps(session_data))
            except Exception:
                pass # 如果解析失败暂不处理，避免中断流程
        
        # 2. 重置 TTL
        pipe.expire(info_key, self.session_ttl)
        pipe.expire(self._get_messages_key(session_id), self.session_ttl)
        
        await pipe.execute()

    async def get_session_messages(self, session_id: str, limit: int = 10) -> List[ChatMessage]:
        """获取原始消息历史 (Pydantic 对象)"""
        if not self.client: return []

        messages_key = self._get_messages_key(session_id)
        # 确保只返回最新的 limit 条消息
        messages_json = await self.client.lrange(messages_key, -limit, -1)
        
        results = []
        for msg in messages_json:
            try:
                results.append(ChatMessage(**json.loads(msg)))
            except Exception:
                continue
        return results

    # --- ⭐️ 核心修复：新增 get_history 方法 ---
    async def get_history(self, session_id: str) -> List[BaseMessage]:
        """
        获取 LangChain 格式的消息历史，供 RAG Service 使用。
        这是 api/endpoints.py 调用的方法。
        """
        # 复用上面的逻辑，获取最近 6 条历史足矣（避免 Prompt 过长）
        chat_messages = await self.get_session_messages(session_id, limit=6)
        
        lc_messages = []
        for msg in chat_messages:
            if msg.role in ["user", "human"]:
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role in ["assistant", "ai", "model"]:
                lc_messages.append(AIMessage(content=msg.content))
                
        return lc_messages
    # ---------------------------------------
    
    async def add_message(self, session_id: str, role: str, content: str):
        """添加消息到会话"""
        if not self.client: 
            # 避免没有连接时调用报错
            logger.warning("Redis 未连接，消息无法保存")
            return

        messages_key = self._get_messages_key(session_id)
        
        # 1. 创建新消息
        new_message = ChatMessage(role=role, content=content)
        message_json = new_message.model_dump_json()

        pipe = self.client.pipeline()
        
        # 2. 推入列表
        pipe.rpush(messages_key, message_json)
        
        # 3. 限制长度 (保留最近 20 条)
        pipe.ltrim(messages_key, -20, -1) 
        
        await pipe.execute()
        
        # 异步更新活动状态
        await self.update_session_activity(session_id)
        
        logger.debug(f"会话 {session_id} 已记录消息: {role}")
    
    async def delete_session(self, session_id: str):
        """删除会话"""
        if not self.client: return

        pipe = self.client.pipeline()
        pipe.delete(self._get_info_key(session_id))
        pipe.delete(self._get_messages_key(session_id))
        await pipe.execute()
        
        logger.info(f"删除会话: {session_id}")

    async def get_or_create_session(self, session_id: str) -> SessionInfo:
        """获取或创建会话"""
        session = await self.get_session(session_id)
        if not session:
            session = await self.create_session(session_id)
        return session
    
# 全局会话存储实例
session_store = SessionStore()