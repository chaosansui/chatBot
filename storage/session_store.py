import json
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from models.api_models import SessionInfo, ChatMessage
from core.config import settings
from loguru import logger
import redis.asyncio as redis
from redis.asyncio import Redis

SESSION_KEY_PREFIX = "session:info:"
MESSAGE_KEY_PREFIX = "session:msgs:"

class SessionStore:
    """会话存储管理 - 基于异步 Redis 的高性能实现"""
    
    def __init__(self):
        self.client: Optional[Redis] = None
        # 会话信息和消息列表分开存储
        self.session_ttl = settings.SESSION_TTL
        if not hasattr(settings, 'SESSION_TTL'):
            # 兼容性设置，建议在 config.py 中添加 SESSION_TTL
            self.session_ttl = 24 * 3600 

    async def connect(self):
        """连接到 Redis 实例"""
        try:
            # 使用配置中的 URL
            self.client = redis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
            # 尝试执行一个操作，确认连接成功
            await self.client.ping()
            logger.success("✅ Redis 连接成功")
        except Exception as e:
            logger.error(f"❌ Redis 连接失败: {e}")
            self.client = None
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
        
        session_data = session_info.model_dump_json()

        # 使用 Redis Hash 存储 Session Info，并设置 TTL
        info_key = self._get_info_key(session_id)
        pipe = self.client.pipeline()
        pipe.set(info_key, session_data)
        pipe.expire(info_key, self.session_ttl)
        pipe.delete(self._get_messages_key(session_id)) # 确保消息列表为空
        await pipe.execute()
        
        logger.info(f"创建新会话: {session_id}")
        return session_info
    
    async def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """获取会话信息"""
        if not self.client: return None

        info_key = self._get_info_key(session_id)
        session_data_json = await self.client.get(info_key)
        
        if session_data_json:
            session_data = json.loads(session_data_json)
            # Redis 已经处理了 TTL，无需手动检查过期时间
            return SessionInfo(**session_data)
        
        return None
    
    async def update_session_activity(self, session_id: str):
        """更新会话活动时间并重置 TTL"""
        if not self.client: return

        info_key = self._get_info_key(session_id)
        pipe = self.client.pipeline()

        # 1. 事务性更新 last_activity 和 message_count
        session_data_json = await self.client.get(info_key)
        if session_data_json:
            session_data = json.loads(session_data_json)
            session_data["last_activity"] = datetime.now().isoformat()
            session_data["message_count"] = session_data.get("message_count", 0) + 1
            pipe.set(info_key, json.dumps(session_data))
        
        # 2. 重置 TTL
        pipe.expire(info_key, self.session_ttl)
        pipe.expire(self._get_messages_key(session_id), self.session_ttl) # 消息列表也重置 TTL
        
        await pipe.execute()


    async def get_session_messages(self, session_id: str, limit: int = 10) -> List[ChatMessage]:
        """获取会话消息历史"""
        if not self.client: return []

        # 使用 LTRIM + LRANGE 实现列表限长和获取
        messages_key = self._get_messages_key(session_id)
        
        # 确保只返回最新的 limit 条消息
        messages_json = await self.client.lrange(messages_key, -limit, -1)
        
        return [ChatMessage(**json.loads(msg)) for msg in messages_json]
    
    async def add_message(self, session_id: str, role: str, content: str):
        """添加消息到会话"""
        if not self.client: raise ConnectionError("Redis 客户端未连接")

        messages_key = self._get_messages_key(session_id)
        
        # 1. 创建新消息并 JSON 序列化
        new_message = ChatMessage(role=role, content=content)
        message_json = new_message.model_dump_json()

        pipe = self.client.pipeline()
        
        # 2. 将新消息推入列表 (RPUSH)
        pipe.rpush(messages_key, message_json)
        
        # 3. 限制消息历史长度（最多保存20条消息）
        # LTRIM 保留最新的 20 条消息 (索引从 -20 开始)
        pipe.ltrim(messages_key, -20, -1) 
        
        # 4. 重置 TTL 和更新活动时间 (事务性)
        await pipe.execute()
        await self.update_session_activity(session_id)
        
        logger.debug(f"会话 {session_id} 添加消息: {role} - {content[:50]}...")
    
    async def delete_session(self, session_id: str):
        """删除会话"""
        if not self.client: return

        pipe = self.client.pipeline()
        pipe.delete(self._get_info_key(session_id))
        pipe.delete(self._get_messages_key(session_id))
        await pipe.execute()
        
        logger.info(f"删除会话: {session_id}")
    
    # 注意：Redis 的 TTL 机制自动处理过期，无需手动 cleanup_expired_sessions()
    async def cleanup_expired_sessions(self):
        """
        [Redis 实现]：该方法在 Redis 中不再需要，因为 Redis 的 TTL 机制会自动清理过期键。
        为保持接口兼容性，保留此方法。
        """
        logger.debug("Redis 模式下，无需手动清理过期会话。")
        pass 

    # 保持原有的同步/异步获取逻辑不变
    async def get_or_create_session(self, session_id: str) -> SessionInfo:
        """获取或创建会话"""
        session = await self.get_session(session_id)
        if not session:
            session = await self.create_session(session_id)
        return session
    
# 全局会话存储实例
session_store = SessionStore()