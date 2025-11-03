import json
import time
from typing import List, Dict, Optional
from datetime import datetime
from models.api_models import SessionInfo, ChatMessage
from core.config import settings
import logging

logger = logging.getLogger(__name__)

class SessionStore:
    """会话存储管理 - 基于内存的简单实现"""
    
    def __init__(self):
        # 使用内存存储（后续可以替换为Redis）
        self.sessions: Dict[str, Dict] = {}
        self.session_messages: Dict[str, List[Dict]] = {}
        self.session_ttl = 24 * 3600  # 24小时
    
    async def create_session(self, session_id: str, user_id: Optional[str] = None) -> SessionInfo:
        """创建新会话"""
        current_time = datetime.now()
        
        session_info = SessionInfo(
            session_id=session_id,
            created_at=current_time,
            last_activity=current_time,
            message_count=0,
            metadata={"user_id": user_id} if user_id else {}
        )
        
        # 存储会话信息
        self.sessions[session_id] = session_info.model_dump()
        self.session_messages[session_id] = []
        
        logger.info(f"创建新会话: {session_id}")
        return session_info
    
    async def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """获取会话信息"""
        session_data = self.sessions.get(session_id)
        if session_data:
            # 检查会话是否过期
            last_activity = datetime.fromisoformat(session_data["last_activity"])
            if (datetime.now() - last_activity).total_seconds() > self.session_ttl:
                await self.delete_session(session_id)
                return None
            return SessionInfo(**session_data)
        return None
    
    async def get_or_create_session(self, session_id: str) -> SessionInfo:
        """获取或创建会话"""
        session = await self.get_session(session_id)
        if not session:
            session = await self.create_session(session_id)
        return session
    
    async def update_session_activity(self, session_id: str):
        """更新会话活动时间"""
        session_data = self.sessions.get(session_id)
        if session_data:
            session_data["last_activity"] = datetime.now().isoformat()
            session_data["message_count"] += 1
    
    async def get_session_messages(self, session_id: str, limit: int = 10) -> List[ChatMessage]:
        """获取会话消息历史"""
        messages_data = self.session_messages.get(session_id, [])
        
        # 返回最新的limit条消息
        recent_messages = messages_data[-limit:]
        return [ChatMessage(**msg) for msg in recent_messages]
    
    async def add_message(self, session_id: str, role: str, content: str):
        """添加消息到会话"""
        if session_id not in self.session_messages:
            self.session_messages[session_id] = []
        
        # 创建新消息
        new_message = ChatMessage(role=role, content=content)
        self.session_messages[session_id].append(new_message.model_dump())
        
        # 限制消息历史长度（最多保存20条消息）
        if len(self.session_messages[session_id]) > 20:
            self.session_messages[session_id] = self.session_messages[session_id][-20:]
        
        # 更新会话活动
        await self.update_session_activity(session_id)
        
        logger.debug(f"会话 {session_id} 添加消息: {role} - {content[:50]}...")
    
    async def delete_session(self, session_id: str):
        """删除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
        if session_id in self.session_messages:
            del self.session_messages[session_id]
        
        logger.info(f"删除会话: {session_id}")
    
    async def cleanup_expired_sessions(self):
        """清理过期会话"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session_data in self.sessions.items():
            last_activity = datetime.fromisoformat(session_data["last_activity"])
            if (current_time - last_activity).total_seconds() > self.session_ttl:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.delete_session(session_id)
        
        if expired_sessions:
            logger.info(f"清理了 {len(expired_sessions)} 个过期会话")

# 全局会话存储实例
session_store = SessionStore()