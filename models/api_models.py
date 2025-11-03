from pydantic import BaseModel, Field,ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class SimpleChatRequest(BaseModel):
    """简化聊天请求模型"""
    message: str = Field(..., min_length=1, max_length=2000, description="用户消息")
    session_id: Optional[str] = Field(None, description="会话ID")

class SimpleChatResponse(BaseModel):
    """简化聊天响应模型"""
    answer: str = Field(..., description="模型回答")
    session_id: Optional[str] = Field(None, description="会话ID")
    processing_time: float = Field(..., description="处理时间(秒)")

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    model_ready: bool

class SessionInfo(BaseModel):
    """会话信息模型"""
    model_config = ConfigDict(extra='forbid')
    
    session_id: str = Field(..., description="会话ID")
    created_at: datetime = Field(..., description="创建时间")
    last_activity: datetime = Field(..., description="最后活动时间")
    message_count: int = Field(0, description="消息数量")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="会话元数据"
    )

class ChatMessage(BaseModel):
    """聊天消息模型"""
    model_config = ConfigDict(extra='forbid')
    
    role: str = Field(..., description="消息角色", examples=["user", "assistant"])
    content: str = Field(..., description="消息内容")
    timestamp: datetime = Field(default_factory=datetime.now, description="消息时间")