# test.py
import sys
import pymilvus
from pymilvus import connections, utility

print(f"🐍 Python Executable: {sys.executable}")
print(f"📦 Pymilvus Version: {pymilvus.__version__}") # 必须是 2.4.x 或 2.6.x

print("-" * 30)
print("🚀 尝试连接 (Host: localhost, Port: 19530)...")

try:
    # 使用最稳健的连接方式
    connections.connect(
        alias="default", 
        host="localhost", 
        port="19530"
    )
    print("✅ 连接成功！")
    
    # 列出集合
    print(f"📚 现有集合: {utility.list_collections()}")
    
except Exception as e:
    print(f"❌ 连接失败: {e}")