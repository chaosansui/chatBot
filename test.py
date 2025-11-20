from milvus import MilvusClient
client = MilvusClient(uri="http://localhost:19530")
print(client.get_version())