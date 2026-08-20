# 目标： 创建知识库检索子智能体（RAG）
# 使用本地技术文档知识库（Milvus 混合检索）检索内部/官方文档片段
from agent.prompts import sub_agents_content
from tools.rag_retrieval_tool import rag_retrieval


knowledge_base_agent = {
    "name": sub_agents_content['kb']['name'],
    "description": sub_agents_content['kb']['description'],
    "system_prompt": sub_agents_content['kb']['system_prompt'],
    "tools": [rag_retrieval]
}
