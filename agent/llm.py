from dotenv import load_dotenv
import os
from pathlib import Path
from langchain_openai import ChatOpenAI

# 加载配置文件：直接定位项目根目录的 .env，
# 避免 find_dotenv() 依赖调用栈在无文件场景（如 python -c）下找不到 .env。
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

# 直接构造 ChatOpenAI 实例（OpenAI 兼容接口，连通义千问 DashScope）。
# 注意：不用 init_chat_model —— 它在当前 langchain 版本返回 _ConfigurableModel
# （非 BaseChatModel 实例），deepagents 的 resolve_model 无法识别并会崩溃。
model = ChatOpenAI(
    model=os.getenv("LLM_QWEN_MAX"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)