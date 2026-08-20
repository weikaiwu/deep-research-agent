"""
GitHub 仓库元数据采集器
======================
采集主流 AI Agent / LLM 框架的仓库元数据（star/fork/issue/最近 push/最近 release），
写入 MySQL tech_research.github_repos 表，作为选型调研助手的结构化数据源。

用法:
    python scripts/fetch_github_repos.py

数据来源: GitHub REST API（无 token 限额 60 次/小时，本脚本约 30 次请求）
"""
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from mysql.connector import connect, Error

# 项目根目录，保证可独立运行
PROJECT_ROOT = Path(__file__).parents[1].resolve()
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

# 采集目标：与"AI Agent 技术选型调研"场景直接相关的开源项目
REPOS = [
    "langchain-ai/langchain",
    "langchain-ai/langgraph",
    "crewAIInc/crewAI",
    "microsoft/autogen",
    "run-llama/llama_index",
    "infiniflow/ragflow",
    "openai/openai-agents-python",
    "langflow-ai/langflow",
    "stanfordnlp/dspy",
    "ollama/ollama",
    "vllm-project/vllm",
    "microsoft/semantic-kernel",
    "flowiseai/flowise",
    "huggingface/transformers",
]

GITHUB_API = "https://api.github.com/repos"
HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "tech-research-agent",
    "X-GitHub-Api-Version": "2022-11-28",
}


def _parse_dt(value: str):
    """ISO8601 时间字符串 → datetime，None 保持 None"""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None


def fetch_repo(repo: str):
    """抓取单个仓库的元数据 + 最近 release 信息"""
    r = requests.get(f"{GITHUB_API}/{repo}", headers=HEADERS, timeout=15)
    if r.status_code != 200:
        print(f"  [X] {repo}: HTTP {r.status_code} ({r.text[:100]})")
        return None
    d = r.json()

    # 最近 release（无 token 下额外 1 次请求）
    release_tag, release_at = None, None
    try:
        rr = requests.get(f"{GITHUB_API}/{repo}/releases/latest", headers=HEADERS, timeout=15)
        if rr.status_code == 200:
            rd = rr.json()
            release_tag = rd.get("tag_name")
            release_at = _parse_dt(rd.get("published_at"))
    except Exception as e:
        print(f"  [!] {repo} release 获取失败: {e}")

    return {
        "repo_name": d["full_name"],
        "description": d.get("description") or "",
        "language": d.get("language") or "",
        "stars": d.get("stargazers_count") or 0,
        "forks": d.get("forks_count") or 0,
        "open_issues": d.get("open_issues_count") or 0,
        "pushed_at": _parse_dt(d.get("pushed_at")),
        "latest_release_tag": release_tag,
        "latest_release_at": release_at,
        "created_at": _parse_dt(d.get("created_at")),
        "url": d.get("html_url") or "",
    }


def main():
    db_cfg = {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER"),
        "password": os.getenv("MYSQL_PASSWORD"),
        "database": os.getenv("MYSQL_DATABASE", "tech_research"),
        "charset": "utf8mb4",
        "autocommit": True,
    }

    ok, fail = 0, 0
    try:
        conn = connect(**db_cfg)
    except Error as e:
        print(f"[X] MySQL 连接失败: {e}")
        return 1

    with conn.cursor() as cur:
        for repo in REPOS:
            print(f"[*] 采集 {repo} ...")
            data = fetch_repo(repo)
            if data is None:
                fail += 1
                continue
            cur.execute(
                """
                INSERT INTO github_repos
                    (repo_name, description, language, stars, forks, open_issues,
                     pushed_at, latest_release_tag, latest_release_at, created_at, url)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE
                    description=VALUES(description), language=VALUES(language),
                    stars=VALUES(stars), forks=VALUES(forks), open_issues=VALUES(open_issues),
                    pushed_at=VALUES(pushed_at), latest_release_tag=VALUES(latest_release_tag),
                    latest_release_at=VALUES(latest_release_at), url=VALUES(url)
                """,
                (
                    data["repo_name"], data["description"], data["language"],
                    data["stars"], data["forks"], data["open_issues"],
                    data["pushed_at"], data["latest_release_tag"],
                    data["latest_release_at"], data["created_at"], data["url"],
                ),
            )
            print(f"  [OK] {data['repo_name']} stars={data['stars']} pushed={data['pushed_at']}")
            ok += 1

    conn.close()
    print(f"=== 采集完成: 成功 {ok} / 失败 {fail} ===")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
