"""
技术选型报告评估脚本
====================
对生成的调研报告（Markdown）自动评估三项 KPI：
  ① 引用准确率：论断句（结论/对比/含数字的句子）中带引用标注（链接/来源/数据出处）的比例
  ② 维度覆盖度：选型必答 5 维度 checklist 覆盖情况
  ③ 耗时记录：报告文件时间戳（生成耗时占位，人工对比另计）

用法:
    python scripts/evaluate_report.py <报告.md路径>
"""
import re
import sys
from pathlib import Path

# 选型必答维度 checklist（可扩展）
DIMENSIONS = {
    "性能": ["性能", "吞吐", "延迟", "latency", "throughput", "benchmark", "基准"],
    "生态": ["生态", "社区", "插件", "集成", "生态圈", "第三方", "兼容"],
    "学习成本": ["学习成本", "上手", "学习曲线", "文档", "教程", "门槛", "易用"],
    "社区活跃度": ["活跃", "star", "issue", "更新频率", "release", "贡献", "commit", "维护"],
    "维护状态": ["维护", "停更", "停止", "弃用", "deprecated", "支持", "路线图", "roadmap"],
}

# 引用标注特征（句子里出现以下任一项 → 视为带引用）
CITATION_PATTERNS = [
    r"https?://",           # 链接
    r"\[[^\]]*\]\([^)]*\)",  # markdown 链接
    r"来源",                 # 来源
    r"数据来源",             # 数据来源
    r"参考资料",             # 参考资料
    r"GitHub",               # GitHub 仓库
    r"据\s*[^。；;]+?(?:统计|调查|报告|显示)",  # 据XX显示
]


def split_sentences(text: str):
    """按中英文句号/分号/换行切句"""
    text = re.sub(r"\n{2,}", "。", text)
    parts = re.split(r"(?<=[。！？；;！])", text)
    return [p.strip() for p in parts if p.strip()]


def is_claim(sentence: str) -> bool:
    """判断是否为论断句：包含结论/对比/数字等实质信息"""
    claim_markers = [
        r"\d+[\d,，.]*\s*%",      # 百分比
        r"\d+\s*[万kK]?",          # 数字
        r"(最高|最低|领先|优于|落后|更|最|推荐|适合|相比|对比|超过|不足|仅)",
        r"(结论|建议|推荐|综上|因此|优势|劣势|风险)",
    ]
    return any(re.search(p, sentence) for p in claim_markers)


def evaluate(md_path: Path):
    text = md_path.read_text(encoding="utf-8")
    sentences = split_sentences(text)

    claims = [s for s in sentences if is_claim(s)]
    claims_with_citation = []
    for s in claims:
        if any(re.search(p, s) for p in CITATION_PATTERNS):
            claims_with_citation.append(s)

    citation_rate = len(claims_with_citation) / len(claims) if claims else 0.0

    # 维度覆盖度
    covered, uncovered = [], []
    for dim, kws in DIMENSIONS.items():
        if any(kw.lower() in text.lower() for kw in kws):
            covered.append(dim)
        else:
            uncovered.append(dim)
    coverage = len(covered) / len(DIMENSIONS)

    # 统计
    print("=" * 60)
    print(f"评估报告: {md_path.name}")
    print(f"文件大小: {md_path.stat().st_size} bytes")
    print("=" * 60)
    print(f"\n① 引用准确率（论断带引用占比）")
    print(f"  总论断句: {len(claims)}")
    print(f"  带引用论断句: {len(claims_with_citation)}")
    print(f"  引用准确率: {citation_rate:.1%}")
    if claims:
        print(f"\n  论断句样例（前3条带引用）:")
        for s in claims_with_citation[:3]:
            print(f"    - {s[:80]}...")
    print(f"\n② 维度覆盖度")
    print(f"  覆盖维度: {len(covered)}/{len(DIMENSIONS)} ({coverage:.0%})")
    print(f"  已覆盖: {', '.join(covered)}")
    if uncovered:
        print(f"  未覆盖: {', '.join(uncovered)}")
    print(f"\n③ 耗时记录")
    import datetime
    mtime = datetime.datetime.fromtimestamp(md_path.stat().st_mtime)
    print(f"  报告生成时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  （注：Agent 端到端耗时需从会话日志计时，人工对比耗时另测）")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/evaluate_report.py <报告.md路径>")
        sys.exit(1)
    evaluate(Path(sys.argv[1]))
