import sys
import re
from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI

# 修复 Windows 环境下 stdin/stdout 编码问题
sys.stdin.reconfigure(encoding="utf-8", errors="replace")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ============================================================
# 配置区
# ============================================================
DEEPSEEK_API_KEY = "sk-c35a8d10c60f44fda26fce3a042c5ce9"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-v4-pro"
PDF_PATH = "./data/info_point.pdf"

# ============================================================
# 初始化 DeepSeek 客户端
# ============================================================
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

# ============================================================
# 加载 PDF 全文
# ============================================================
def load_full_text(pdf_path: str) -> str:
    print(f"[加载] 正在读取 PDF：{pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"[加载] 共读取 {len(documents)} 页")

    full_text = "\n\n".join([doc.page_content for doc in documents])
    # 清理无效 Unicode 代理字符
    full_text = full_text.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")
    full_text = re.sub(r'[\ud800-\udfff]', '', full_text)

    estimated_tokens = len(full_text) // 2
    print(f"[统计] 全文约 {len(full_text)} 字符，预估 {estimated_tokens} tokens")
    return full_text


# ============================================================
# 调用 DeepSeek V4 生成答案
# ============================================================
SYSTEM_PROMPT = """你是一个专业的知识库问答助手。请严格根据用户提供的文档内容回答用户的问题。

要求：
1. 如果文档中有明确答案，请直接引用并总结
2. 如果文档中只有部分相关信息，请说明哪些是文档提到的、哪些不确定
3. 如果文档中完全没有相关信息，请如实告知："文档中未找到相关信息"
4. 回答使用中文，简洁明了"""

def generate_answer(question: str, full_text: str) -> str:
    user_message = f"""=== 知识库文档全文 ===
{full_text}

=== 用户问题 ===
{question}

=== 回答 ==="""

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=4096,
    )
    return response.choices[0].message.content


# ============================================================
# 交互式问答主循环
# ============================================================
def main():
    try:
        full_text = load_full_text(PDF_PATH)
    except FileNotFoundError:
        print(f"[错误] PDF 文件未找到：{PDF_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"[错误] 加载 PDF 失败：{e}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("  RAG 知识库问答系统已就绪（1M 上下文模式）")
    print(f"  LLM：DeepSeek V4 Pro | 全文直接推理")
    print("  输入 'quit' / 'exit' / 'q' 退出")
    print("=" * 60)
    print()

    while True:
        try:
            query = input(">>> 请输入问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[系统] 退出。")
            break

        if query.lower() in ("quit", "exit", "q"):
            print("[系统] 再见！")
            break
        if not query:
            continue

        print("[生成] 正在调用 DeepSeek V4 生成回答...")
        try:
            answer = generate_answer(query, full_text)
            print("\n" + "=" * 60)
            print(answer)
            print("=" * 60)
            print()
        except Exception as e:
            print(f"[错误] 生成回答失败：{e}")


if __name__ == "__main__":
    main()
