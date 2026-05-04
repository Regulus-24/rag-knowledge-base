# rag_deepseek_v4.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ==================== 配置 DeepSeek V4 ====================
DEEPSEEK_API_KEY = "sk-c35a8d10c60f44fda26fce3a042c5ce9"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

os.environ["OPENAI_API_KEY"] = DEEPSEEK_API_KEY
os.environ["OPENAI_API_BASE"] = DEEPSEEK_BASE_URL

# ==================== 1. 加载 PDF ====================
pdf_path = "data/your_doc.pdf"  # 改成你的 PDF 路径
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"✅ 加载了 {len(documents)} 页")

# ==================== 2. 切分文本 ====================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"✅ 切分为 {len(chunks)} 个块")

# ==================== 3. 创建向量数据库 ====================
# 使用 DeepSeek V4 的嵌入模型
embeddings = OpenAIEmbeddings(
    model="deepseek-embed",
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,
)

# 创建或加载向量数据库
persist_directory = "./chroma_db"
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)
vectorstore.persist()
print(f"✅ 向量数据库已创建，保存在 {persist_directory}")

# ==================== 4. 创建检索器 ====================
retriever = vectorstore.as_retriever(
    search_type="similarity",  # 相似度检索
    search_kwargs={"k": 4}     # 返回最相关的4个块
)

# ==================== 5. 创建 LLM 生成器 ====================
llm = ChatOpenAI(
    model="deepseek-v4-pro",  # 或 "deepseek-v4-flash"
    temperature=0,
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,
)

# ==================== 6. 问答函数 ====================
def ask_question(question: str) -> dict:
    """
    检索相关文档 + 生成答案
    返回: {"answer": str, "sources": list}
    """
    # 步骤1：检索相关文档块
    relevant_docs = retriever.get_relevant_documents(question)
    
    if not relevant_docs:
        return {"answer": "未找到相关内容。", "sources": []}
    
    # 步骤2：拼接上下文
    context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
    
    # 步骤3：构造 prompt
    prompt = f"""你是一个严谨的知识库助手。请严格根据以下文档内容回答问题。

注意事项：
1. 只使用文档中给出的信息回答
2. 如果文档中没有相关信息，请直接说"文档中没有提到"
3. 不要编造或添加文档外的知识
4. 如果文档中有多条相关信息，可以综合回答

=== 文档内容 ===
{context}

=== 问题 ===
{question}

=== 答案 ===
"""
    
    # 步骤4：调用 LLM 生成答案
    response = llm.invoke(prompt)
    
    # 步骤5：提取答案和来源
    return {
        "answer": response.content,
        "sources": [doc.page_content[:150] + "..." for doc in relevant_docs]  # 来源摘要
    }

# ==================== 7. 交互式问答主循环 ====================
def main():
    print("\n" + "="*50)
    print("📚 RAG 知识库问答系统 (DeepSeek V4)")
    print("="*50)
    print(f"📄 文档块数: {len(chunks)}")
    print(f"🔍 检索数量: 每次检索 top-4 相关块")
    print("\n💡 输入问题开始问答，输入 'exit' 退出\n")
    
    while True:
        question = input("❓ 你的问题: ").strip()
        
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 再见！")
            break
        
        if not question:
            print("⚠️ 请输入有效问题\n")
            continue
        
        print("\n🔍 正在检索...")
        result = ask_question(question)
        
        print("\n" + "-"*40)
        print(f"🤖 答案:\n{result['answer']}")
        
        if result['sources']:
            print("\n📖 参考来源:")
            for i, src in enumerate(result['sources'], 1):
                print(f"  [{i}] {src}")
        print("-"*40 + "\n")

# ==================== 8. 入口 ====================
if __name__ == "__main__":
    main()