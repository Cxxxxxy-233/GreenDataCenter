
import os
from tqdm import tqdm

# --- LangChain Core Components ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader

# --- Configuration ---
KNOWLEDGE_BASE_PATH = "knowledge_base"
VECTOR_STORE_PATH = "vector_store/faiss_index"

# 使用一个开源的、效果优秀的支持中文的嵌入模型
# 第一次运行时，它会自动从HuggingFace下载模型文件（约400-500MB）
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- Document Loaders Mapping ---
# 将文件扩展名映射到对应的 LangChain 加载器
DOCUMENT_LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".doc": UnstructuredWordDocumentLoader,
    # 您可以在这里添加更多格式, 例如 .txt, .html 等
    # ".txt": TextLoader,
}

def load_documents(path):
    """
    扫描指定路径下的所有支持的文档，并使用对应的加载器加载它们。
    """
    print(f"📚 开始从 '{path}' 文件夹加载文档...")
    documents = []
    if not os.path.exists(path):
        print(f"⚠️ 警告: 知识库文件夹 '{path}' 不存在。将创建一个空知识库。")
        return documents

    # 使用 tqdm 创建一个进度条来显示加载过程
    files_to_load = [f for f in os.listdir(path) if any(f.endswith(ext) for ext in DOCUMENT_LOADERS)]
    if not files_to_load:
        print(f"ℹ️ 信息: 在 '{path}' 中未找到支持的文档文件（.pdf, .docx）。")
        return documents

    with tqdm(total=len(files_to_load), desc="加载文档") as pbar:
        for file in files_to_load:
            file_path = os.path.join(path, file)
            ext = "." + file.rsplit(".", 1)[-1]
            if ext in DOCUMENT_LOADERS:
                try:
                    loader = DOCUMENT_LOADERS[ext](file_path)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"❌ 加载文件 '{file_path}' 时出错: {e}")
            pbar.update(1)
            
    print(f"✅ 文档加载完成，共加载了 {len(documents)} 页/部分内容。")
    return documents

def build_vector_store(documents):
    """
    将加载的文档进行分割、嵌入，并构建一个FAISS向量存储。
    """
    if not documents:
        print("ℹ️ 信息: 没有文档可供处理，跳过向量库构建。")
        return None

    # 1. 分割 (Split)
    print("🔪 开始将文档分割成小块 (Chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"✅ 分割完成，共得到 {len(chunks)} 个文本块。")

    # 2. 嵌入 (Embed)
    print(f"🧠 正在初始化嵌入模型: '{EMBEDDING_MODEL_NAME}'...")
    print("(首次运行时会自动下载模型，请耐心等待...)")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # 3. 存储 (Store)
    print("💾 正在构建并持久化向量数据库 (FAISS)...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"✅ 向量数据库已成功构建并保存到: '{VECTOR_STORE_PATH}'")
    return vector_store

def build_or_load_vector_store(rebuild: bool = False):
    """
    主函数：构建或加载向量数据库。
    - 如果 rebuild=True，或向量存储文件不存在，则强制重新构建。
    - 否则，直接加载已有的向量存储。
    
    返回: 一个 FAISS 向量存储对象，如果没有任何文档则返回 None。
    """
    print("\n--- RAG 知识库模块 --- ")
    if rebuild or not os.path.exists(VECTOR_STORE_PATH):
        if rebuild:
            print("🔧 检测到 'rebuild=True'，将强制重建知识库。")
        else:
            print("⚠️ 未发现已缓存的向量数据库，将开始构建新的知识库。")
        
        # 执行完整的构建流程
        documents = load_documents(KNOWLEDGE_BASE_PATH)
        vector_store = build_vector_store(documents)
    else:
        # 直接加载已有的向量存储
        print(f"✅ 发现已缓存的向量数据库，直接从 '{VECTOR_STORE_PATH}' 加载。")
        print(f"🧠 正在初始化嵌入模型: '{EMBEDDING_MODEL_NAME}'...")
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        print("✅ 向量数据库加载成功。")
        
    return vector_store

import argparse

# --- 主程序入口 (用于独立测试) ---
if __name__ == '__main__':
    # 使用 argparse 来处理命令行参数
    parser = argparse.ArgumentParser(description="构建或加载RAG知识库的向量存储。")
    parser.add_argument(
        "--rebuild",
        action="store_true", # 当出现 --rebuild 参数时，其值为 True
        help="如果指定，则强制从头开始重建知识库，而不是加载现有缓存。"
    )
    args = parser.parse_args()

    print("===== 开始独立测试 RAG 知识库构建模块 =====")
    
    # 根据命令行参数决定是否重建
    vector_store = build_or_load_vector_store(rebuild=args.rebuild)

    if vector_store:
        print("\n===== 知识库查询测试 =====")
        # 将向量数据库转换为一个“检索器”，它可以找出与问题最相关的文本块
        retriever = vector_store.as_retriever()
        
        query = "数据中心的制冷策略有哪些？"
        print(f"❓ 测试查询: {query}")
        
        # relevant_docs 是一个包含与问题最相关的 Document 对象的列表
        relevant_docs = retriever.invoke(query)
        
        print("\n🔍 检索到的最相关内容:")
        for i, doc in enumerate(relevant_docs):
            print(f"\n--- [相关片段 {i+1}] ---")
            print(doc.page_content)
            print(f"(来源: {os.path.basename(doc.metadata.get('source', 'N/A'))}, 页码: {doc.metadata.get('page', 'N/A')})")
    else:
        print("\n⚠️ 知识库为空，无法进行查询测试。")

