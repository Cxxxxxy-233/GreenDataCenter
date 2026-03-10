
import os
import sys

# 将项目的根目录添加到Python的模块搜索路径中
# 这使得我们可以直接运行此脚本进行测试，而不会出现模块找不到的错误
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatTongyi

# 假设这是我们的主LLM
llm = ChatTongyi(model="qwen-plus")

def create_infrastructure_agent(retriever):
    """
    创建基础设施架构师Agent。
    这个Agent的核心是一个RAG链，它能够根据知识库内容回答问题。

    参数:
        retriever: 一个已经配置好的LangChain检索器对象。

    返回:
        一个可以被调用的RAG链 (Runnable)。
    """
    
    # 1. 定义Prompt模板
    # 这个模板指导LLM如何利用检索到的上下文来回答问题
    prompt_template = """
    你是一位世界顶级的数据中心基础设施架构师。你的知识库中包含了关于制冷、储能、供配电等领域的专业文档。
    请基于以下检索到的上下文信息，专业、严谨地回答用户的问题。
    如果上下文中没有足够信息，请明确指出，并可以基于你的通用知识给出一些方向性建议。

    **检索到的上下文:**
    ----------------
    {context}
    ----------------

    **用户问题:**
    {question}

    **你的回答:**
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # 2. 构建RAG链 (The Magic Happens Here!)
    # 这就是LangChain表达力强大的地方，我们用 "|" 操作符像管道一样连接组件
    rag_chain = (
        # RunnableParallel允许我们并行处理任务
        # "context"的输入是：先获取用户问题，然后通过retriever获取相关文档
        # "question"的输入是：直接传递用户问题
        RunnableParallel(context=RunnablePassthrough() | retriever, question=RunnablePassthrough())
        |
        # 将上面处理好的 context 和 question 字典传入 prompt
        prompt
        |
        # 调用LLM进行推理
        llm
        |
        # 将LLM的输出（一个AIMessage对象）解析为纯字符串
        StrOutputParser()
    )
    
    return rag_chain

# --- 主程序入口 (用于独立测试) ---
if __name__ == '__main__':
    # 这是一个测试桩，实际项目中 retriever 会从 rag_builder 模块获取
    from rag_builder import build_or_load_vector_store

    print("===== 开始独立测试基础设施架构师Agent =====")
    
    # 1. 获取知识库检索器
    vector_store = build_or_load_vector_store()
    if not vector_store:
        print("❌ 知识库为空，无法创建Agent。请先在 knowledge_base 文件夹中添加文档并运行 rag_builder.py。")
    else:
        retriever = vector_store.as_retriever()
        
        # 2. 创建Agent (RAG链)
        infra_agent = create_infrastructure_agent(retriever)
        
        # 3. 测试调用
        print("\n--- 测试场景 1: 查询制冷技术 ---")
        question1 = "对于一个计划在华南地区建设的高密度算力中心，你会推荐哪种制冷技术？请说明理由。"
        print(f"❓ 问题: {question1}")
        
        # 直接调用链，传入问题
        # LangChain会自动处理背后复杂的RAG流程
        answer1 = infra_agent.invoke(question1)
        print(f"\n🤖 架构师回答:\n{answer1}")

