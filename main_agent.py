
import os
import json
from typing import List, TypedDict, Optional

import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_community.chat_models import ChatTongyi
from langgraph.graph import StateGraph, END
from typing_extensions import Annotated

# --- 导入我们新的数据预处理器接口 ---
from data_preprocessor import get_env_state, create_dispatch_features
# --- 导入报告生成器（我们稍后会创建它） ---
from report_generator import generate_retro_analysis_report

# --- 环境变量配置 ---
# 请确保您已设置 DASHSCOPE_API_KEY
if "DASHSCOPE_API_KEY" not in os.environ:
    os.environ["DASHSCOPE_API_KEY"] = "sk-4c03c712f47c4853922b6563a56f730c"

# --- LLM 初始化 ---
# 主LLM，用于生成调度建议
llm = ChatTongyi(model="qwen-plus")
# 评估LLM，用于评估建议的质量
eval_llm = ChatTongyi(model="qwen-turbo")

# --- 状态定义：简化后，专注于小时数据和LLM输出 ---
class DataCenterState(TypedDict):
    messages: List[BaseMessage]
    hourly_data: dict  # 存储 get_env_state 返回的单小时数据
    llm_insights: str  # LLM生成的调度建议
    evaluation_report: str # 评估报告
    # 我们将把每小时的完整状态追加到一个外部列表中，而不是在State内部管理

# --- 节点定义 ---

# 节点 1: LLM 智能调度 (首席架构师)
def llm_reasoning_node(state: DataCenterState) -> DataCenterState:
    print("\n[节点 1: LLM 智能调度]")
    hourly_data = state['hourly_data']
    hour = hourly_data['hour']
    it_load = hourly_data['it_load_mw']
    pue = hourly_data['pue']
    green_supply = hourly_data['green_supply_mw']
    ref_index = hourly_data['ref_index']

    prompt = f"""
    你是零碳数据中心首席架构师。当前是第 {hour} 小时，数据中心状态如下：
    - IT 负载: {it_load:.2f} MW
    - 实时 PUE: {pue:.3f}
    - 绿色能源供应 (风光): {green_supply:.2f} MW
    - 可再生能源利用率 (REF): {ref_index:.2%}

    基于以上信息，请提供简洁、明确、可执行的调度建议。专注于以下一点或两点：
    1.  **负载与能源匹配**：当前是应该鼓励承接更多计算任务，还是应该通过负载迁移/延迟来降低能耗？
    2.  **储能策略**：如果绿电过剩 (REF > 95%)，建议如何充电？如果绿电不足 (REF < 50%)，建议如何使用储能或从电网购电？
    3.  **运营风险**：是否存在任何潜在风险（例如，PUE过高，绿电供应远低于预期）？

    你的建议应在200字以内，直接给出结论。
    """
    state["messages"].append(HumanMessage(content=prompt))
    response = llm.invoke(state["messages"])
    state["messages"].append(response)
    state["llm_insights"] = response.content
    print(f"  - 首席架构师建议: {response.content}")
    return state

# 节点 2: 专家评估
def evaluate_suggestions_node(state: DataCenterState) -> DataCenterState:
    print("\n[节点 2: 专家评估]")
    insights = state["llm_insights"]
    prompt = f"""
    你是一位数据中心运营的独立评估专家。请评估以下由首席架构师提供的调度建议的质量，从三个维度打分（1-5分）：
    1.  **可行性**：建议是否符合物理规律和运营实际？
    2.  **经济性**：建议是否能有效降低成本（电费、碳税）？
    3.  **安全性**：建议是否会引入新的运营风险？

    调度建议：
    --- 
    {insights}
    --- 

    请以JSON格式返回你的评估报告，包含每个维度的分数和简短的评语。
    例如：{{"feasibility": 5, "feasibility_comment": "建议合理", ...}}
    """
    eval_response = eval_llm.invoke([HumanMessage(content=prompt)])
    state["evaluation_report"] = eval_response.content
    print(f"  - 评估报告: {eval_response.content}")
    return state

# --- 图（Graph）的构建 --- 
def create_scheduling_graph():
    graph = StateGraph(DataCenterState)
    graph.add_node("llm_reasoning", llm_reasoning_node)
    graph.add_node("evaluate_suggestions", evaluate_suggestions_node)

    graph.set_entry_point("llm_reasoning")
    graph.add_edge("llm_reasoning", "evaluate_suggestions")
    graph.add_edge("evaluate_suggestions", END)

    return graph.compile()

# --- 主程序入口：24小时闭环模拟 ---
if __name__ == '__main__':
    print("===== 开始24小时零碳调度闭环模拟 =====")

    # 步骤 1: 确保我们有预处理好的数据
    # 这会加载或创建 processed_metrics.csv
    all_hours_data = create_dispatch_features()

    # 步骤 2: 初始化图和结果存储
    app = create_scheduling_graph()
    full_day_results = []

    # 步骤 3: 按小时循环模拟
    for hour in range(24):
        print(f"\n{'='*20} 正在模拟第 {hour} 小时 {'='*20}")
        
        # 从预处理数据中获取当前小时的状态
        current_hour_data = get_env_state(hour)
        
        # 设置本次运行的初始状态
        initial_input = {
            "messages": [SystemMessage(content="你是一个数据中心调度AI助手。请根据指令行事。")],
            "hourly_data": current_hour_data
        }

        # 运行 LangGraph
        final_hour_state = {}
        # 流式运行并捕获最终状态
        for event in app.stream(initial_input):
            # event的key是节点名，value是该节点返回的state
            # 我们只关心最后返回的那个完整的state
            final_hour_state.update(event.get('llm_reasoning', {}))
            final_hour_state.update(event.get('evaluate_suggestions', {}))

        print(f"{'='*20} 第 {hour} 小时模拟结束 {'='*20}")
        full_day_results.append(final_hour_state)

    print("\n===== 24小时模拟全部完成 =====")
    print("正在将模拟结果保存到文件...")

    # 步骤 4: 将结果保存到JSON文件，实现断点续跑能力
    results_file = "simulation_results.json"
    # 我们需要处理一下数据，因为BaseMessage对象不能直接JSON序列化
    serializable_results = []
    for res in full_day_results:
        # 创建一个可序列化的副本
        serializable_res = res.copy()
        # 移除不可序列化的messages
        if 'messages' in serializable_res:
            del serializable_res['messages'] 
        serializable_results.append(serializable_res)

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=4)
    print(f"✅ 模拟结果已成功保存到 {results_file}")

    print("\n准备生成复盘报告...")
    # 步骤 5: 调用报告生成器
    try:
        generate_retro_analysis_report(serializable_results, all_hours_data)
        print("✅ 复盘报告生成成功！请查看 '调度复盘报告.md' 文件。")
    except Exception as e:
        print(f"❌ 生成报告时出错: {e}")
        print("请确保 report_generator.py 文件存在且无误。")

