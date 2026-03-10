
import os
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END

# --- 导入我们已经创建的模块 ---
from rag_builder import build_or_load_vector_store
from agents.infrastructure_agent import create_infrastructure_agent

# --- 1. 定义将在所有Agent之间传递的共享状态 ---
class SystemState(TypedDict):
    """ 定义整个系统的状态 """
    user_requirements: dict
    load_profile: Annotated[dict, lambda x, y: {**x, **y}]
    renewable_potential: Annotated[dict, lambda x, y: {**x, **y}]
    infrastructure_plan: Annotated[dict, lambda x, y: {**x, **y}]
    economic_analysis: Annotated[dict, lambda x, y: {**x, **y}]
    final_report: str

# --- 2. 定义各个专家Agent的节点函数 ---

# 我们已经完成的Agent
# 首先，我们需要在全局加载一次知识库
print("🧠 正在加载或构建RAG知识库...")
vector_store = build_or_load_vector_store()
retriever = vector_store.as_retriever() if vector_store else None
infrastructure_agent = create_infrastructure_agent(retriever) if retriever else None

# --- Agent占位符函数 ---
# 这些函数模拟了其他Agent的工作，以便我们可以测试完整的流程

def project_manager_agent(state: SystemState) -> SystemState:
    print("\n--- [专家Agent: 项目经理] ---")
    print("  -> 正在分析和结构化用户需求...")
    # 在真实场景中，这里会与用户交互或解析复杂输入
    # 目前，我们只打印用户需求
    print(f"  -> 用户需求: {state['user_requirements']}")
    return {}

def load_analyst_agent(state: SystemState) -> SystemState:
    print("\n--- [专家Agent: 负载分析师] ---")
    business_type = state['user_requirements'].get('business_type', '通用')
    print(f"  -> 正在根据业务类型 '{business_type}' 生成负载曲线...")
    # 模拟输出
    mock_profile = {
        "profile_name": f"{business_type} 典型24小时负载",
        "peak_load_mw": 150 if business_type == "大模型训练" else 80,
        "avg_load_mw": 140 if business_type == "大模型训练" else 50,
    }
    print(f"  -> 生成结果: {mock_profile}")
    return {"load_profile": mock_profile}

# 这是您同事将要替换的Agent
def renewable_planner_agent(state: SystemState) -> SystemState:
    print("\n--- [专家Agent: 可再生能源规划师 (占位符)] ---")
    area = state['user_requirements'].get('area_sqm', 10000)
    print(f"  -> 正在根据占地面积 {area} sqm 和API数据评估绿电潜力...")
    # 模拟输出
    mock_potential = {
        "distributed_pv_mw": area * 0.1 / 1000, # 假设每平米0.1kWp
        "comment": "已调用天气API（模拟），结果显示该地区光照资源良好。"
    }
    print(f"  -> 生成结果: {mock_potential}")
    return {"renewable_potential": mock_potential}

def infrastructure_node(state: SystemState) -> SystemState:
    print("\n--- [专家Agent: 基础设施架构师] ---")
    if not infrastructure_agent:
        print("  -> ⚠️ RAG Agent未初始化，跳过此步骤。")
        return {"infrastructure_plan": {"error": "RAG Agent not available."}}
    
    # 根据已有信息，构造一个问题给RAG Agent
    question = (
        f"根据以下条件，为数据中心推荐制冷和储能方案：\n"
        f"- 业务类型: {state['user_requirements'].get('business_type')}\n"
        f"- 算力密度: {state['user_requirements'].get('power_density')}\n"
        f"- IT峰值负载: {state['load_profile'].get('peak_load_mw')} MW"
    )
    print(f"  -> 正在向RAG知识库查询: {question}")
    answer = infrastructure_agent.invoke(question)
    print(f"  -> RAG Agent回答: {answer}")
    return {"infrastructure_plan": {"recommendation": answer}}

def economic_analyst_agent(state: SystemState) -> SystemState:
    print("\n--- [专家Agent: 经济与策略分析师 (占位符)] ---")
    print("  -> 正在进行能源配比和成本效益分析...")
    # 模拟输出
    mock_analysis = {
        "energy_mix": "30% 分布式光伏 + 20% 绿电长协 + 50% 电网调峰",
        "pareto_front": [
            {"green_ratio": 80, "cost_increase": 0.1},
            {"green_ratio": 90, "cost_increase": 0.25},
            {"green_ratio": 100, "cost_increase": 0.45},
        ],
        "comment": "该配比在满足用户90%绿电目标的同时，实现了成本最优。"
    }
    print(f"  -> 生成结果: {mock_analysis}")
    return {"economic_analysis": mock_analysis}

def report_synthesizer_agent(state: SystemState) -> SystemState:
    print("\n--- [专家Agent: 报告合成师] ---")
    print("  -> 正在汇总所有专家意见，生成最终规划方案...")
    # 简单地将所有状态信息格式化为字符串
    report = """
# 数据中心智能规划方案

## 1. 用户核心需求
{user_requirements}

## 2. 负载特性分析
{load_profile}

## 3. 可再生能源潜力
{renewable_potential}

## 4. 基础设施规划 (由RAG Agent提供)
{infrastructure_plan}

## 5. 经济与能源策略
{economic_analysis}
    """.format(**state)
    print("  -> ✅ 报告生成完毕。")
    return {"final_report": report}

# --- 3. 构建LangGraph工作流 ---

# 创建一个StateGraph对象，并绑定我们定义的状态
graph = StateGraph(SystemState)

# 添加所有专家Agent作为图中的节点
graph.add_node("project_manager", project_manager_agent)
graph.add_node("load_analyst", load_analyst_agent)
graph.add_node("renewable_planner", renewable_planner_agent)
graph.add_node("infrastructure_architect", infrastructure_node)
graph.add_node("economic_analyst", economic_analyst_agent)
graph.add_node("report_synthesizer", report_synthesizer_agent)

# 定义工作流的起点
graph.set_entry_point("project_manager")

# 定义节点之间的连接关系
graph.add_edge("project_manager", "load_analyst")
graph.add_edge("load_analyst", "renewable_planner")
graph.add_edge("renewable_planner", "infrastructure_architect")
graph.add_edge("infrastructure_architect", "economic_analyst")
graph.add_edge("economic_analyst", "report_synthesizer")
graph.add_edge("report_synthesizer", END)

# 编译图，生成一个可执行的应用
app = graph.compile()

# --- 4. 主程序入口：运行和测试 ---

if __name__ == "__main__":
    print("\n" + "="*40)
    print("🚀 开始运行绿色数据中心智能规划设计系统...")
    print("="*40)

    # 模拟用户输入
    user_input = {
        "user_requirements": {
            "target_pue": 1.2,
            "target_green_ratio": 0.9,
            "business_type": "大模型训练",
            "area_sqm": 50000,
            "power_density": "高 (20 kW/机柜)",
            "location": "内蒙古乌兰察布"
        }
    }

    # 运行整个工作流
    final_state = app.invoke(user_input)

    print("\n" + "="*40)
    print("✅ 系统运行结束，最终报告如下:")
    print("="*40)
    print(final_state['final_report'])

