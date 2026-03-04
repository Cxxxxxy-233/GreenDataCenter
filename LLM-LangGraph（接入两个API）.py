# LLM-LangGraph.py
import os
import json
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, FunctionMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


# --- 消息累加函数 (移动到 DataCenterState 定义之前) ---
def add_messages(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    """将消息列表合并，用于状态管理"""
    return left + right


# --- 1. 定义状态 (DataCenterState) ---
# 定义数据中心状态，用于在图的不同节点之间传递信息
class DataCenterState(TypedDict):
    current_datacenter_load_factor: float  # 当前数据中心负载率 (%)
    predicted_green_energy_ratio: float  # 预测的绿电占比 (%)
    grid_carbon_intensity: float  # 电网碳排放强度 (gCO2/kWh)
    target_pue: float  # 目标PUE值
    energy_storage_soc_current_percent: float  # 储能当前电量百分比 (%)
    energy_storage_capacity_mwh: float  # 储能总容量 (MWh)
    grid_stability_index: float  # 电网稳定性指数 (0-1，1为最稳定)
    energy_price_forecast_per_kwh: dict  # 电价预测 {小时: 价格}

    green_energy_allocation: dict  # 绿电分配方案 (由LLM生成)
    energy_storage_strategy: dict  # 储能调度策略 (由LLM生成)
    migration_path: dict  # 负载迁移或外部资源使用方案 (由LLM生成)
    llm_insights: str  # LLM生成的电力电子优化建议
    final_plan: dict  # 最终的综合调度方案
    evaluation_report: Optional[str]  # 新增：评估LLM对方案的评估报告
    human_feedback: str  # 人工反馈意见
    human_approved: bool  # 是否通过人工审核
    messages: Annotated[list, add_messages]  # 这里引用了 add_messages


# 2. 初始化主大模型 (使用 XSimple 提供的模型接口或兼容接口)
# 请替换为实际可用的 API Key 和 Base URL
# 确保在运行前设置 DASHSCOPE_API_KEY 环境变量
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 新增：3. 初始化评估模型 (使用 DeepSeek 模型接口或兼容接口)
# 请替换为实际可用的 API Key 和 Base URL，注意这里使用了 DASHSCOPE_API_KEY2
# 确保在运行前设置 DASHSCOPE_API_KEY2 环境变量
eval_llm = ChatOpenAI(
    model="deepseek-v3.2",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# --- 节点定义 ---

# 节点 1: 基础数据初步分析
def perform_initial_analysis(state: DataCenterState) -> DataCenterState:
    """对绿电占比、负载率、碳排强度、PUE目标等进行初步评估"""
    print("\n[节点 1: 基础数据初步分析]")
    green_ratio = state["predicted_green_energy_ratio"]
    load_factor = state["current_datacenter_load_factor"]
    carbon_intensity_grid = state["grid_carbon_intensity"]
    pue_target = state["target_pue"]
    soc = state["energy_storage_soc_current_percent"]
    stability = state["grid_stability_index"]

    # 简单的逻辑判断，生成初步评估建议
    analysis_report = f"""
    初步评估报告：
    - 预测绿电占比: {green_ratio * 100:.1f}%
    - 当前数据中心负载率: {load_factor * 100:.1f}%
    - 电网碳排放强度: {carbon_intensity_grid} gCO2/kWh
    - 目标 PUE: {pue_target:.2f}
    - 储能当前电量: {soc:.1f}%
    - 电网稳定性: {stability:.2f}

    初步结论和建议：
    """
    if green_ratio > 0.6 and load_factor < 0.7 and stability > 0.7:
        analysis_report += "系统状态良好，绿电充足，负载适中，电网稳定。建议优先考虑最大化绿电消纳，并优化储能充放电策略。"
    elif green_ratio < 0.3 or load_factor > 0.8 or stability < 0.5:
        analysis_report += "系统面临挑战，绿电可能不足，或负载较高，或电网不稳定。需谨慎制定策略，优先保障核心业务，考虑需求侧响应或外部资源。"
    else:
        analysis_report += "系统状态一般，需要平衡绿电利用、负载管理和成本效益。建议关注细致的调度方案。"

    state["messages"].append(
        AIMessage(content=f"✅ 完成初步数据分析。\n{analysis_report}")
    )
    return state


# 节点 2: 绿电分配方案生成
def generate_green_allocation_plan(state: DataCenterState) -> DataCenterState:
    """基于初步分析，制定绿电分配方案"""
    print("\n[节点 2: 绿电分配方案生成]")
    green_ratio = state["predicted_green_energy_ratio"]
    load_factor = state["current_datacenter_load_factor"]
    pue_target = state["target_pue"]
    carbon_intensity_grid = state["grid_carbon_intensity"]
    soc = state["energy_storage_soc_current_percent"]

    prompt = f"""
    当前数据中心状态：
    - 预测绿电占比: {green_ratio * 100:.1f}%
    - 当前负载率: {load_factor * 100:.1f}%
    - 目标PUE: {pue_target:.2f}
    - 电网碳强度: {carbon_intensity_grid} gCO2/kWh
    - 储能当前SOC: {soc:.1f}%

    请基于以上信息，制定一个绿电分配方案。方案应该考虑：
    1. 优先满足核心业务的绿电需求。
    2. 优化次要业务和可中断业务的绿电使用。
    3. 考虑到储能状态，是否应充/放电。
    4. 降低PUE和碳排放。
    5. 返回一个JSON格式的方案，包括：
        - "critical_workloads": {{"core_business_services": "x%", "high_priority_computing": "y%"}}
        - "flexible_workloads": {{"batch_processing": "a%", "data_analytics": "b%"}}
        - "recommendation": "文本形式的调度建议"
    """

    messages = [
        AIMessage(content=f"初步分析结果已参考。"),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)
    try:
        # 尝试解析LLM返回的JSON
        allocation = json.loads(response.content)
    except json.JSONDecodeError:
        # 如果不是标准JSON，则尝试提取建议，并用默认结构包裹
        # 注意：这里也需要更新默认结构的JSON值，使其符合LLM生成预期，或者至少是有效JSON
        allocation = {
            "critical_workloads": {"core_business_services": "50%", "high_priority_computing": "30%"},
            "flexible_workloads": {"batch_processing": "10%", "data_analytics": "10%"},
            "recommendation": response.content  # 将非JSON内容作为建议
        }
        print(f"Warning: LLM did not return perfect JSON. Extracted recommendation: {response.content}")

    # 示例内部逻辑，如果LLM输出直接是百分比，这里可以处理
    # 假设LLM返回的 allocation 已经是处理好的字典
    # 示例中，我直接使用了LLM的输出作为 allocation
    state["green_energy_allocation"] = allocation
    state["messages"].append(
        AIMessage(content="✅ 绿电分配方案制定完成")
    )
    return state


# 节点 3: 储能调度与负载迁移建议
def generate_storage_and_migration_plan(state: DataCenterState) -> DataCenterState:
    """生成储能调度策略和负载迁移建议"""
    print("\n[节点 3: 储能调度与负载迁移建议]")
    green_ratio = state["predicted_green_energy_ratio"]
    load_factor = state["current_datacenter_load_factor"]
    soc = state["energy_storage_soc_current_percent"]
    capacity = state["energy_storage_capacity_mwh"]
    grid_stability = state["grid_stability_index"]
    price_forecast = state["energy_price_forecast_per_kwh"]
    current_allocation = json.dumps(state["green_energy_allocation"], indent=2, ensure_ascii=False)

    prompt = f"""
    当前数据中心状态：
    - 预测绿电占比: {green_ratio * 100:.1f}%
    - 当前负载率: {load_factor * 100:.1f}%
    - 储能当前SOC: {soc:.1f}%
    - 储能总容量: {capacity} MWh
    - 电网稳定性: {grid_stability:.2f}
    - 电价预测 (未来几小时): {json.dumps(price_forecast)}
    - 已制定的绿电分配方案：
    {current_allocation}

    请根据上述信息，综合考虑绿电分配方案，制定详细的储能调度策略和负载迁移/外部资源使用方案。
    储能调度策略应包括充放电的时机、持续时长和目标SOC。
    负载迁移方案应指出哪些负载可以迁移，或何时寻求外部绿色算力。

    返回一个JSON格式的方案，包括：
    - "energy_storage_strategy": {{"action": "charge/discharge/idle", "duration_hours": X, "target_soc_percent": Y, "reason": "..."}}
    - "migration_or_external_resource_plan": {{"action": "migrate_batch_jobs_to_cloud/seek_external_green_compute/none", "details": "..."}}
    - "overall_recommendation": "综合建议文本"
    """

    messages = [
        AIMessage(content=f"已参考绿电分配方案。"),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)
    try:
        plan = json.loads(response.content)
    except json.JSONDecodeError:
        plan = {
            "energy_storage_strategy": {"action": "idle", "duration_hours": 0, "target_soc_percent": soc,
                                        "reason": "LLM output non-JSON, default to idle."},
            "migration_or_external_resource_plan": {"action": "none", "details": response.content},
            "overall_recommendation": response.content
        }
        print(f"Warning: LLM did not return perfect JSON. Extracted content: {response.content}")

    state["energy_storage_strategy"] = plan.get("energy_storage_strategy", {})
    state["migration_path"] = plan.get("migration_or_external_resource_plan", {})
    state["messages"].append(
        AIMessage(content="✅ 储能调度与负载迁移建议制定完成")
    )
    return state


# 节点 4: LLM生成电力电子优化建议
def generate_llm_insights_power_electronics(state: DataCenterState) -> DataCenterState:
    """从电力电子角度对调度方案进行优化和深入建议"""
    print("\n[节点 4: LLM生成电力电子优化建议]")
    current_green_allocation = json.dumps(state["green_energy_allocation"], indent=2, ensure_ascii=False)
    current_storage_strategy = json.dumps(state["energy_storage_strategy"], indent=2, ensure_ascii=False)
    current_migration_plan = json.dumps(state["migration_path"], indent=2, ensure_ascii=False)
    pue_target = state["target_pue"]

    prompt = f"""
    你是一名资深的电力电子专家，请基于以下已制定的数据中心调度方案，从电力电子、系统效率和绿色能源利用最大化的角度，提供深入的优化建议。
    尤其关注以下方面：
    - 如何通过先进的电力电子技术（如高效率模块、智能拓扑、DC-DC转换器优化）进一步降低PUE。
    - 储能系统的具体运行模式、电池管理系统(BMS)优化、功率转换效率提升建议。
    - 绿电并网与系统稳定性的电力电子解决方案。
    - 应对负载波动和绿电波动的技术措施。

    目标PUE: {pue_target:.2f}
    已制定绿电分配方案：
    {current_green_allocation}
    已制定储能调度策略：
    {current_storage_strategy}
    已制定负载迁移方案：
    {current_migration_plan}

    请提供专业、具体、可操作的电力电子优化建议。
    """

    messages = [
        AIMessage(content=f"已参考当前调度方案，开始生成电力电子优化建议。"),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)
    state["llm_insights"] = response.content
    state["messages"].append(
        AIMessage(content="✅ 大模型洞察 (电力电子优化视角) 生成成功")
    )
    return state


# 节点 5: 整合所有策略，生成最终调度方案
def integrate_and_finalize_plan(state: DataCenterState) -> DataCenterState:
    """整合所有策略和建议，生成最终调度方案"""
    print("\n[节点 5: 整合所有策略，生成最终调度方案]")

    final_plan = {
        "timestamp": "2024-XX-XX HH:MM:SS",  # 实际应用中替换为当前时间
        "datacenter_status_overview": {
            "predicted_green_energy_ratio": f"{state['predicted_green_energy_ratio'] * 100:.1f}%",
            "current_load_factor": f"{state['current_datacenter_load_factor'] * 100:.1f}%",
            "grid_carbon_intensity": f"{state['grid_carbon_intensity']} gCO2/kWh",
            "target_PUE": f"{state['target_pue']:.2f}",
            "energy_storage_SOC": f"{state['energy_storage_soc_current_percent']:.1f}%",
            "grid_stability": f"{state['grid_stability_index']:.2f}"
        },
        "migration_or_external_resource_plan": state["migration_path"],
        "energy_storage_strategy": state["energy_storage_strategy"],
        "green_energy_allocation_detail": state["green_energy_allocation"],  # 绿电分配方案
        "compliance_and_impact": {
            "pue_target_consideration": f"Current PUE target {state['target_pue']:.2f} needs continuous monitoring.",
            "carbon_reduction_focus": "Emphasis on maximizing green energy use and efficient storage."
        },
        "expert_advice_from_power_electronics_perspective": state["llm_insights"]  # 包含LLM的深度建议
    }

    state["final_plan"] = final_plan
    state["messages"].append(
        AIMessage(content="✅ 完整调度方案生成成功")
    )
    return state


# 新增节点 6: 方案评估节点 (使用第二个API DeepSeek)
def evaluate_suggestions_node(state: DataCenterState) -> DataCenterState:
    """使用第二个LLM对当前生成的方案和专家建议进行评估"""
    print("\n[节点 6: 自动评估方案环节]")

    final_plan = state.get("final_plan", {})
    green_allocation = final_plan.get("green_energy_allocation_detail", "无绿电分配方案")
    expert_advice = final_plan.get("expert_advice_from_power_electronics_perspective", "无专家建议")
    current_state_overview = final_plan.get("datacenter_status_overview", "无状态概览")
    # 这里的 overall_recommendation 应该来自 migration_path，如果直接从 final_plan 里的顶层获取可能会出错
    # 修正为从 migration_or_external_resource_plan 中获取
    migration_plan_details = final_plan.get("migration_or_external_resource_plan", {})
    overall_recommendation = migration_plan_details.get("overall_recommendation", "无综合建议")
    storage_strategy = final_plan.get("energy_storage_strategy", "无储能策略")

    # 构建评估提示
    prompt_content = f"""
    你是一个严谨的评估专家，请对以下数据中心智能调度方案和电力电子专家建议进行公正、客观的评估。
    评估要点：
    1. 方案的合理性与可行性：是否充分考虑了所有关键因素（如绿电占比、负载、储能、PUE目标、电网稳定性、电价等）？
    2. 建议的深度与实用性：专家建议是否具有开创性或是否切实可行？
    3. 潜在风险与改进点：方案或建议可能存在的风险，以及可以改进的地方。
    4. 整体评级：给出一个简要的整体评价（例如：优秀、良好、一般、需改进），并附上详细理由。
    5. 评估报告需结构化，包含总结和具体评估点。

    --- 待评估的智能调度方案 ---

    数据中心当前状态概览: {json.dumps(current_state_overview, indent=2, ensure_ascii=False)}

    计划的绿电分配方案: {json.dumps(green_allocation, indent=2, ensure_ascii=False)}

    储能调度策略: {json.dumps(storage_strategy, indent=2, ensure_ascii=False)}

    负载迁移/外部资源方案: {json.dumps(migration_plan_details, indent=2, ensure_ascii=False)}

    电力电子专家建议: {expert_advice}

    --- 评估报告 ---
    请输出你的评估报告，要求详细且有建设性。
    """

    messages = [
        HumanMessage(content=prompt_content)
    ]

    try:
        response = eval_llm.invoke(messages)
        evaluation_report = response.content
    except Exception as e:
        evaluation_report = f"评估模型调用失败: {e}"
        print(f"Error calling evaluation LLM (DeepSeek): {e}")

    state["evaluation_report"] = evaluation_report
    state["messages"].append(
        AIMessage(content=f"📝 方案自动评估完成。评估报告已生成。")
    )
    print("评估报告已生成。")
    return state


# 节点 7: 人工审核节点 (原节点 6)
def human_review_node(state: DataCenterState) -> DataCenterState:
    """展示最终方案、自动评估报告并等待人工确认"""
    print("\n[节点 7: 人工审核环节]")
    plan = state["final_plan"]
    evaluation_report = state.get("evaluation_report", "无自动评估报告。")

    print("\n" + "=" * 30 + " 人工审核环节 " + "=" * 30)
    print("当前生成的最终方案如下 (摘要)：")
    print("绿电分配:")
    print(json.dumps(plan["green_energy_allocation_detail"], indent=2, ensure_ascii=False))
    print("\n专家建议摘要:")
    print(plan["expert_advice_from_power_electronics_perspective"])
    print("\n储能调度策略摘要:")
    print(json.dumps(plan["energy_storage_strategy"], indent=2, ensure_ascii=False))
    print("\n负载迁移/外部资源方案摘要:")
    print(json.dumps(plan["migration_or_external_resource_plan"], indent=2, ensure_ascii=False))

    print("\n" + "=" * 25 + " 自动评估报告 " + "=" * 25)
    print(evaluation_report)
    print("=" * 60)

    # 获取用户输入
    print("\n>>> 请审核方案 <<<")
    user_input = input("输入 'y' 确认通过并结束，或输入具体修改意见(例如: '增加储能放电比例') 以重新生成: ")

    if user_input.strip().lower() in ['y', 'yes', 'ok', '通过', '']:
        state["human_approved"] = True
        state["messages"].append(HumanMessage(content="✅ 人工审核通过"))
        print(">>> 审核通过，流程结束。")
    else:
        state["human_approved"] = False
        state["human_feedback"] = user_input
        state["messages"].append(HumanMessage(content=f"❌ 人工审核未通过，反馈意见: {user_input}"))
        print(f">>> 收到反馈: {user_input}。正在重新进行智能分析...")

    return state


# 构建LangGraph工作流
def create_scheduling_graph():
    workflow = StateGraph(DataCenterState)

    # 添加节点
    workflow.add_node("analyze", perform_initial_analysis)
    workflow.add_node("generate_green_allocation", generate_green_allocation_plan)
    workflow.add_node("generate_storage_and_migration", generate_storage_and_migration_plan)
    workflow.add_node("generate_llm_insights", generate_llm_insights_power_electronics)
    workflow.add_node("integrate_and_finalize", integrate_and_finalize_plan)
    workflow.add_node("evaluate", evaluate_suggestions_node)  # 新增评估节点
    workflow.add_node("human_review", human_review_node)

    # 构建边
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "generate_green_allocation")
    workflow.add_edge("generate_green_allocation", "generate_storage_and_migration")
    workflow.add_edge("generate_storage_and_migration", "generate_llm_insights")
    workflow.add_edge("generate_llm_insights", "integrate_and_finalize")

    # 新增边：在最终方案生成后，进行自动评估
    workflow.add_edge("integrate_and_finalize", "evaluate")

    # 修改边：从评估节点进入人工审核
    workflow.add_edge("evaluate", "human_review")

    # 循环逻辑：如果人工审核未通过，则返回到初始分析节点（或根据需求返回到其他节点）
    workflow.add_conditional_edges(
        "human_review",
        lambda state: "redo" if not state["human_approved"] else "end",
        {
            "redo": "analyze",  # 返回到初始分析节点进行重新思考
            "end": END
        }
    )

    return workflow.compile()


if __name__ == "__main__":
    # 初始化环境变量 (请在实际运行前设置你的API Key)
    # os.environ["DASHSCOPE_API_KEY"] = "YOUR_DASHSCOPE_QWEN_API_KEY"
    # os.environ["DASHSCOPE_API_KEY2"] = "YOUR_DASHSCOPE_DEEPSEEK_API_KEY"

    # 初始化工作流
    app = create_scheduling_graph()

    # 定义初始状态数据
    initial_state = {
        "current_datacenter_load_factor": 0.75,  # 75% 负载率
        "predicted_green_energy_ratio": 0.5,  # 50% 绿电占比
        "grid_carbon_intensity": 450.0,  # 电网碳排放强度 450 gCO2/kWh
        "target_pue": 1.3,  # 目标 PUE
        "energy_storage_soc_current_percent": 0.6,  # 储能当前 60% 电量
        "energy_storage_capacity_mwh": 100.0,  # 储能总容量 100 MWh
        "grid_stability_index": 0.8,  # 电网稳定性良好
        "energy_price_forecast_per_kwh": {
            "h0": 0.8, "h1": 0.75, "h2": 0.7, "h3": 0.65,
            "h4": 0.6, "h5": 0.7, "h6": 0.85, "h7": 0.9,
            "h8": 0.95, "h9": 1.0, "h10": 0.9, "h11": 0.8
        },
        "green_energy_allocation": {},
        "energy_storage_strategy": {},
        "migration_path": {},
        "llm_insights": "",
        "final_plan": {},
        "evaluation_report": None,  # 初始化为 None
        "human_feedback": "",
        "human_approved": False,
        "messages": []
    }

    print("开始执行数据中心智能调度工作流...")
    print("初始状态:")
    for key, value in initial_state.items():
        if key != "messages":
            print(f"  - {key}: {value}")
    print("=" * 60)

    # 执行工作流
    result = app.invoke(initial_state)

    # 输出结果
    print("\n[最终绿电分配方案及专家建议]")
    if "final_plan" in result and "green_energy_allocation_detail" in result["final_plan"]:
        print("绿电分配细则:")
        print(json.dumps(result["final_plan"]["green_energy_allocation_detail"], indent=2, ensure_ascii=False))

    if "final_plan" in result and "expert_advice_from_power_electronics_perspective" in result["final_plan"]:
        print("\n专家LLM建议 (侧重电力电子优化):")
        print(result["final_plan"]["expert_advice_from_power_electronics_perspective"])

    if "evaluation_report" in result and result["evaluation_report"]:
        print("\n自动评估报告:")
        print(result["evaluation_report"])

    print("\n[关键执行日志]")
    for msg in result["messages"]:
        # 修正了判断条件，避免在打印日志时遗漏评估失败的信息
        if isinstance(msg,
                      AIMessage) and "大模型洞察 (电力电子优化视角)" not in msg.content and "评估模型调用失败" not in msg.content:
            print(f"AIMessage: {msg.content}")
        elif isinstance(msg, HumanMessage):
            print(f"HumanMessage: {msg.content}")

    if result["human_approved"]:
        print("\n最终方案已通过人工审核。")
    else:
        print(f"\n最终方案未通过人工审核。反馈意见: {result.get('human_feedback', '无具体意见')}")