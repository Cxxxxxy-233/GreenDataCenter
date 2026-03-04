import os
import json
import pandas as pd
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, FunctionMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import operator  # 用于 add_messages 列表操作
from pycaret.time_series import load_model, predict_model


# --- 消息累加函数 ---
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

    # --- 新增变量，参考“评估维度与数据输入” ---
    raw_sensor_data_input: Dict[str, Any]  # 底层原始输入数据，模拟传感器、智能电表、楼宇自控系统(BAS)数据
    # 示例: {"total_electricity_consumption_kwh": 200000, "it_equipment_power_kwh": 130000, "data_center_temperature_c": 22.5, ...}
    power_purchase_contract_info: Dict[str, Any]  # 电力交易合同信息
    # 示例: {"green_energy_purchase_kwh": 100000, "carbon_credit_offset_tco2": 50, "contract_end_date": "2024-12-31", ...}
    calculated_kpis: Dict[str, Any]  # 核心合规指标 KPIs，由预处理程序或分析节点计算
    # 示例: {"PUE_actual": 1.4, "Renewable_Energy_Ratio_actual": 0.6, "Carbon_Emission_Intensity_actual": 300, ...}
    compliance_redlines: Dict[
        str, bool]  # 明确存储四大红线的符合情况，例如 {"PUE_redline_met": True, "Green_Ratio_redline_met": False, ...}
    compliance_overall_status: Optional[str]  # 整体合规情况描述，例如 "完全符合规定要求" 或 "存在一项不合规"
    # --- 新增变量结束 ---

    # 新增：用于存储数据中心负载预测结果
    load_prediction_results: Optional[pd.DataFrame]
    # 新增：用于存储风光出力预测结果
    renewable_prediction_results: Optional[pd.DataFrame]

    messages: Annotated[List[BaseMessage], add_messages]
    analysis_result: Optional[str]
    green_energy_allocation: Optional[str]
    energy_storage_strategy: Optional[str]
    migration_path: Optional[str]
    llm_insights: Optional[str]
    final_plan: Optional[Dict]
    evaluation_report: Optional[str]
    human_approved: Optional[bool]
    human_feedback: Optional[str]


# 2. 初始化主大模型 (使用 XSimple 提供的模型接口或兼容接口)
# 请替换为实际可用的 API Key 和 Base URL
# 确保在运行前设置 DASHSCOPE_API_KEY 环境变量
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 3. 初始化评估模型 (使用 DeepSeek 模型接口或兼容接口)
# 请替换为实际可用的 API Key 和 Base URL，注意这里使用了 DASHSCOPE_API_KEY2
# 确保在运行前设置 DASHSCOPE_API_KEY2 环境变量
eval_llm = ChatOpenAI(
    model="deepseek-v3.2",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# --- 节点定义 ---

# 节点: 数据中心负载预测节点
def load_prediction_node(state: DataCenterState) -> DataCenterState:
    """加载并运行预训练的数据中心负载预测模型 (load.pkl)"""
    print("--- 正在调用数据中心负载预测模型 --- ")
    try:
        model = load_model('load', verbose=False)
        predictions = predict_model(model)
        print("--- ✅ 数据中心负载预测成功 ---")
        state["load_prediction_results"] = predictions
        state["messages"].append(AIMessage(content="✅ 负载预测模型调用成功。"))
    except Exception as e:
        error_message = f"--- ❌ 数据中心负载预测失败: {e} ---"
        print(error_message)
        state["load_prediction_results"] = pd.DataFrame({"error": [error_message]})
        state["messages"].append(AIMessage(content=error_message))
    return state


# 节点: 风光出力预测节点
def renewable_prediction_node(state: DataCenterState) -> DataCenterState:
    """加载并运行预训练的风光出力预测模型 (power.pkl)"""
    print("--- 正在调用风光出力预测模型 --- ")
    try:
        model = load_model('power', verbose=False)
        predictions = predict_model(model)
        print("--- ✅ 风光出力预测成功 ---")
        state["renewable_prediction_results"] = predictions
        state["messages"].append(AIMessage(content="✅ 风光出力预测模型调用成功。"))
    except Exception as e:
        error_message = f"--- ❌ 风光出力预测失败: {e} ---"
        print(error_message)
        state["renewable_prediction_results"] = pd.DataFrame({"error": [error_message]})
        state["messages"].append(AIMessage(content=error_message))
    return state


# 节点 1: 基础数据初步分析
def perform_initial_analysis(state: DataCenterState) -> DataCenterState:
    """对绿电占比、负载率、碳排强度、PUE目标以及合规性指标进行初步评估"""
    print("\n[节点 1: 基础数据初步分析 ]")
    green_ratio = state["predicted_green_energy_ratio"]
    load_factor = state["current_datacenter_load_factor"]
    carbon_intensity = state["grid_carbon_intensity"]
    target_pue = state["target_pue"]

    # 新增对底层原始输入数据和电力交易合同信息的处理
    raw_sensors = state.get("raw_sensor_data_input", {})
    purchase_contract = state.get("power_purchase_contract_info", {})

    # --- 模拟计算核心合规指标 (CalculatedKPIs) ---
    # 假设我们从传感器数据和合同信息中提取并计算实际PUE、实际绿电比例、实际碳排放强度等
    actual_total_electricity_consumption = raw_sensors.get("total_electricity_consumption_kwh", 0)
    # 如果未提供IT功耗，则默认设置为总功耗，以避免除零，虽然实际情况PUE会是1.0
    it_equipment_power_consumption = raw_sensors.get("it_equipment_power_kwh", actual_total_electricity_consumption)

    # 避免除零错误
    calculated_actual_pue = actual_total_electricity_consumption / it_equipment_power_consumption if it_equipment_power_consumption > 0 else 99.9

    actual_green_energy_consumed = purchase_contract.get("green_energy_purchase_kwh", 0)
    calculated_actual_green_ratio = actual_green_energy_consumed / actual_total_electricity_consumption if actual_total_electricity_consumption > 0 else 0

    # 模拟实际碳排放强度：基于电网强度和实际绿电占比计算
    # 实际场景可能更复杂，这里简化处理，假设绿电完全无碳排放
    calculated_actual_carbon_intensity = carbon_intensity * (1 - calculated_actual_green_ratio)

    state["calculated_kpis"] = {
        "PUE_actual": calculated_actual_pue,
        "Renewable_Energy_Ratio_actual": calculated_actual_green_ratio,
        "Carbon_Emission_Intensity_actual": calculated_actual_carbon_intensity,
        "Total_Electricity_Consumption_kWh": actual_total_electricity_consumption,
        "IT_Equipment_Power_kWh": it_equipment_power_consumption,
        "Data_Center_Temperature_C": raw_sensors.get("data_center_temperature_c"),  # 示例
        "Cooling_Load_kW": raw_sensors.get("cooling_load_kw"),  # 示例
        "Server_Utilization_Rate": raw_sensors.get("server_utilization_rate"),  # 示例
    }

    # --- 模拟对照《标准》中的四大红线 ---
    # 实际应用中这些红线会是具体的数值或逻辑，这里先设定一些示例边界
    redline_pue_limit = 1.35  # 假设PUE红线：低于1.35
    redline_green_ratio_min = 0.55  # 假设绿电占比最低要求：高于55%
    redline_carbon_intensity_max = 350  # 假设碳排放强度最高值：低于350 gCO2/kWh
    redline_load_factor_min = 0.70  # 假设负载率最低要求：高于70%

    state["compliance_redlines"] = {
        "PUE_redline_met": calculated_actual_pue <= redline_pue_limit,
        "Green_Ratio_redline_met": calculated_actual_green_ratio >= redline_green_ratio_min,
        "Carbon_Intensity_redline_met": calculated_actual_carbon_intensity <= redline_carbon_intensity_max,
        "Load_Factor_redline_met": load_factor >= redline_load_factor_min,  # 注意这里仍用 predicted_load_factor
    }

    compliance_issues = [k for k, v in state["compliance_redlines"].items() if not v]
    if not compliance_issues:
        state["compliance_overall_status"] = "完全符合规定要求。"
    else:
        # 提取不合规项的友好名称
        issue_names = {
            "PUE_redline_met": "PUE不达标",
            "Green_Ratio_redline_met": "绿电占比不达标",
            "Carbon_Intensity_redline_met": "碳排放强度不达标",
            "Load_Factor_redline_met": "负载率不达标",
        }
        problematic_items = [issue_names.get(issue, issue) for issue in compliance_issues]
        state["compliance_overall_status"] = f"存在以下不合规项：{', '.join(problematic_items)}。"

    prompt = f"""作为一个数据中心智能调度助手，你当前的职责是根据以下数据对数据中心的运行情况进行初步分析，并识别潜在的问题或改进机会。

    **当前和预测运营数据:**
    - 当前数据中心负载率: {load_factor * 100:.2f}% (目标红线: {redline_load_factor_min * 100:.2f}%)
    - 预测绿电占比: {green_ratio * 100:.2f}%
    - 电网碳排放强度: {carbon_intensity:.2f} gCO2/kWh
    - 目标PUE: {target_pue:.2f} (目标红线: {redline_pue_limit:.2f})

    **实际计算的KPIs (基于RawSensors和PowerPurchaseContractInfo):**
    - 实际PUE: {state['calculated_kpis']['PUE_actual']:.2f}
    - 实际可再生能源占比: {state['calculated_kpis']['Renewable_Energy_Ratio_actual'] * 100:.2f}%
    - 实际碳排放强度: {state['calculated_kpis']['Carbon_Emission_Intensity_actual']:.2f} gCO2/kWh
    - 总用电量: {state['calculated_kpis']['Total_Electricity_Consumption_kWh']:.0f} kWh
    - IT设备用电量: {state['calculated_kpis']['IT_Equipment_Power_kWh']:.0f} kWh
    - 数据中心温度: {state['calculated_kpis'].get('Data_Center_Temperature_C', 'N/A')} °C

    **合规性检查结果 (模拟四大红线):**
    - PUE是否达标 (低于{redline_pue_limit:.2f}): {state['compliance_redlines']['PUE_redline_met']}
    - 绿电占比是否达标 (高于{redline_green_ratio_min * 100:.2f}%): {state['compliance_redlines']['Green_Ratio_redline_met']}
    - 碳排放强度是否达标 (低于{redline_carbon_intensity_max:.2f} gCO2/kWh): {state['compliance_redlines']['Carbon_Intensity_redline_met']}
    - 负载率是否达标 (高于{redline_load_factor_min * 100:.2f}%): {state['compliance_redlines']['Load_Factor_redline_met']}
    **整体合规状态:** {state['compliance_overall_status']}

    请总结当前数据中心的运行状况，特别指出合规性方面的问题、潜在的风险，并提出初步的改进方向。
    """
    ai_message = llm.invoke(prompt)
    state["analysis_result"] = ai_message.content
    state["messages"].append(AIMessage(content=state["analysis_result"]))

    print(f"初步分析结果:\n{state['analysis_result']}")
    return state


# 节点 2: 制定绿电分配策略
def allocate_green_energy(state: DataCenterState) -> DataCenterState:
    """根据初步分析和当前绿电数据制定绿电分配策略"""
    print("\n[节点 2: 制定绿电分配策略]")
    green_ratio = state["predicted_green_energy_ratio"]
    analysis = state["analysis_result"]
    current_kpis = state["calculated_kpis"]
    compliance_status = state["compliance_overall_status"]

    prompt = f"""根据以下数据中心运营分析结果、实时KPIs和合规性状态，制定详细的绿电分配策略：
    初步分析结果: {analysis}
    当前预测绿电占比: {green_ratio * 100:.2f}%
    实际PUE: {current_kpis.get('PUE_actual', 'N/A'):.2f}
    实际可再生能源占比: {current_kpis.get('Renewable_Energy_Ratio_actual', 'N/A') * 100:.2f}%
    整体合规状态: {compliance_status}

    请给出具体的绿电分配方案，例如设定绿电优先供给哪些服务、如何优化绿电使用效率、是否需要考虑绿电交易市场等，以提高绿电消纳能力和降低碳排放。
    """
    ai_message = llm.invoke(prompt)
    state["green_energy_allocation"] = ai_message.content
    state["messages"].append(AIMessage(content=f"绿电分配策略:\n{state['green_energy_allocation']}"))

    print(f"绿电分配策略:\n{state['green_energy_allocation']}")
    return state


# 节点 3: 制定储能和负载迁移/外部资源策略
def formulate_strategies(state: DataCenterState) -> DataCenterState:
    """根据绿电分配策略，制定储能和负载迁移/外部资源策略"""
    print("\n[节点 3: 制定储能和负载迁移/外部资源策略]")
    green_allocation = state["green_energy_allocation"]
    analysis = state["analysis_result"]
    current_kpis = state["calculated_kpis"]
    compliance_status = state["compliance_overall_status"]

    prompt_storage = f"""根据绿电分配策略、数据中心分析、实际KPIs和合规性状态，制定详细的储能调度策略：
    初步分析结果: {analysis}
    绿电分配策略: {green_allocation}
    实际PUE: {current_kpis.get('PUE_actual', 'N/A'):.2f}
    实际可再生能源占比: {current_kpis.get('Renewable_Energy_Ratio_actual', 'N/A') * 100:.2f}%
    整体合规状态: {compliance_status}

    请围绕以下方面给出建议:
    - 储能的充放电策略 (何时充电？何时放电？优先满足哪些负载？)。
    - 如何利用储能平滑绿电波动。
    - 如何优化储能系统寿命和效率。
    """
    storage_message = llm.invoke(prompt_storage)
    state["energy_storage_strategy"] = storage_message.content
    state["messages"].append(AIMessage(content=f"储能调度策略:\n{state['energy_storage_strategy']}"))
    print(f"储能调度策略:\n{state['energy_storage_strategy']}")

    prompt_migration = f"""根据绿电分配策略、储能调度策略、数据中心分析、实际KPIs和合规性状态，制定负载迁移或外部资源调度方案：
    初步分析结果: {analysis}
    绿电分配策略: {green_allocation}
    储能调度策略: {state["energy_storage_strategy"]}
    实际PUE: {current_kpis.get('PUE_actual', 'N/A'):.2f}
    实际可再生能源占比: {current_kpis.get('Renewable_Energy_Ratio_actual', 'N/A') * 100:.2f}%
    整体合规状态: {compliance_status}

    请给出具体的建议:
    - 何时进行负载迁移 (例如，当绿电不足或电价高企时)。
    - 哪些类型的负载适合迁移。
    - 如何与外部数据中心或云计算资源进行协同。
    - 考虑合规性，例如，是否迁移会影响关键指标的达标。
    """
    migration_message = llm.invoke(prompt_migration)
    state["migration_path"] = migration_message.content
    state["messages"].append(AIMessage(content=f"负载迁移/外部资源策略:\n{state['migration_path']}"))
    print(f"负载迁移/外部资源策略:\n{state['migration_path']}")

    return state


# 节点 4: 基于供需预测的LLM智能调度
def llm_reasoning_node(state: DataCenterState) -> DataCenterState:
    """利用LLM根据供需预测，制定核心调度策略"""
    print("\n[节点 4: 基于供需预测的LLM智能调度]")

    if state.get("human_feedback"):
        print("--- 正在根据人工反馈重新进行智能分析 ---")
        feedback_prompt = f"\n重要补充：请根据以下用户反馈调整你的建议：'{state['human_feedback']}'"
    else:
        feedback_prompt = ""

    prompt = ChatPromptTemplate.from_template("""你是一位零碳数据中心首席架构师。根据下面24小时的“负载预测”(需求)和“风光出力预测”(供应)，制定一份简洁的调度方案。

**负载预测 (需求):**
{load_data}

**风光出力预测 (供应):**
{renewable_data}

**请简洁回答：**
1.  **供需关系**: 哪些时段绿电富余，哪些时段存在缺口？
2.  **采购策略**: 针对缺口，建议购买绿电(PPA)还是绿证(REC)？
3.  **调度指令**: 明确储能的充/放电策略，以及何时启动电网购电。

**输出要求：**
直接输出方案，不要添加任何签名或日期。
{feedback_prompt}
""")

    chain = prompt | llm
    load_str = state.get("load_prediction_results", pd.DataFrame()).to_string()
    renewable_str = state.get("renewable_prediction_results", pd.DataFrame()).to_string()

    response = chain.invoke({
        "load_data": load_str,
        "renewable_data": renewable_str,
        "feedback_prompt": feedback_prompt
    })

    state["llm_insights"] = response.content
    state["messages"].append(AIMessage(content="✅ 大模型分析完成 (首席架构师视角)"))
    return state


# 节点 5: 整合并生成最终调度方案
def integrate_and_finalize_plan(state: DataCenterState) -> DataCenterState:
    print("\n[节点 5: 整合并生成最终调度方案]")
    final_plan = {
        "chief_architect_recommendations": state.get("llm_insights", "无建议"),
        "load_forecast_24h": state.get("load_prediction_results").to_dict('records') if state.get("load_prediction_results") is not None else "N/A",
        "renewable_energy_forecast_24h": state.get("renewable_prediction_results").to_dict('records') if state.get("renewable_prediction_results") is not None else "N/A",
        "initial_conditions": {
            "predicted_green_energy_ratio": state.get("predicted_green_energy_ratio"),
            "current_datacenter_load_factor": state.get("current_datacenter_load_factor"),
            "grid_carbon_intensity": state.get("grid_carbon_intensity"),
            "target_pue": state.get("target_pue"),
        }
    }

    state["final_plan"] = final_plan
    state["messages"].append(
        AIMessage(content="✅ 最终调度方案已整合生成。")
    )
    return state


# 节点 6: 方案评估节点 (使用第二个API DeepSeek)
def evaluate_suggestions_node(state: DataCenterState) -> DataCenterState:
    print("\n[节点 6: 方案评估节点 (更新)]")
    final_plan_json = json.dumps(state["final_plan"], ensure_ascii=False, indent=2)

    eval_prompt = f"""你是一个智能评估专家，你需要评估以下数据中心调度方案的合理性、可行性、以及最重要的——合规性。
    请特别关注方案是否能够解决在初步分析中发现的合规性问题（如果存在），并确保方案能促使数据中心符合相关的PUE、绿电占比、碳排放强度、负载率等指标要求（包括模拟的四大红线）。

    **初始分析结果概要:**
    {state['analysis_result']}

    **数据中心实际运行KPIs:**
    {json.dumps(state.get('calculated_kpis', {}), ensure_ascii=False, indent=2)}
    **合规性红线检查结果:**
    {json.dumps(state.get('compliance_redlines', {}), ensure_ascii=False, indent=2)}
    **整体合规状态:** {state.get('compliance_overall_status', '未评估')}

    **生成的最终调度方案:**
    {final_plan_json}

    请给出详细的评估报告，包括：
    1.  方案的优点和亮点。
    2.  潜在的风险或不足。
    3.  是否有效地考虑并解决了合规性问题？评估其对PUE、绿电占比、碳排放强度、负载率等核心合规指标的改善潜力。
    4.  给出最终的建议：该方案是否可行？需要哪些修改才能达到最佳效果或完全合规？
    """

    try:
        eval_message = eval_llm.invoke(eval_prompt)
        state["evaluation_report"] = eval_message.content
        state["messages"].append(AIMessage(content=f"评估报告:\n{state['evaluation_report']}"))
    except Exception as e:
        state["evaluation_report"] = f"评估模型调用失败: {e}"
        state["messages"].append(AIMessage(content=f"评估模型调用失败: {e}"))
        print(f"评估模型调用失败: {e}")

    print(f"评估报告:\n{state['evaluation_report']}")
    return state


# 节点 7: 人工审核节点
def human_review_node(state: DataCenterState) -> DataCenterState:
    print("\n[节点 7: 人工审核]")
    llm_insights = state.get("llm_insights", "未能获取方案摘要。")
    load_predictions = state.get("load_prediction_results")
    renewable_predictions = state.get("renewable_prediction_results")

    print("\n" + "=" * 25 + " 请审核以下方案 " + "=" * 25)

    print("\n--- 未来24小时数据中心负载预测 ---")
    if load_predictions is not None and not load_predictions.empty:
        print(load_predictions.to_string())
    else:
        print("(无负载预测数据)")

    print("\n--- 未来24小时风光出力预测 ---")
    if renewable_predictions is not None and not renewable_predictions.empty:
        print(renewable_predictions.to_string())
    else:
        print("(无风光出力预测数据)")

    print("\n--- 首席架构师建议摘要 ---")
    print(llm_insights)
    print("=" * 60)

    # 获取用户输入
    print("\n>>> 请审核方案 <<<")
    try:
        user_input = input("输入 'y' 确认通过并结束，或输入具体修改意见以重新生成: ")
    except KeyboardInterrupt:
        print("\n操作被中断，默认批准方案。")
        user_input = 'y'

    if user_input.strip().lower() in ['y', 'yes', 'ok', '通过', '']:
        state["human_approved"] = True
        state["human_feedback"] = "方案已通过人工审核。"
        state["messages"].append(HumanMessage(content="人工审核: 方案已批准"))
    else:
        state["human_approved"] = False
        state["human_feedback"] = user_input.strip()
        state["messages"].append(HumanMessage(content=f"人工审核: 方案未批准，反馈意见: {user_input.strip()}"))
    return state


# --- 定义路由函数 ---
def route_human_review(state: DataCenterState) -> str:
    """根据人工审核结果选择下一步骤"""
    if state["human_approved"]:
        print("人工审核通过，工作流结束。")
        return "end"
    else:
        print("人工审核未通过，返回LLM Reasoning节点重新生成。")
        return "llm_reasoning" # 返回到LLM节点重新生成


# --- 5. 构建图 ---
def create_scheduling_graph():
    workflow = StateGraph(DataCenterState)

    # 添加节点
    workflow.add_node("perform_initial_analysis", perform_initial_analysis)
    workflow.add_node("load_prediction", load_prediction_node)
    workflow.add_node("renewable_prediction", renewable_prediction_node)
    workflow.add_node("llm_reasoning", llm_reasoning_node)
    workflow.add_node("integrate_and_finalize_plan", integrate_and_finalize_plan)
    workflow.add_node("evaluate_suggestions_node", evaluate_suggestions_node)
    workflow.add_node("human_review_node", human_review_node)

    # 设置起始节点
    workflow.set_entry_point("perform_initial_analysis")

    # 定义边
    workflow.add_edge("perform_initial_analysis", "load_prediction")
    workflow.add_edge("load_prediction", "renewable_prediction")
    workflow.add_edge("renewable_prediction", "llm_reasoning")
    workflow.add_edge("llm_reasoning", "integrate_and_finalize_plan")
    workflow.add_edge("integrate_and_finalize_plan", "evaluate_suggestions_node")
    workflow.add_edge("evaluate_suggestions_node", "human_review_node")

    # 添加条件边
    workflow.add_conditional_edges(
        "human_review_node",
        route_human_review,
        {
            "llm_reasoning": "llm_reasoning", # 如果未通过，回到LLM Reasoning重新开始迭代
            "end": END  # 如果通过，结束
        }
    )

    return workflow.compile()


# --- 主执行逻辑 ---
if __name__ == "__main__":
    # 初始化环境变量 (请在实际运行前设置你的API Key)

    # 初始化工作流
    app = create_scheduling_graph()

    # 定义初始状态数据
    initial_state = {
        "current_datacenter_load_factor": 0.75,  # 75% 负载率
        "predicted_green_energy_ratio": 0.5,  # 50% 预测绿电占比
        "grid_carbon_intensity": 450.0,  # 电网碳排放强度 450 gCO2/kWh
        "target_pue": 1.3,  # 目标 PUE
        # --- 新增的初始数据，用于模拟RawSensors和PowerPurchaseContractInfo ---
        "raw_sensor_data_input": {
            "total_electricity_consumption_kwh": 200000,  # 实际总用电量
            "it_equipment_power_kwh": 140000,  # IT设备用电量 (PUE = 20万/14万 ≈ 1.42)
            "data_center_temperature_c": 23.0,  # 数据中心温度
            "cooling_load_kw": 600,  # 冷却负载
            "server_utilization_rate": 0.65,  # 服务器利用率
            # ... 其他传感器数据
        },
        "power_purchase_contract_info": {
            "green_energy_purchase_kwh": 90000,  # 实际采购绿电量 (实际绿电占比 = 9万/20万 = 0.45)
            "carbon_credit_offset_tco2": 50,  # 碳信用抵消量
            "contract_end_date": "2024-12-31",
            # ... 其他合同信息
        },
        "calculated_kpis": {},  # 初始为空，由perform_initial_analysis节点计算
        "compliance_redlines": {},  # 初始为空，由perform_initial_analysis节点计算
        "compliance_overall_status": None,  # 初始为空，由perform_initial_analysis节点设定
        # --- 新增初始化数据结束 ---
        "messages": [],  # 初始化消息列表
        "human_approved": None,  # 初始为None，等待人工审核
        "evaluation_report": None,  # 自动评估报告
        "final_plan": None,
    }

    print("开始执行数据中心智能调度工作流...")
    print("初始状态:")
    for key, value in initial_state.items():
        if key not in ["messages", "raw_sensor_data_input", "power_purchase_contract_info", "calculated_kpis",
                       "compliance_redlines", "final_plan"]:  # 避免打印冗长字典
            print(f"  - {key}: {value}")
        elif key in ["raw_sensor_data_input", "power_purchase_contract_info"]:
            print(f"  - {key}: {json.dumps(value, ensure_ascii=False, indent=2)}")
    print("=" * 60)

    # --- 运行工作流并获取最终结果 ---
    print("\n" + "="*20 + " 开始运行Agent工作流 " + "="*20)
    # 使用 invoke 获取最终状态，以便生成PDF
    final_state = app.invoke(initial_state, {"recursion_limit": 100})
    print("="*20 + " Agent工作流运行结束 " + "="*20 + "\n")


    # --- 生成Markdown报告 ---
    try:
        from markdown_generator import save_plan_to_markdown

        md_filename = "零碳调度方案.md"
        print(f"--- 正在生成Markdown报告: {md_filename} ---")
        
        # 提取最终方案用于生成Markdown
        final_plan_data = final_state.get('final_plan')
        if final_plan_data:
            save_plan_to_markdown(final_plan_data, md_filename)
            print(f"--- ✅ Markdown报告已成功保存为 {md_filename} ---")
        else:
            print("--- ⚠️ 未在最终状态中找到'final_plan'，无法生成Markdown。 ---")

    except ImportError:
        print("--- ❌ 生成Markdown失败: 找不到 markdown_generator.py 文件。请确保它和主脚本在同一目录下。 ---")
    except Exception as e:
        print(f"--- ❌ 生成Markdown时发生未知错误: {e} ---")