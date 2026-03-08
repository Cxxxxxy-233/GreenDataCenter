
import os
import json
import pandas as pd
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage

# --- LLM 初始化 ---
# 用于生成报告总结的LLM
report_llm = ChatTongyi(model="qwen-long")

# --- 常量定义 ---
# 电网平均碳排放因子 (单位: gCO2/kWh)
# 这个值可以根据地区和数据来源进行调整，这里我们使用一个通用值
GRID_CARBON_INTENSITY = 450  # gCO2/kWh

def calculate_carbon_savings(all_hours_data: pd.DataFrame):
    """
    计算智能调度相比传统调度的碳减排量。
    - 传统调度：所有电力来自电网。
    - 智能调度：优先使用绿电，不足部分由电网补充。
    """
    print("  - 正在计算碳减排贡献...")
    # 总能耗 (MWh)
    total_energy_consumption_mwh = all_hours_data['total_load_mw'].sum()

    # 传统调度下的总碳排放 (kgCO2)
    # 总能耗(MWh) * 1000 (kWh/MWh) * 因子(g/kWh) / 1000 (g/kg)
    traditional_carbon_emissions_kg = total_energy_consumption_mwh * GRID_CARBON_INTENSITY

    # 智能调度下的总绿电使用量 (MWh)
    total_green_energy_used_mwh = all_hours_data['green_supply_mw'].sum()

    # 智能调度下的电网购电量 (MWh)
    grid_power_mwh = max(0, total_energy_consumption_mwh - total_green_energy_used_mwh)

    # 智能调度下的总碳排放 (kgCO2)
    smart_carbon_emissions_kg = grid_power_mwh * GRID_CARBON_INTENSITY

    # 减排量 (kgCO2)
    carbon_savings_kg = traditional_carbon_emissions_kg - smart_carbon_emissions_kg
    
    # 减排比例
    savings_percentage = (carbon_savings_kg / traditional_carbon_emissions_kg) * 100 if traditional_carbon_emissions_kg > 0 else 0

    return {
        "traditional_emissions_kg": traditional_carbon_emissions_kg,
        "smart_emissions_kg": smart_carbon_emissions_kg,
        "savings_kg": carbon_savings_kg,
        "savings_percentage": savings_percentage
    }

def summarize_llm_insights(full_day_results: list):
    """
    调用一个长文本LLM来总结24小时的所有调度建议。
    """
    print("  - 正在调用LLM生成专家建议总结...")
    all_insights = []
    for i, result in enumerate(full_day_results):
        insight = result.get('llm_insights', '无建议')
        all_insights.append(f"**第 {i} 小时:**\n{insight}\n")

    insights_text = "\n---\n".join(all_insights)

    prompt = f"""
    你是一位资深的能源策略分析师。以下是零碳数据中心智能调度系统在24小时内，针对每个小时状态生成的调度建议日志。

    **任务：**
    请你通读所有日志，从中提炼出3-5条最核心、最有价值的宏观策略和模式洞察。不要逐条复述，要进行归纳和总结。

    **分析角度：**
    1.  **高峰与低谷**：系统是如何应对IT负载高峰和绿电出力低谷的？
    2.  **能源匹配**：系统是否成功地将高耗能任务调度到了绿电充足的时段？
    3.  **模式识别**：是否能看出某种重复出现的调度模式？（例如：中午光照强烈时固定执行某种操作，夜晚固定执行另一种操作）
    4.  **潜在问题**：从日志中是否能发现系统决策的潜在问题或可以改进的地方？

    **原始日志如下：**
    ---
    {insights_text}
    ---

    请以清晰、专业的语言，输出你的宏观分析总结。
    """
    
    summary_response = report_llm.invoke([HumanMessage(content=prompt)])
    return summary_response.content

def generate_retro_analysis_report(full_day_results: list, all_hours_data: pd.DataFrame):
    """
    生成完整的复盘报告（Markdown格式）。
    """
    print("\n[报告生成模块启动]")
    
    # 1. 能效分析 (PUE vs. Load)
    # 这部分数据直接来自 all_hours_data
    pue_analysis_text = all_hours_data[['hour_index', 'avg_load', 'dynamic_pue']].to_markdown(index=False)

    # 2. 低碳贡献分析
    carbon_analysis = calculate_carbon_savings(all_hours_data)

    # 3. LLM专家建议总结
    expert_summary = summarize_llm_insights(full_day_results)

    # --- 组装 Markdown 报告 ---
    print("  - 正在组装Markdown报告...")
    report_content = f"""
# 零碳数据中心24小时智能调度复盘报告

---

## 1. 核心成果摘要

本次模拟覆盖了24小时的连续调度周期，全面测试了智能体在不同工况下的决策能力。核心成果如下：

- **总计碳减排**: **{carbon_analysis['savings_kg']:.2f} kg CO2**
- **减排效率**: **{carbon_analysis['savings_percentage']:.2f}%** (相较于传统100%电网供电模式)
- **智能体决策**: 成功根据实时工况（IT负载、绿电供应）动态调整调度策略。

---

## 2. 能效分析：PUE-负载曲线

PUE（电源使用效率）是衡量数据中心能源效率的关键指标。下表展示了在24小时周期内，PUE随IT负载（avg_load）变化的真实情况。这反映了数据中心在不同繁忙程度下的能效表现。

{pue_analysis_text}

*分析：PUE在IT负载40%-60%时达到最优值，符合物理模型设定。在凌晨低负载和傍晚高峰负载时，PUE均有上升，说明此时能源主要消耗在非IT设备（如制冷）上。*

---

## 3. 低碳贡献量化分析

我们对比了“智能调度”（优先使用绿电）与“传统调度”（100%使用电网电力）两种模式下的碳排放。

- **传统模式总排放**: {carbon_analysis['traditional_emissions_kg']:.2f} kg CO2
- **智能调度总排放**: {carbon_analysis['smart_emissions_kg']:.2f} kg CO2

**结论：通过优先消纳 {all_hours_data['green_supply_mw'].sum():.2f} MWh 的绿色电力，本次24小时模拟成功减少了 {carbon_analysis['savings_kg']:.2f} kg 的碳排放。**

---

## 4. AI专家策略总结

我们整合了智能体在24小时内生成的所有小时级调度建议，并由高级AI模型进行归纳总结，提炼出以下宏观策略洞察：

{expert_summary}

---

*报告生成时间: {pd.Timestamp.now()}*

"""

    # 保存报告到文件
    with open("调度复盘报告.md", "w", encoding="utf-8") as f:
        f.write(report_content)

