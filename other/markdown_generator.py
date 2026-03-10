import json
import pandas as pd

def save_plan_to_markdown(state_data, filename="零碳调度方案.md"):
    """
    将完整的state字典保存为结构化的Markdown文件，包含所有关键节点的分析。
    """
    markdown_content = "# 零碳数据中心全方位调度方案\n\n"

    # 1. 初步分析报告
    markdown_content += "## 1. 初步分析报告\n\n"
    initial_analysis = state_data.get("analysis_result", "无初步分析报告。")
    markdown_content += f"{initial_analysis}\n\n"

    # 2. 首席架构师建议
    markdown_content += "## 2. 首席架构师建议\n\n"
    recommendations = state_data.get("llm_insights", "无首席架构师建议。")
    markdown_content += f"{recommendations}\n\n"

    # 3. 专家评估报告
    markdown_content += "## 3. 专家评估报告\n\n"
    evaluation = state_data.get("evaluation_report", "无评估报告。")
    markdown_content += f"{evaluation}\n\n"

    # 4. 附录：详细预测数据
    markdown_content += "---\n## 附录: 详细预测数据\n\n"

    # 4.1 数据中心负载预测
    markdown_content += "### 4.1 数据中心负载预测 (未来24小时)\n\n"
    load_forecast_df = state_data.get("load_prediction_results")
    if load_forecast_df is not None and not load_forecast_df.empty:
        markdown_content += "```\n"
        markdown_content += load_forecast_df.to_string()
        markdown_content += "\n```\n\n"
    else:
        markdown_content += "无负载预测数据。\n\n"

    # 4.2 风光出力预测
    markdown_content += "### 4.2 风光电站出力预测 (未来24小时)\n\n"
    renewable_forecast_df = state_data.get("renewable_prediction_results")
    if renewable_forecast_df is not None and not renewable_forecast_df.empty:
        markdown_content += "```\n"
        markdown_content += renewable_forecast_df.to_string()
        markdown_content += "\n```\n\n"
    else:
        markdown_content += "无风光出力预测数据。\n\n"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(markdown_content)
