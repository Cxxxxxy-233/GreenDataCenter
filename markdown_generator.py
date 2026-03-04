import json
import pandas as pd

def save_plan_to_markdown(plan_data, filename="零碳调度方案.md"):
    """
    将方案字典保存为Markdown文件。
    """
    markdown_content = "# 零碳数据中心全方位调度方案\n\n"

    # 1. 首席架构师建议摘要
    markdown_content += "## 1. 首席架构师建议摘要\n"
    recommendations = plan_data.get("chief_architect_recommendations", "无建议。")
    markdown_content += f"{recommendations}\n\n"

    # 2. 数据中心负载预测
    markdown_content += "## 2. 数据中心负载预测 (未来24小时)\n"
    load_forecast = plan_data.get("load_forecast_24h")
    if isinstance(load_forecast, list) and load_forecast:
        load_df = pd.DataFrame(load_forecast)
        markdown_content += "```\n"
        markdown_content += load_df.to_string()
        markdown_content += "\n```\n\n"
    else:
        markdown_content += "无负载预测数据。\n\n"

    # 3. 风光出力预测
    markdown_content += "## 3. 风光电站出力预测 (未来24小时)\n"
    renewable_forecast = plan_data.get("renewable_energy_forecast_24h")
    if isinstance(renewable_forecast, list) and renewable_forecast:
        renewable_df = pd.DataFrame(renewable_forecast)
        markdown_content += "```\n"
        markdown_content += renewable_df.to_string()
        markdown_content += "\n```\n\n"
    else:
        markdown_content += "无风光出力预测数据。\n\n"

    # 4. 初始条件
    markdown_content += "## 4. 方案初始条件\n"
    initial_conditions = plan_data.get("initial_conditions", {})
    conditions_str = json.dumps(initial_conditions, ensure_ascii=False, indent=2)
    markdown_content += "```json\n"
    markdown_content += conditions_str
    markdown_content += "\n```\n\n"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(markdown_content)
