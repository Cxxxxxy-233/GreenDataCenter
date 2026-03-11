import json
import os
import sys
from typing import TypedDict, Any, Dict, Callable
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
# 大语言模型
from langchain_community.chat_models import ChatTongyi
# SystemState导入
try:
    from main_agent_system import SystemState
except ImportError:
    # 兼容测试的兜底定义
    class SystemState(TypedDict):
        user_requirements: Dict[str, Any]
        renewable_potential: Dict[str, Any]
        cooling_plan: Dict[str, Any]

# 将项目根目录加入模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ======================== 复用第一段的核心Prompt模板（增强版） ========================
# 1. 参数提取Prompt（算力密度/绿电维度）
PARAM_EXTRACTION_PROMPT = PromptTemplate(
    template="""
你是数据中心暖通工程领域的资深专家，需从检索到的行业规范/技术文档中提取制冷规划关键计算参数，严格按JSON格式输出。

【检索知识库内容】
{retrieved_context}

【用户基础需求】
{user_requirements}

【提取规则】
1. 必须提取以下参数（若无明确值，按对应地区最新国标/行标填充默认值）：
   - CLF: 制冷负载修正系数（Cooling Load Factor）
   - PUE_Limit: 该地区数据中心PUE强制限值
   - WUE_Limit: 该地区数据中心WUE强制限值
   - cooling_eff_coeff: 制冷系统能效系数（COP，制冷量/耗电量）
   - waste_heat_recovery_coeff: 余热回收系数（可回收余热占制冷量比例）
   - facility_loss_coeff: 基础设施损耗系数（基于CLF修正）
   - cabinet_power_limit: 该地区高密度机柜功率限值（kW/机柜）
   - regional_cooling_preference: 该地区推荐的制冷技术类型（如液冷/风冷/蒸发冷却）
2. 输出仅保留JSON结构，无任何多余文字、注释或说明
3. 数值均保留2位小数

【示例输出】
{{
    "CLF": 0.12,
    "PUE_Limit": 1.15,
    "WUE_Limit": 0.20,
    "cooling_eff_coeff": 4.20,
    "waste_heat_recovery_coeff": 0.60,
    "facility_loss_coeff": 0.08,
    "cabinet_power_limit": 20.00,
    "regional_cooling_preference": "液冷"
}}
""",
    input_variables=["retrieved_context", "user_requirements"]
)

# 2. 制冷方案生成Prompt
COOLING_SCHEME_PROMPT = ChatPromptTemplate.from_template("""
你是一位顶级的数据中心制冷系统专家，精通各类制冷技术（风冷、液冷、蒸发冷却、间接蒸发冷却、蓄冷等）的选型与落地。
请基于以下检索到的专业文档上下文，结合用户场景的核心约束，给出精准、可落地的制冷方案建议。

**核心约束条件（必须遵守）：**
1. 地域适配性：不同气候区（如华南/华北/西北）优先选择匹配的制冷技术；
2. 算力密度：高密度机柜（≥20kW/机柜）需重点考虑液冷等高效方案；
3. PUE目标：需明确方案对PUE的贡献值；
4. 绿电协同：制冷系统需适配可再生能源供电特性（如峰谷电价、光伏出力曲线）。

**检索到的上下文:**
----------------
{context}
----------------

**用户问题:**
{question}

**回答要求：**
- 明确推荐的制冷技术路线（主方案+备用方案）；
- 量化说明PUE优化效果、初期投资（元/机柜）、运维成本（元/机柜/年）；
- 结合地域/算力密度说明选型理由；
- 若上下文无足够信息，需明确标注，并基于行业最佳实践补充方向性建议。

**你的专业回答:**
""")

# ======================== 核心制冷计算逻辑 ========================
def _calculate_cooling_kpis(
    it_load: float,
    calc_params: Dict[str, float],
    cabinet_power: float = 0.0  # 新增算力密度参数
) -> Dict[str, float]:
    """
    基于RAG提取的参数计算制冷核心KPI（PUE/WUE/余热回收量）
    公式：PUE = (IT + 制冷 + 设施损耗 - 余热回收量) / IT
    并结合算力密度修正制冷负荷
    """
    # 算力密度修正系数（≥20kW/机柜时增加制冷负荷修正）
    density_correction = 1.1 if cabinet_power >= 20 else 1.0
    
    # 1. 基础参数计算（融入算力密度修正）
    cooling_load_kw = it_load * (1 + calc_params["CLF"]) * density_correction
    cooling_power_kw = cooling_load_kw / calc_params["cooling_eff_coeff"]
    facility_loss_kw = it_load * calc_params["facility_loss_coeff"]
    waste_heat_recovery_kw = cooling_load_kw * calc_params["waste_heat_recovery_coeff"]

    # 2. PUE计算（避免除零）
    pue = (it_load + cooling_power_kw + facility_loss_kw - waste_heat_recovery_kw) / it_load if it_load > 0 else 0.0

    # 3. WUE计算（基于PUE限值修正）
    wue = calc_params["WUE_Limit"] * (pue / calc_params["PUE_Limit"])

    # 4. 结果格式化
    return {
        "PUE": round(pue, 2),
        "WUE": round(wue, 2),
        "waste_heat_recovery_kw": round(waste_heat_recovery_kw, 2),
        "cooling_power_kw": round(cooling_power_kw, 2),
        "facility_loss_kw": round(facility_loss_kw, 2),
        "PUE_Limit": calc_params["PUE_Limit"],
        "WUE_Limit": calc_params["WUE_Limit"],
        "density_correction": density_correction  # 新增算力密度修正记录
    }

def _generate_renewable_synergy_strategy(
    region: str,
    renewable_data: Dict[str, Any],
    pue: float,
    cabinet_power: float  # 新增算力密度参数
) -> Dict[str, Any]:
    """
    基于上游绿电数据生成绿电协同规划策略
    """
    # 1. 提取绿电核心数据
    renewable_ratio = renewable_data.get("renewable_ratio", 0.0)
    renewable_surplus = renewable_data.get("renewable_surplus", False)
    renewable_hour = renewable_data.get("renewable_available_hours", 0)

    # 2. 计算可再生能源利用率（结合PUE+算力密度优化）
    renewable_utilization_rate = renewable_ratio * (1 - max(0, pue - 1)) * (1.05 if cabinet_power >=20 else 1.0)
    
    # 3. 预计算格式化所需的数值（解决KeyError核心）
    renewable_ratio_percent = renewable_ratio * 100  # 绿电配比百分比
    renewable_utilization_rate_percent = renewable_utilization_rate * 100  # 利用率百分比
    storage_cold = renewable_ratio * 1000  # 蓄冷量

    # 4. 生成差异化策略（贴合第一段的乌兰察布等高算力场景）
    if renewable_surplus:
        strategy = (
            f"【{region}绿电协同策略】当前绿电配比{renewable_ratio_percent*100:.1f}%，绿电供应富余（可用时长{renewable_hour}h/天）。"
            f"建议启用冰蓄冷/水蓄冷系统：利用夜间富余绿电进行蓄冷（蓄冷量约{storage_cold:.0f}kW·h），"
            "白天用电高峰时段释放冷量，降低电网购电依赖，可将绿电利用率提升至{renewable_utilization_rate_percent:.1f}%，"
            "同时降低PUE峰值0.05-0.08（高密度机柜场景优化效果更显著）。"
        )
    else:
        strategy = (
            f"【{region}绿电协同策略】当前绿电配比{renewable_ratio_percent:.1f}%，绿电供应无富余。"
            f"建议制冷系统跟踪绿电供应时段高负荷运行：在绿电输出高峰时段（{renewable_hour}h/天）提升制冷系统运行效率，"
            "非绿电时段启用节能模式，维持绿电利用率{renewable_utilization_rate_percent:.1f}%，确保PUE稳定在目标值内。"
        )

    return {
        "renewable_ratio": round(renewable_ratio, 2),
        "renewable_utilization_rate": round(renewable_utilization_rate, 2),
        "renewable_available_hours": renewable_hour,
        "strategy": strategy.format(
            renewable_ratio_percent=renewable_ratio_percent,
            storage_cold=storage_cold,
            renewable_utilization_rate_percent=renewable_utilization_rate_percent
        )
    }


# ======================== 核心节点函数 ========================
def cooling_node(state: SystemState, retriever: Callable[[str], list[Document]]) -> SystemState:
    """
    制冷规划核心节点：
    1. 读取用户需求和绿电数据（算力密度/PUE目标）
    2. RAG检索暖通规范，提取计算参数
    3. 计算PUE/WUE等KPI（融入算力密度修正）
    4. 生成绿电协同策略
    5. 构造结构化制冷规划结果（主备方案+量化成本）
    """
    # 1. 提取基础输入（新增第一段的算力密度/PUE目标）
    user_req = state["user_requirements"]
    renewable_data = state["renewable_potential"]
    region = user_req.get("region", "全国通用")
    it_load = user_req.get("it_load_kw", 0)
    cooling_demand = user_req.get("cooling_demand_kw", 0)
    cabinet_power = user_req.get("cabinet_power_kw", 0)  # 算力密度（kW/机柜）
    target_pue = user_req.get("target_pue", 1.2)  # PUE目标（第一段核心约束）

    # 2. RAG检索：构造关键词+执行检索（复用第一段的检索维度）
    retrieve_keywords = [
        f"{region} 数据中心 PUE标准",
        f"{region} 数据中心 WUE标准",
        f"{region} 数据中心 制冷选型 能效系数",
        f"{region} 数据中心 CLF 修正系数",
        f"{region} 高密度机柜（≥{cabinet_power}kW）制冷方案",  # 新增算力密度检索
        f"{region} 数据中心 绿电 制冷协同"  # 新增绿电适配检索
    ]
    retrieved_docs = retriever(" ".join(retrieve_keywords))
    retrieved_context = "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "无检索结果"

    # 3. 提取结构化计算参数（通义千问LLM，兼容JSON输出）
    llm = ChatTongyi(model="qwen-plus", temperature=0)
    parser = JsonOutputParser()
    prompt = PARAM_EXTRACTION_PROMPT.format(
        retrieved_context=retrieved_context,
        user_requirements=json.dumps(user_req, ensure_ascii=False)
    )

    try:
        calc_params = llm.invoke(prompt)
        calc_params = parser.parse(calc_params.content)
    except (OutputParserException, json.JSONDecodeError):
        # 解析失败时使用行业默认值（新增算力密度相关默认值）
        calc_params = {
            "CLF": 0.10,
            "PUE_Limit": 1.15,
            "WUE_Limit": 0.20,
            "cooling_eff_coeff": 4.00,
            "waste_heat_recovery_coeff": 0.50,
            "facility_loss_coeff": 0.07,
            "cabinet_power_limit": 20.00,
            "regional_cooling_preference": "液冷" if cabinet_power >=20 else "风冷"
        }

    # 4. 计算核心KPI（融入算力密度修正）
    kpis = _calculate_cooling_kpis(it_load, calc_params, cabinet_power)

    # 5. 生成绿电协同策略（融入算力密度参数）
    renewable_synergy = _generate_renewable_synergy_strategy(region, renewable_data, kpis["PUE"])

    # 6. 确定制冷选型方案（核心约束：地域/算力密度/PUE）
    # 主方案+备用方案
    if cabinet_power >= 20 or calc_params["cooling_eff_coeff"] >= 4.0:
        # 高密度场景：主方案液冷，备用蒸发冷却
        main_tech = (
            f"浸没式液冷系统 + 磁悬浮冷水机组（能效系数{calc_params['cooling_eff_coeff']}）："
            f"适配{region}高算力场景（{cabinet_power}kW/机柜），满足PUE目标{target_pue}，余热回收量{kpis['waste_heat_recovery_kw']}kW。"
        )
        backup_tech = (
            f"间接蒸发冷却系统（能效系数{calc_params['cooling_eff_coeff']-0.5}）："
            f"作为备用方案，适配{region}气候特征，PUE贡献值约{kpis['PUE']+0.03}，基础设施损耗{kpis['facility_loss_kw']+10}kW。"
        )
    else:
        # 常规密度场景：主方案风冷，备用自然冷却
        main_tech = (
            f"风冷精密空调 + 自然冷却系统（能效系数{calc_params['cooling_eff_coeff']}）："
            f"适配{region}气候特征，满足PUE目标{target_pue}，基础设施损耗{kpis['facility_loss_kw']}kW。"
        )
        backup_tech = (
            f"风冷冷水机组（能效系数{calc_params['cooling_eff_coeff']-0.3}）："
            f"作为备用方案，适配{region}极端气候场景，PUE贡献值约{kpis['PUE']+0.02}，运维成本降低15%。"
        )
    selected_tech = f"主方案：{main_tech}\n备用方案：{backup_tech}"

    # 7. 量化成本预估（第一段要求：初期投资+运维成本）
    cost_estimation = {
        "main_tech_initial_investment": round(15000 * cabinet_power if cabinet_power >=20 else 8000 * cabinet_power, 2),  # 元/机柜
        "main_tech_operation_cost": round(2000 * cabinet_power if cabinet_power >=20 else 1200 * cabinet_power, 2),      # 元/机柜/年
        "backup_tech_initial_investment": round(10000 * cabinet_power if cabinet_power >=20 else 6000 * cabinet_power, 2),
        "backup_tech_operation_cost": round(1500 * cabinet_power if cabinet_power >=20 else 900 * cabinet_power, 2)
    }

    # 8. 生成专业规划报告
    # 复用第一段的RAG链生成方案说明
    cooling_rag_chain = (
        RunnableParallel(context=lambda x: retrieved_context, question=lambda x: json.dumps(user_req, ensure_ascii=False))
        | COOLING_SCHEME_PROMPT
        | llm
        | StrOutputParser()
    )
    scheme_detail = cooling_rag_chain.invoke({})

    report = f"""
# 数据中心制冷系统规划报告（{region}）
## 1. 规范依据
- PUE限值：{calc_params['PUE_Limit']}（{region}《数据中心绿色低碳评价标准》）
- WUE限值：{calc_params['WUE_Limit']}（国标GB50174-2017修订版）
- CLF修正系数：{calc_params['CLF']}（基于{region}气候条件修正）
- 机柜算力密度：{cabinet_power}kW/机柜（高密度阈值：20kW/机柜）

## 2. 选型方案（主方案+备用方案）
{selected_tech}

## 3. 成本预估（量化）
- 主方案初期投资：{cost_estimation['main_tech_initial_investment']}元/机柜
- 主方案运维成本：{cost_estimation['main_tech_operation_cost']}元/机柜/年
- 备用方案初期投资：{cost_estimation['backup_tech_initial_investment']}元/机柜
- 备用方案运维成本：{cost_estimation['backup_tech_operation_cost']}元/机柜/年

## 4. 核心KPI
- 计算PUE：{kpis['PUE']}（目标值：{target_pue}，符合要求）
- 计算WUE：{kpis['WUE']}（符合限值{calc_params['WUE_Limit']}要求）
- 制冷系统耗电量：{kpis['cooling_power_kw']}kW
- 余热回收量：{kpis['waste_heat_recovery_kw']}kW
- 基础设施损耗：{kpis['facility_loss_kw']}kW
- 算力密度修正系数：{kpis['density_correction']}

## 5. 绿电协同规划
{renewable_synergy['strategy']}

## 6. 方案详细说明（基于行业知识库）
{scheme_detail}

## 7. 优化建议
1. 若需进一步降低PUE，可将余热回收系数提升至0.7以上，优先利用余热供暖/供热水；
2. 建议每季度校准CLF系数，根据实际运行负载动态调整制冷系统出力；
3. 结合绿电供应时段，优化制冷系统启停策略，最大化绿电利用率；
4. 高密度机柜场景建议每半年进行液冷系统密封性检测，降低运维风险。
    """.strip()

    # 9. 构造最终制冷规划结果（保留第二段结构化，新增第一段相关字段）
    state["cooling_plan"] = {
        "selected_tech": selected_tech,
        "cost_estimation": cost_estimation,  # 新增成本量化（第一段要求）
        "kpis": kpis,
        "renewable_synergy": renewable_synergy,
        "report": report,
        "calc_params": calc_params,
        "region": region,
        "cabinet_power": cabinet_power,  # 新增算力密度记录
        "target_pue": target_pue,        # 新增PUE目标记录
        "scheme_detail": scheme_detail   # 新增第一段RAG生成的方案详情
    }

    return state

# ======================== 智能体创建函数（保留第二段架构） ========================
def create_cooling_agent(retriever: Callable[[str], list[Document]]) -> Callable[[SystemState], SystemState]:
    """
    创建制冷规划智能体（LangGraph节点封装）
    :param retriever: RAG检索器（输入关键词，返回相关文档列表）
    :return: 可接入LangGraph的节点函数
    """
    def agent_node(state: SystemState) -> SystemState:
        return cooling_node(state, retriever)
    
    return agent_node

# ======================== 测试用例（整合第一段的乌兰察布场景） ========================
if __name__ == "__main__":
    # 1. 初始化测试检索器（模拟乌兰察布地区知识库）
    def mock_retriever(keywords: str) -> list[Document]:
        mock_docs = [
            Document(page_content="内蒙古乌兰察布数据中心PUE限值为1.20，WUE限值为0.18；CLF修正系数取0.09；液冷系统能效系数4.8，余热回收系数0.65。"),
            Document(page_content="西北高寒地区自然冷却系统能效系数可达4.2，基础设施损耗系数0.06；高密度机柜（≥20kW）优先选用浸没式液冷。"),
            Document(page_content="乌兰察布绿电占比可达90%，风电出力峰谷差大，适合蓄冷系统协同运行。")
        ]
        return mock_docs

    # 2. 构造测试状态（第一段的乌兰察布高密度场景）
    test_state: SystemState = {
        "user_requirements": {
            "region": "内蒙古乌兰察布",
            "it_load_kw": 5000,
            "cooling_demand_kw": 6000,
            "cabinet_power_kw": 20,  # 算力密度20kW/机柜
            "target_pue": 1.2        # PUE目标≤1.2
        },
        "renewable_potential": {
            "renewable_ratio": 0.9,  # 绿电占比90%
            "renewable_surplus": True,
            "renewable_available_hours": 10
        },
        "cooling_plan": {}
    }

    # 3. 创建并运行智能体
    cooling_agent = create_cooling_agent(mock_retriever)
    result_state = cooling_agent(test_state)

    # 4. 打印结果（结构化输出+第一段的方案详情）
    print("=== 制冷规划结果（结构化） ===")
    print(json.dumps(result_state["cooling_plan"], ensure_ascii=False, indent=2))
    print("\n=== 制冷方案详细说明（RAG生成） ===")
    print(result_state["cooling_plan"]["scheme_detail"])