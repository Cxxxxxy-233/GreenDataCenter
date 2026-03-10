
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# --- 任务 1 & 8: 高效数据读取与聚合 (集成 tqdm) ---
def process_large_csv(file_path, chunk_size=200000):
    """
    高效读取和处理大型 machine_usage.csv 文件。
    - 使用 chunksize 避免内存溢出。
    - 在每个块内进行聚合，而不是将整个文件加载到内存中。
    - 将秒级时间戳转换为 datetime 对象。
    - 按小时重采样并计算均值。
    - 提取一个典型的 24 小时周期。
    """
    print(f"🚀 开始处理大型文件: {file_path}")

    if not os.path.exists(file_path):
        print(f"⚠️ 警告: 文件 '{file_path}' 未找到。将生成模拟数据进行演示。")
        timestamps = pd.to_datetime(np.arange(86400 * 2) * 10, unit='s')
        cpu_usage = np.sin(np.linspace(0, 8 * np.pi, len(timestamps))) * 25 + 35
        simulated_df = pd.DataFrame({'timestamp': timestamps, 'cpu_usage': cpu_usage})
        simulated_df = simulated_df.set_index('timestamp')
        hourly_load = simulated_df['cpu_usage'].resample('H').mean().dropna()
    else:
        total_size = os.path.getsize(file_path)
        iterator = pd.read_csv(
            file_path,
            chunksize=chunk_size,
            header=None,
            usecols=[1, 2],
            names=['timestamp_s', 'cpu_usage']
        )

        hourly_aggregated_chunks = []
        print("✅ 文件读取与分块聚合开始...")
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="处理数据") as pbar:
            for chunk in iterator:
                chunk.dropna(inplace=True)
                if chunk.empty:
                    continue
                
                # 更新进度条（基于原始数据大小）
                pbar.update(chunk.memory_usage(index=True).sum())

                chunk['timestamp'] = pd.to_datetime(chunk['timestamp_s'], unit='s')
                chunk.set_index('timestamp', inplace=True)
                
                # 在块内进行小时级聚合
                hourly_chunk = chunk['cpu_usage'].resample('H').mean()
                if not hourly_chunk.empty:
                    hourly_aggregated_chunks.append(hourly_chunk)

        print("✅ 分块聚合完成，开始合并最终结果...")
        if not hourly_aggregated_chunks:
             raise ValueError("未能从文件中聚合任何数据。请检查文件内容和格式。")
        
        combined_hourly = pd.concat(hourly_aggregated_chunks)
        
        # 对合并后的结果再次聚合，以处理跨块的同一个小时的数据
        hourly_load = combined_hourly.groupby(combined_hourly.index).mean()

    # 提取一个典型的 24 小时周期 (例如，所有天的平均小时负载)
    if len(hourly_load) >= 24:
        # 为了确保周期性，我们按小时对所有数据进行平均
        daily_load_profile = hourly_load.groupby(hourly_load.index.hour).mean()
        # 确保索引是从 0 到 23
        daily_load_profile.index.name = 'hour'
        daily_load_profile = daily_load_profile.reindex(range(24), method='ffill').fillna(method='bfill')
    else:
        # 如果数据不足24小时，则进行填充
        print("⚠️ 警告: 聚合后数据不足24小时，将使用已有数据并填充。")
        daily_load_profile = hourly_load.reindex(pd.date_range(hourly_load.index.min(), periods=24, freq='H'), method='ffill')
        daily_load_profile = daily_load_profile.groupby(daily_load_profile.index.hour).mean()


    print("✅ 24 小时负载曲线生成完毕。")
    return daily_load_profile


# --- 任务 2: 物理模型 - 动态 PUE 计算 ---
def calculate_pue(load):
    """
    根据 IT 负载计算动态 PUE (Power Usage Effectiveness)。
    - PUE 基础值为 1.15。
    - 在 40%-60% 负载区间 PUE 最优。
    - 在低负载 (<20%) 和高负载 (>80%) 区间，PUE 线性惩罚至 1.4。
    """
    pue_min = 1.15
    pue_max = 1.4
    
    if load < 20:
        # 低负载区：PUE 从 1.4 (0%负载) 线性下降到 1.25 (20%负载)
        return pue_max - (load / 20) * (pue_max - 1.25)
    elif load <= 40:
        # 20%-40% 区间：PUE 从 1.25 线性下降到 1.15
        return 1.25 - ((load - 20) / 20) * (1.25 - pue_min)
    elif load <= 60:
        # 最佳效率区间
        return pue_min
    elif load <= 80:
        # 60%-80% 区间：PUE 从 1.15 线性上升到 1.25
        return pue_min + ((load - 60) / 20) * (1.25 - pue_min)
    else: # load > 80
        # 高负载区：PUE 从 1.25 线性上升到 1.4
        return 1.25 + ((load - 80) / 20) * (pue_max - 1.25)

# --- 任务 3: 物理模型 - 绿电出力模拟 ---
def generate_green_supply(hours=24, max_solar=80, base_wind=30):
    """
    生成模拟的 24 小时绿色电力供应曲线 (单位: MW)。
    - 光伏 (Solar): 采用正弦波模拟日照，峰值在 10:00-16:00。
    - 风电 (Wind): 稳定的基荷叠加随机波动。
    """
    hour_array = np.arange(hours)
    
    # 光伏出力 (正弦波模拟)
    # 峰值大约在 13:00 (np.pi/2), 范围从 7:00 到 19:00
    solar_power = np.sin((hour_array - 7) * np.pi / 12) * max_solar
    solar_power[solar_power < 0] = 0 # 夜间出力为0
    
    # 风电出力 (基荷 + 随机波动)
    wind_power = base_wind + np.random.normal(0, 5, hours)
    wind_power[wind_power < 0] = 0 # 确保风电不为负
    
    total_green_supply = solar_power + wind_power
    return total_green_supply

# --- 主函数，整合所有逻辑 ---
def create_dispatch_features():
    """
    主流程函数，增加了缓存检查机制：
    1. 检查 processed_metrics.csv 是否存在，若存在则直接加载。
    2. 若不存在，则执行完整的数据处理流程并创建该文件。
    """
    output_csv = 'processed_metrics.csv'
    
    # --- 缓存检查逻辑 ---
    if os.path.exists(output_csv):
        print(f"✅ 发现已缓存的特征文件，直接从 '{output_csv}' 加载。")
        return pd.read_csv(output_csv)
    
    print("⚠️ 未发现缓存的特征文件，开始执行完整的数据预处理流程...")
    
    # 任务 1: 获取 24 小时负载数据
    # 假设 IT 总负载容量为 200 MW, CPU 利用率等比例映射
    IT_CAPACITY_MW = 200
    avg_load_percent = process_large_csv('machine_usage.csv')
    avg_it_load_mw = avg_load_percent * (IT_CAPACITY_MW / 100)

    # 创建结果 DataFrame
    df = pd.DataFrame({
        'hour_index': np.arange(24),
        'avg_load_percent': avg_load_percent
    })
    df['avg_it_load_mw'] = avg_it_load_mw

    # 任务 2 & 4: 计算 PUE 和总负载
    df['dynamic_pue'] = df['avg_load_percent'].apply(calculate_pue)
    df['total_load_mw'] = df['avg_it_load_mw'] * df['dynamic_pue']

    # 任务 3: 生成绿电供应
    df['green_supply_mw'] = generate_green_supply(hours=24)

    # 任务 4: 计算 REF 指数
    df['ref_index'] = np.minimum(1.0, df['green_supply_mw'] / df['total_load_mw'])
    
    # 任务 5: 保存到 CSV
    # 确保 total_load_mw 被包含在最终输出中
    final_df = df[['hour_index', 'avg_it_load_mw', 'dynamic_pue', 'total_load_mw', 'green_supply_mw', 'ref_index']]
    final_df.columns = ['hour_index', 'avg_load', 'dynamic_pue', 'total_load_mw', 'green_supply_mw', 'ref_index'] # 重命名以匹配要求
    final_df.to_csv(output_csv, index=False)
    print(f"✅ 特征表已保存到: {output_csv}")

    # 任务 6: 生成可视化图表
    generate_visualization(final_df)
    
    return final_df

def generate_visualization(df):
    """使用 matplotlib 生成双 Y 轴图表"""
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Y 轴 1: IT 负载
    color = 'tab:blue'
    ax1.set_xlabel('小时 (Hour of Day)')
    ax1.set_ylabel('IT 负载 (MW)', color=color)
    ax1.plot(df['hour_index'], df['avg_load'], color=color, marker='o', linestyle='-', label='IT Load (MW)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Y 轴 2: REF 指数
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('可再生能源利用率 (REF)', color=color)
    ax2.plot(df['hour_index'], df['ref_index'], color=color, marker='x', linestyle='--', label='REF Index')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.1) # REF 指数范围 0-1

    # 图表标题和图例
    plt.title('数据中心24小时负载与可再生能源利用率（REF）', fontsize=16)
    fig.tight_layout()
    
    # 保存图表
    output_png = 'dispatch_preview.png'
    plt.savefig(output_png)
    print(f"✅ 可视化图表已保存到: {output_png}")
    plt.close()

# --- 任务 7: 接口封装 ---
# 使用一个全局变量来缓存数据，避免重复加载
_cached_df = None

def get_env_state(hour: int):
    """
    根据小时返回数据中心状态字典。
    如果数据未加载，则先执行主流程。
    """
    global _cached_df
    if _cached_df is None:
        print("首次调用，正在生成和缓存特征数据...")
        _cached_df = create_dispatch_features()

    if not 0 <= hour <= 23:
        raise ValueError("小时数必须在 0-23 之间。")
        
    state_data = _cached_df.iloc[hour]
    
    # 构造成符合您 LangGraph Agent 可能需要的字典结构
    return {
        'hour': int(state_data['hour_index']),
        'it_load_mw': state_data['avg_load'],
        'pue': state_data['dynamic_pue'],
        'total_load_mw': state_data['avg_load'] * state_data['dynamic_pue'],
        'green_supply_mw': state_data['green_supply_mw'],
        'ref_index': state_data['ref_index']
    }

# --- 主程序入口 ---
if __name__ == '__main__':
    print("===== 开始执行数据预处理与特征工程 =====")
    
    # 执行主流程
    final_features = create_dispatch_features()
    
    print("===== 任务完成 =====")
    print("生成的 24 小时特征数据预览:")
    print(final_features.head())
    
    print("===== 接口函数测试 =====")
    print("查询第 10 小时的环境状态:")
    hour_10_state = get_env_state(10)
    print(hour_10_state)
    
    print("查询第 18 小时的环境状态:")
    hour_18_state = get_env_state(18)
    print(hour_18_state)