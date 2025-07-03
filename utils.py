"""
此模块提供了一系列辅助函数，用于支持车载边缘计算（VEC）仿真项目中的计算、
数据处理和算法实现。

其功能主要可以分为以下几类：

1.  **性能指标计算函数:**
    基于论文中的公式，计算各种性能指标：
    - `calculate_transmission_rate`: 计算数据传输速率 (Eq 1)。
    - `calculate_transmission_delay`: 计算数据传输延迟 (Eq 2)。
    - `calculate_transmission_energy`: 计算数据传输能耗 (Eq 3)。
    - `calculate_local_compute_delay`: 计算任务在车辆本地执行的延迟 (Eq 4)。
    - `calculate_local_compute_energy`: 计算任务在车辆本地执行的能耗 (Eq 5)。
    - `calculate_rsu_compute_delay`: 计算子任务在RSU上执行的延迟 (Eq 9, 部分)。
    - `calculate_rsu_compute_energy`: 计算子任务在RSU上执行的能耗（非论文明确公式，为完整性补充）。
    - `calculate_objective`: 计算组合目标函数值，综合考虑延迟和能耗。

2.  **A-LSH (Adaptive Locality Sensitive Hashing) 相关函数:**
    - `initialize_lsh_params`: 初始化LSH所需的随机投影向量和偏移量。
    - `compute_lsh_hash`: 根据子任务的特征向量和RSU的动态桶宽 `dj` 计算其LSH哈希值 (Eq 7)。
      此哈希值用于RSU缓存中的快速相似性查找，以实现计算结果的复用。

3.  **环境与配置相关辅助函数:**
    - `get_noise_power`: 根据带宽计算信道噪声功率。
    - `get_vehicle_index`, `get_rsu_index`: 从ID字符串中提取车辆或RSU的数字索引。

4.  **专家策略生成函数:**
    - `generate_expert_policy_heuristic`: 生成一个基于启发式的“专家”卸载决策。
      这个启发式策略会估算任务在本地执行和在各个RSU上执行的成本（延迟和能耗），
      然后使用带温度系数的softmax选择一个概率性的决策。
      生成的专家数据可用于训练模仿学习模型。

这些工具函数被项目中的其他模块广泛使用，例如 `algorithms.py` 中的算法实现、
`environment.py` 中的环境模拟，以及 `main.py` 中的仿真流程。
它们依赖于 `config.py` 中定义的参数和 `task.py`、`environment.py` 中定义的数据结构。
"""
# utils.py
import numpy as np
import torch
import config
from task import TaskDAG, SubTask
import environment # <-- IMPORT ADDED HERE
import traceback # For printing errors in heuristic

# --- Calculation Functions (Based on Paper Equations) ---

def calculate_transmission_rate(vehicle_power, channel_gain, bandwidth, noise_power):
    """
    根据香农公式计算数据传输速率 (对应论文中的 Eq 1)。

    传输速率 R = B * log2(1 + SNR)，其中 SNR (信噪比) = (P * h) / N0。
    - B 是带宽 (`bandwidth`)。
    - P 是车辆的传输功率 (`vehicle_power`)。
    - h 是信道增益 (`channel_gain`)。
    - N0 是噪声功率 (`noise_power`)。

    参数:
        vehicle_power (float): 车辆的传输功率 (单位: Watts)。
        channel_gain (float): 车辆与目标RSU之间的信道增益 (无单位)。
        bandwidth (float): 分配用于传输的信道带宽 (单位: Hz)。
        noise_power (float): 信道中的噪声功率 (单位: Watts)。

    返回:
        float: 计算得到的数据传输速率 (单位: bps, bits per second)。
               如果带宽、信道增益或噪声功率无效（例如小于等于0），
               或者计算出的SNR导致log2的参数无效，则返回0。
               确保返回的速率是非负的。
    """
    if bandwidth <= 0 or channel_gain <= 0 or noise_power <= 1e-21: # Use epsilon for noise_power
        return 0

    # --- FIX: Calculate signal_power ---
    signal_power = vehicle_power * channel_gain
    # --- END FIX ---

    # Prevent division by zero or negative SNR argument
    if noise_power <= 0: return 0 # Already checked above, but double-check
    snr = signal_power / noise_power

    # Shannon formula: BW * log2(1 + SNR)
    # Ensure argument to log2 is positive
    if (1 + snr) <= 0:
        return 0
    try:
        # log2 can return NaN if argument is very slightly negative due to float issues
        rate = bandwidth * np.log2(max(1e-9, 1 + snr)) # Use max to prevent log2(<=0)
    except ValueError:
        rate = 0
    return max(0, rate) # Ensure rate is non-negative

def calculate_transmission_delay(data_size, trans_rate):
    """
    计算数据传输延迟 (对应论文中的 Eq 2)。

    传输延迟 T_trans = D / R，其中 D 是数据大小，R 是传输速率。

    参数:
        data_size (float): 要传输的数据的总大小 (单位: bits)。
        trans_rate (float): 数据传输速率 (单位: bps)。

    返回:
        float: 计算得到的传输延迟 (单位: seconds)。
               如果传输速率 `trans_rate` 非常小或为0，则返回 `float('inf')`
               以表示传输不可能或需要无限长时间。
    """
    if trans_rate <= 1e-9: # Use a small epsilon instead of == 0 for float comparison
        return float('inf') # Avoid division by zero or near-zero
    return data_size / trans_rate

def calculate_transmission_energy(vehicle_power, trans_delay):
    """
    计算数据传输能耗 (对应论文中的 Eq 3)。

    传输能耗 E_trans = P_trans * T_trans，其中 P_trans 是车辆的传输功率，
    T_trans 是传输延迟。

    参数:
        vehicle_power (float): 车辆在传输数据时的功率消耗 (单位: Watts)。
        trans_delay (float): 数据传输的延迟时间 (单位: seconds)。

    返回:
        float: 计算得到的传输能耗 (单位: Joules)。
               如果传输延迟 `trans_delay` 是无限的 (`float('inf')`)，
               则能耗也返回 `float('inf')`。
    """
    if np.isinf(trans_delay): # If delay is infinite, energy is also infinite (or max)
        return float('inf')
    return vehicle_power * trans_delay

def calculate_local_compute_delay(subtask: SubTask, vehicle_cpu_freq):
    """
    计算子任务在车辆本地CPU上执行的延迟 (对应论文中的 Eq 4)。

    本地计算延迟 T_local = C_s / f_local，其中 C_s 是子任务的CPU周期数，
    f_local 是车辆的CPU频率。

    参数:
        subtask (SubTask): 要计算延迟的子任务对象。
                           需要其 `cpu_cycles` 属性。
        vehicle_cpu_freq (float): 车辆本地CPU的处理频率 (单位: Hz)。

    返回:
        float: 计算得到的本地计算延迟 (单位: seconds)。
               如果车辆CPU频率 `vehicle_cpu_freq` 非常小或为0，则返回 `float('inf')`。
               如果子任务的CPU周期数 `subtask.cpu_cycles` 为0或负数，则返回0。
    """
    if vehicle_cpu_freq <= 1e-9:
        return float('inf')
    # Ensure cpu_cycles is positive
    if subtask.cpu_cycles <=0: return 0 # Zero cycles takes zero time
    return subtask.cpu_cycles / vehicle_cpu_freq

def calculate_local_compute_energy(vehicle_power_compute, local_delay):
    """
    计算子任务在车辆本地CPU上执行的能耗 (对应论文中的 Eq 5)。

    本地计算能耗 E_local = P_compute * T_local，其中 P_compute 是车辆的计算功率，
    T_local 是本地计算延迟。

    参数:
        vehicle_power_compute (float): 车辆在本地执行计算时的功率消耗 (单位: Watts)。
        local_delay (float): 子任务的本地计算延迟 (单位: seconds)。

    返回:
        float: 计算得到的本地计算能耗 (单位: Joules)。
               如果本地计算延迟 `local_delay` 是无限的 (`float('inf')`)，
               则能耗也返回 `float('inf')`。
    """
    if np.isinf(local_delay):
        return float('inf')
    return vehicle_power_compute * local_delay

def calculate_rsu_compute_delay(subtask: SubTask, allocated_rsu_freq, is_reused):
    """
    计算子任务在RSU上执行的计算延迟 (对应论文中 Eq 9 的计算部分)。

    RSU计算延迟 T_rsu_comp = (1 - ε) * C_s / f_rsu_alloc，其中:
    - ε (`epsilon`) 是一个指示符，如果子任务结果被复用 (`is_reused` 为 True)，则为1，否则为0。
    - C_s 是子任务的CPU周期数 (`subtask.cpu_cycles`)。
    - f_rsu_alloc 是分配给该子任务的RSU的CPU频率 (`allocated_rsu_freq`)。

    参数:
        subtask (SubTask): 要计算延迟的子任务对象。需要其 `cpu_cycles` 属性。
        allocated_rsu_freq (float): 分配给此子任务的RSU的CPU处理频率 (单位: Hz)。
        is_reused (bool): 一个布尔标志，指示此子任务的结果是否从缓存中复用。
                          如果为 `True`，则计算延迟为0。

    返回:
        float: 计算得到的RSU计算延迟 (单位: seconds)。
               如果 `is_reused` 为 `True`，返回0。
               如果分配的RSU频率 `allocated_rsu_freq` 非常小或为0，返回 `float('inf')`。
               如果子任务的CPU周期数 `subtask.cpu_cycles` 为0或负数，且未复用，则返回0。
    """
    if is_reused:
        return 0 # Reused tasks have negligible computation delay on RSU side
    if allocated_rsu_freq <= 1e-9:
        return float('inf')
    if subtask.cpu_cycles <= 0: return 0 # Zero cycles takes zero time

    epsilon = 1 if is_reused else 0
    return (1 - epsilon) * subtask.cpu_cycles / allocated_rsu_freq

def calculate_rsu_compute_energy(rsu_power_per_cycle, cpu_cycles, is_reused):
    """
    估算子任务在RSU上执行的计算能耗。

    注意：此能耗计算模型在原始论文中可能没有明确对应公式，是为仿真完整性添加的。
    它基于一个简化的假设：能耗与执行的CPU周期数成正比，比例因子为一个假设的
    每周期能耗值 (`assumed_energy_per_cycle`)。

    如果子任务结果被复用 (`is_reused` 为 True)，则计算能耗为0。

    参数:
        rsu_power_per_cycle (float): 此参数当前未在函数体中使用，但保留可能是为了
                                     未来更复杂的RSU能耗模型。
        cpu_cycles (float): 子任务执行所需的CPU周期数。
        is_reused (bool): 一个布尔标志，指示此子任务的结果是否从缓存中复用。

    返回:
        float: 估算得到的RSU计算能耗 (单位: Joules)。
               如果 `is_reused` 为 `True`，或 `cpu_cycles` 小于等于0，则返回0。
    """
    if is_reused:
        return 0
    if cpu_cycles <= 0: return 0
    assumed_energy_per_cycle = 1e-9 # Joules/cycle (EXAMPLE VALUE)
    return cpu_cycles * assumed_energy_per_cycle


# --- Noise Power Calculation Helper ---
def get_noise_power(bandwidth, noise_density=config.NOISE_POWER_SPECTRAL_DENSITY):
    """
    根据给定的带宽和噪声功率谱密度计算信道中的噪声功率。

    噪声功率 N0 = n0 * B，其中 n0 是噪声功率谱密度，B 是带宽。

    参数:
        bandwidth (float): 信道带宽 (单位: Hz)。
        noise_density (float, optional): 噪声功率谱密度 (单位: W/Hz)。
                                         默认为 `config.NOISE_POWER_SPECTRAL_DENSITY`。

    返回:
        float: 计算得到的噪声功率 (单位: Watts)。
               确保返回值为正，且有一个最小下限 (1e-21) 以避免计算问题。
               如果 `noise_density` 小于等于0，则直接返回最小下限。
    """
    if noise_density <= 0:
        return 1e-21
    noise = noise_density * bandwidth
    return max(1e-21, noise) # Ensure noise is positive


# --- A-LSH Hashing (Eq 7) ---
LSH_PARAMS = {}

def initialize_lsh_params(num_tables=config.NUM_HASH_TABLES, num_funcs=config.NUM_HASH_FUNCTIONS_PER_TABLE, feature_dim=config.FEATURE_DIM):
    """
    初始化用于LSH（Locality Sensitive Hashing）的随机投影参数。

    LSH通过将高维特征向量投影到低维空间并进行分桶来实现快速近似最近邻搜索。
    此函数为多个哈希表（每个表有多个哈希函数）生成所需的随机参数：
    - `A`: 随机投影向量矩阵。对于每个哈希表，`A` 的形状是 `(num_funcs, feature_dim)`。
    - `B`: 随机偏移量向量。对于每个哈希表，`B` 的形状是 `(num_funcs)`。

    这些参数存储在全局字典 `LSH_PARAMS` 中。如果 `LSH_PARAMS` 已被初始化，
    则此函数不执行任何操作。

    参数:
        num_tables (int, optional): 要创建的LSH哈希表的数量。
                                    默认为 `config.NUM_HASH_TABLES`。
        num_funcs (int, optional): 每个哈希表中的哈希函数（即投影向量）的数量。
                                   默认为 `config.NUM_HASH_FUNCTIONS_PER_TABLE`。
        feature_dim (int, optional): 输入特征向量的维度。
                                     默认为 `config.FEATURE_DIM`。

    副作用:
        - 如果 `LSH_PARAMS` 为空，则会填充它。
          `LSH_PARAMS['A']` 将成为一个列表，其中每个元素是一个NumPy数组 (A_table)。
          `LSH_PARAMS['B']` 将成为一个列表，其中每个元素是一个NumPy数组 (B_table)。
        - 打印LSH参数初始化的日志信息。
    """
    global LSH_PARAMS
    if LSH_PARAMS:
        return
    LSH_PARAMS['A'] = []
    LSH_PARAMS['B'] = []
    print(f"Initializing LSH params: {num_tables} tables, {num_funcs} funcs/table, dim={feature_dim}")
    for _ in range(num_tables):
        A_table = np.random.randn(num_funcs, feature_dim)
        B_table = np.random.rand(num_funcs)
        LSH_PARAMS['A'].append(A_table)
        LSH_PARAMS['B'].append(B_table)

# String hint 'environment.RSU' relies on environment being imported
def compute_lsh_hash(feature_vector, rsu: 'environment.RSU', table_index): # <--- Uses string hint
    """
    为给定的特征向量计算其在指定LSH表中的哈希值 (对应论文中的 Eq 7)。

    哈希值的计算公式为 `h(v) = floor((A * v + B) / dj)`，其中：
    - `v` 是输入的 `feature_vector`。
    - `A` 和 `B` 是LSH表 `table_index` 对应的随机投影矩阵和偏移向量。
    - `dj` 是RSU的自适应桶宽度，通过调用 `rsu.get_adaptive_bucket_width()` 获得。

    此函数首先确保LSH参数已初始化。然后，它根据指定的 `table_index`
    获取相应的 `A` 和 `B` 参数，并从RSU对象获取当前的 `dj`。
    接着执行投影、加偏移、除以桶宽和向下取整操作，得到一组哈希分量。
    这些分量最终组成一个元组作为该特征向量在该LSH表中的哈希键。

    参数:
        feature_vector (np.ndarray or list): 要计算哈希值的子任务特征向量。
                                             应为一维数组或可转换为一维数组的列表。
        rsu (environment.RSU): 目标RSU对象。需要其 `get_adaptive_bucket_width` 方法
                               来获取当前的桶宽度 `dj`。
        table_index (int): 要使用的LSH哈希表的索引（从0开始）。

    返回:
        tuple: 一个由整数组成的元组，代表特征向量在指定LSH表中的哈希值（哈希桶的标识）。
               元组的长度等于该LSH表中的哈希函数数量。

    异常:
        ValueError: 如果 `table_index` 无效，或者特征向量与LSH参数的维度不匹配。
        TypeError: 如果传入的 `rsu` 对象没有 `get_adaptive_bucket_width` 方法。
    """
    if not LSH_PARAMS:
        initialize_lsh_params()

    if table_index >= len(LSH_PARAMS['A']):
        raise ValueError(f"Invalid LSH table index {table_index}, only {len(LSH_PARAMS['A'])} tables initialized.")

    A = LSH_PARAMS['A'][table_index]
    B = LSH_PARAMS['B'][table_index]

    if not hasattr(rsu, 'get_adaptive_bucket_width'):
        raise TypeError(f"Passed RSU object (ID: {getattr(rsu, 'id', 'N/A')}) does not have method 'get_adaptive_bucket_width'")

    dj = rsu.get_adaptive_bucket_width()
    if dj <= 1e-9:
        print(f"Warning: Bucket width dj is near zero ({dj:.2e}) for RSU {rsu.id}. Using default large hash components.")
        hash_components = np.full(A.shape[0], fill_value=np.iinfo(np.int32).max >> 1)
    else:
        if isinstance(feature_vector, list):
            feature_vector = np.array(feature_vector)
        if feature_vector.ndim > 1:
             feature_vector = feature_vector.flatten()
        if A.shape[1] != feature_vector.shape[0]:
             raise ValueError(f"Dimension mismatch for LSH: A shape {A.shape}, feature_vector shape {feature_vector.shape}")

        projection = A.dot(feature_vector)
        scaled_projection = (projection + B) / dj
        hash_components = np.floor(scaled_projection).astype(np.int64)

    final_hash = tuple(hash_components)
    return final_hash


# --- Helper Functions ---
def calculate_objective(latency, energy, alpha=config.ALPHA, beta=config.BETA):
    """
    计算组合的目标函数值，通常是延迟和能耗的加权和。

    目标函数旨在量化任务执行的“成本”，其中较低的值表示更好的性能。
    公式为: Objective = α * Latency + β * Energy。
    权重 α (`alpha`) 和 β (`beta`) 控制了延迟和能耗在总成本中的相对重要性。

    参数:
        latency (float): 任务执行的总延迟 (单位: seconds)。
        energy (float): 任务执行的总能耗 (单位: Joules)。
        alpha (float, optional): 延迟的权重因子。默认为 `config.ALPHA`。
        beta (float, optional): 能耗的权重因子。默认为 `config.BETA`。

    返回:
        float: 计算得到的组合目标函数值。
               如果输入的 `latency` 或 `energy` 是无限的 (`float('inf')`)，
               则目标值也返回 `float('inf')`。
               在计算前，确保延迟和能耗值非负。
    """
    if np.isinf(latency) or np.isinf(energy):
        return float('inf')
    # Ensure non-negative components before calculation
    latency = max(0, latency)
    energy = max(0, energy)
    return alpha * latency + beta * energy

def get_vehicle_index(vehicle_id):
    """
    从车辆的ID字符串中提取数字索引。

    例如，如果 `vehicle_id` 是 "veh_5"，此函数将返回整数 `5`。
    它假设ID的格式是 "前缀_数字索引"。

    参数:
        vehicle_id (str): 车辆的ID字符串。

    返回:
        int: 从ID中解析出的数字索引。
             如果无法从ID字符串中成功解析出数字索引（例如，格式不匹配或发生错误），
             则打印警告信息并返回 -1。
    """
    try:
        parts = vehicle_id.split('_')
        if len(parts) > 1 and parts[-1].isdigit():
             return int(parts[-1])
        else:
             print(f"Warning: Could not parse index from vehicle_id '{vehicle_id}'. Returning -1.")
             return -1
    except Exception as e:
        print(f"Error parsing vehicle_id '{vehicle_id}': {e}")
        return -1

def get_rsu_index(rsu_id):
    """
    从RSU的ID字符串中提取数字索引。

    例如，如果 `rsu_id` 是 "rsu_2"，此函数将返回整数 `2`。
    它假设ID的格式是 "前缀_数字索引"。

    参数:
        rsu_id (str): RSU的ID字符串。

    返回:
        int: 从ID中解析出的数字索引。
             如果无法从ID字符串中成功解析出数字索引（例如，格式不匹配或发生错误），
             则打印警告信息并返回 -1。
    """
    try:
        parts = rsu_id.split('_')
        if len(parts) > 1 and parts[-1].isdigit():
             return int(parts[-1])
        else:
             print(f"Warning: Could not parse index from rsu_id '{rsu_id}'. Returning -1.")
             return -1
    except Exception as e:
        print(f"Error parsing rsu_id '{rsu_id}': {e}")
        return -1

# --- Simplified Stochastic Heuristic Expert Policy Generation ---
# String hint 'environment.VECEnvironment' relies on environment being imported
def generate_expert_policy_heuristic(task: TaskDAG, env: 'environment.VECEnvironment'): # <--- Uses string hint
    """
    使用基于成本估算和Softmax选择的随机启发式方法，为给定的任务DAG生成一个“专家”卸载策略。

    该函数旨在模拟一个合理的专家决策过程，用于生成模仿学习所需的 (状态, 动作) 数据。
    它会评估将任务在本地执行与卸载到每个可用RSU执行的估计成本（延迟和能耗的加权和），
    然后根据这些成本使用Softmax函数概率性地选择一个执行选项。

    主要步骤:
    1.  **估算本地执行成本**:
        -   遍历任务DAG中的所有子任务，计算在车辆本地CPU上执行它们的总延迟和总能耗。
        -   如果可行（所有子任务都能在本地完成且成本非无限），则计算本地执行的综合目标成本，
            并将其作为一个选项（offload_target=0）添加到 `options` 列表中。
            本地执行的CPU和带宽请求被设为固定值（例如，CPU为1.0，带宽为0.0）。

    2.  **估算卸载到各RSU的成本**:
        -   遍历环境中的每个RSU。
        -   **启发式资源分配**: 估算一个简化的资源分配方案。例如，假设任务会使用RSU
            当前可用CPU资源的一半，并分配标准的信道带宽。
        -   计算数据传输到该RSU的延迟和能耗。如果传输不可行，则跳过此RSU。
        -   估算任务在RSU上计算的总延迟。此估算 **不考虑缓存复用**，以简化启发式逻辑。
            如果任何子任务在分配的资源下无法计算，则认为此RSU不可行。
        -   计算卸载到此RSU的总延迟（传输+计算）和总能耗（仅车辆传输能耗计入目标）。
        -   计算卸载到此RSU的综合目标成本。
        -   如果可行且成本非无限，则将此RSU作为一个卸载选项（offload_target = rsu_index + 1）
            添加到 `options` 列表中。同时记录基于启发式分配的CPU和带宽请求百分比。

    3.  **随机选择决策**:
        -   如果 `options` 列表为空（即没有找到可行的执行方案），则返回 `None`。
        -   从 `options` 列表中提取所有可行选项的成本。
        -   使用Softmax函数将成本转换为选择概率：`p_i = exp(-cost_i / T) / sum(exp(-cost_j / T))`，
            其中 `T` 是温度参数 (`config.HEURISTIC_TEMPERATURE`)，控制随机性。
            较低的 `T` 使选择更倾向于成本最低的选项（更确定性），较高的 `T` 增加探索性。
            为防止数值溢出，在计算指数前会对成本进行缩放和平移。
        -   根据计算出的概率分布，随机选择一个执行选项。如果概率计算或采样失败，
            则回退到确定性地选择成本最低的选项。

    4.  **格式化输出**:
        -   将选定选项的卸载目标ID、CPU分配请求和带宽分配请求打包成一个字典作为专家决策。

    参数:
        task (TaskDAG): 需要为其生成专家决策的任务DAG对象。
        env (environment.VECEnvironment): 当前的VEC环境实例，用于获取车辆和RSU的信息、
                                          信道条件等。

    返回:
        dict or None:
            如果成功生成决策，则返回一个包含专家决策的字典，格式为：
            `{'offload_target': int, 'cpu_alloc_request': float, 'bw_alloc_request': float}`。
            `offload_target` 为0表示本地执行，为 `j+1` 表示卸载到第 `j` 个RSU。
            `cpu_alloc_request` 和 `bw_alloc_request` 是0到1之间的比例值。
            如果没有可行的执行选项，则返回 `None`。
    """
    options = [] # List to store tuples: (option_index, cost, cpu_req, bw_req)
                 # option_index: 0 for local, 1 to N for RSUs

    vehicle = env.get_vehicle_by_id(task.vehicle_id)
    if not vehicle:
        print(f"Error: Vehicle {task.vehicle_id} not found in environment for heuristic.")
        return None

    # --- 1. Estimate Local Cost ---
    local_latency = 0
    local_energy = 0
    local_possible = True
    try:
        for subtask_id in task.get_topological_nodes():
            st = task.get_subtask(subtask_id)
            if not st: continue
            delay = calculate_local_compute_delay(st, vehicle.cpu_frequency)
            energy = calculate_local_compute_energy(vehicle.power_compute, delay)
            if np.isinf(delay) or np.isinf(energy):
                local_possible = False
                break
            local_latency += delay
            local_energy += energy
    except Exception as e:
        print(f"Error calculating local cost for task {task.id}: {e}")
        local_possible = False

    if local_possible:
        local_cost = calculate_objective(local_latency, local_energy)
        if not np.isinf(local_cost):
            options.append({'id': 0, 'cost': local_cost, 'cpu_req': 1.0, 'bw_req': 0.0})
        else:
             local_possible = False # Treat as impossible if objective is inf

    # --- 2. Estimate RSU Costs ---
    veh_idx = get_vehicle_index(task.vehicle_id)
    if veh_idx == -1:
        print(f"Error: Could not get index for vehicle {task.vehicle_id}.")
        # If only local was possible, proceed with that option only
        if not options: return None

    else:
        for j, rsu in enumerate(env.rsus):
            rsu_idx = j
            rsu_id_for_decision = rsu_idx + 1

            # Heuristic resource allocation (Simplified - could be improved)
            available_cpu = max(0, rsu.total_cpu_frequency - rsu.current_cpu_load)
            # Request a random fraction of available, or fixed fraction? Let's try fixed fraction.
            effective_rsu_freq = max(1e3, available_cpu * 0.5) # Simplified: assume use 50% of what's available
            # Estimate needed BW based on config, assume RSU grants this
            allocated_bw = max(1e3, config.CHANNEL_BANDWIDTH_PER_VEHICLE)

            # Estimate Transmission
            noise_power = get_noise_power(allocated_bw)
            gain = env.get_channel_gain(veh_idx, rsu_idx)
            rate = calculate_transmission_rate(vehicle.power_transmit, gain, allocated_bw, noise_power)
            trans_delay = calculate_transmission_delay(task.data_size, rate)
            trans_energy = calculate_transmission_energy(vehicle.power_transmit, trans_delay)

            if np.isinf(trans_delay) or np.isinf(trans_energy):
                continue # Skip this RSU if transmission fails

            # Estimate Computation (Ignoring cache reuse for simplicity in heuristic cost estimate)
            rsu_compute_delay_total = 0
            possible_on_rsu = True
            try:
                for subtask_id in task.get_topological_nodes():
                    st = task.get_subtask(subtask_id)
                    if not st: continue
                    # Heuristic ignores potential cache hits when estimating cost
                    compute_delay = calculate_rsu_compute_delay(st, effective_rsu_freq, is_reused=False)
                    if np.isinf(compute_delay):
                        possible_on_rsu = False
                        break
                    rsu_compute_delay_total += compute_delay
            except Exception as e:
                print(f"Error calculating RSU compute cost for task {task.id} on RSU {rsu.id}: {e}")
                possible_on_rsu = False

            if not possible_on_rsu:
                continue

            total_latency = trans_delay + rsu_compute_delay_total
            total_energy = trans_energy # Vehicle energy only for objective

            rsu_cost = calculate_objective(total_latency, total_energy)

            if not np.isinf(rsu_cost):
                # Calculate % requests based on the *assumed* allocation
                cpu_req_perc = (effective_rsu_freq / rsu.total_cpu_frequency) if rsu.total_cpu_frequency > 0 else 0
                bw_req_perc = (allocated_bw / config.RSU_BANDWIDTH_TOTAL) if config.RSU_BANDWIDTH_TOTAL > 0 else 0 # Fraction of RSU total BW

                options.append({
                    'id': rsu_id_for_decision,
                    'cost': rsu_cost,
                    'cpu_req': max(0.01, min(cpu_req_perc, 1.0)), # Clamp 0.01-1.0
                    'bw_req': max(0.01, min(bw_req_perc, 1.0))  # Clamp 0.01-1.0
                })

    # --- 3. Stochastic Selection based on Costs ---
    if not options:
        print(f"Warning: Heuristic found no viable execution options for task {task.id}")
        return None

    costs = np.array([opt['cost'] for opt in options])

    # Use softmax probability: p_i = exp(-cost_i / T) / sum(exp(-cost_j / T))
    # Handle potential overflow with large negative exponents by subtracting max
    if config.HEURISTIC_TEMPERATURE <= 1e-9: # Avoid division by zero; treat as deterministic
        selected_index = np.argmin(costs)
    else:
        scaled_costs = -costs / config.HEURISTIC_TEMPERATURE
        # Shift to prevent overflow in exp (max will become 0)
        scaled_costs_shifted = scaled_costs - np.max(scaled_costs)
        exp_costs = np.exp(scaled_costs_shifted)
        probabilities = exp_costs / np.sum(exp_costs)

        # Ensure probabilities sum to 1 (handle potential floating point issues)
        probabilities /= probabilities.sum()

        try:
             option_indices = np.arange(len(options))
             selected_index = np.random.choice(option_indices, p=probabilities)
        except ValueError as e:
             print(f"Error sampling expert choice (probs={probabilities}): {e}. Falling back to argmin.")
             # Fallback to deterministic choice if sampling fails
             selected_index = np.argmin(costs)


    selected_option = options[selected_index]

    # --- 4. Format Output ---
    expert_decision = {
        'offload_target': selected_option['id'],
        'cpu_alloc_request': selected_option['cpu_req'],
        'bw_alloc_request': selected_option['bw_req']
    }

    return expert_decision