"""
主脚本，用于运行车载边缘计算（VEC）仿真。

该脚本执行以下操作：
1. 初始化VEC环境，包括车辆和路边单元（RSU）。
2. 生成专家数据集，用于训练模仿学习（IL）模型。
3. 实例化并训练不同的卸载算法模型，包括：
    - GAT-IL (无复用)
    - REUSE-GAT-IL (有复用机制的模仿学习)
    - REUSE-GAT-DRL (有复用机制的深度强化学习)
4. 运行一系列仿真来评估不同算法在各种场景下的性能。
5. 生成并保存结果图，比较算法的性能指标，例如：
    - 训练损失 vs. Epoch (图1)
    - 平均系统成本 vs. 时间槽 (图2)
    - 不同相似性阈值 (δ) 对系统成本的影响 (图3)
    - 平均桶宽 vs. RSU 计算能力 (图4)
    - 平均任务执行延迟 vs. RSU 计算能力 (图5)

脚本依赖于项目中的其他模块：
- `environment.py`: 定义VEC环境组件。
- `config.py`: 存储所有仿真参数和配置。
- `task.py`: 定义任务和子任务的数据结构。
- `algorithms.py`: 实现各种卸载决策算法。
- `utils.py`: 提供辅助计算函数和专家策略生成。
- `gat_model.py`: 定义图注意力网络（GAT）模型结构。

使用方法：
直接运行此脚本 (python main.py) 将启动整个仿真和绘图流程。
确保所有依赖项都已正确安装，并且配置文件 (`config.py`) 已根据需要进行了设置。
"""
#测试,能看到吗
import os # Keep os import at the top
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Set env variable *before* importing numpy/torch/matplotlib
print("Warning: KMP_DUPLICATE_LIB_OK=TRUE is set. This may potentially lead to crashes or inaccurate results.")

import numpy as np
import matplotlib
# matplotlib.use('Agg') # Option 1: Try Agg backend if interactive backend fails (uncomment if needed)
import matplotlib.pyplot as plt
import copy # To deep copy environment/models for different simulations
import traceback # To print detailed errors


import environment # Import the module itself
import config
from task import TaskDAG
# Import specific classes needed elsewhere if required
from environment import VECEnvironment, Vehicle, RSU # Keep these if needed directly
from algorithms import GAT_REUSE_IL, GAT_IL_NoReuse, GAT_REUSE_DRL # Import algorithms
from utils import generate_expert_policy_heuristic # Use heuristic expert
from gat_model import dag_to_pyg_data # For expert data generation

# --- FUNCTION DEFINITIONS START ---

# --- Helper to pad lists ---
def pad_list(lst, length, pad_value=np.nan):
    """
    将列表填充到指定的长度。

    如果列表的当前长度小于目标长度，则使用 `pad_value` 填充列表尾部，
    直到达到目标长度。如果列表当前长度大于或等于目标长度，则截断列表
    至目标长度。

    参数:
        lst (list or iterable): 需要填充或截断的输入列表（或可转换为列表的可迭代对象）。
        length (int): 目标长度。
        pad_value (any, optional): 用于填充的数值。默认为 `np.nan`。

    返回:
        list: 经过填充或截断后达到目标长度的新列表。
              如果输入 `lst` 无法转换成列表，则返回一个用 `pad_value` 填充到 `length` 的列表。
    """
    if not isinstance(lst, list):
        try:
            lst = list(lst)
        except TypeError:
            # print(f"Warning: Cannot convert input of type {type(lst)} to list for padding. Returning empty list.")
            return [pad_value] * length
    current_len = len(lst)
    if current_len < length:
        return lst + [pad_value] * (length - current_len)
    return lst[:length]


# --- Helper to filter data for plotting ---
def filter_plot_data(*data_lists):
    """
    同时从多个列表中过滤掉NaN和Inf值，确保数据对齐。

    此函数接收任意数量的列表作为输入。它会检查每个位置上的元素在所有输入列表中是否都
    是有限的（非NaN且非Inf）。只有当对应索引上的元素在所有列表中都有效时，
    该索引处的数据点才会被保留在所有输出列表中。

    主要用于绘图前的数据清洗，确保例如x轴和y轴数据在移除无效点后仍然对齐。

    参数:
        *data_lists (list of list or iterable): 一个或多个列表（或可转换为列表的可迭代对象）。
                                              所有列表应具有相同的期望长度（在填充之前）。

    返回:
        list of list: 过滤后的列表组成的列表。如果输入为空，或者在过滤后没有有效数据，
                      则返回相应数量的空列表。如果在处理过程中发生错误，也会返回空列表。
    """
    if not data_lists:
        return []
    try:
        data_lists = [list(lst) if not isinstance(lst, list) else lst for lst in data_lists]
        if not data_lists or not data_lists[0]: return [[] for _ in data_lists]
        expected_len = len(data_lists[0])
        padded_lists = [pad_list(lst, expected_len) for lst in data_lists]
        np_lists = [np.array(lst, dtype=float) for lst in padded_lists]
        valid_mask = np.ones(expected_len, dtype=bool)
        for arr in np_lists:
            if len(arr) != expected_len:
                 print(f"Error: Length mismatch after padding filter. Expected {expected_len}, got {len(arr)}.")
                 return [[] for _ in data_lists]
            valid_mask &= np.isfinite(arr)
        filtered_lists = [arr[valid_mask].tolist() for arr in np_lists]
    except Exception as e:
        print(f"Error during data filtering: {e}")
        traceback.print_exc()
        return [[] for _ in data_lists]
    return filtered_lists


# --- Generate Expert Data for IL ---
def generate_il_expert_dataset(num_samples, env_prototype):
    """
    使用启发式策略生成用于模仿学习（IL）的专家数据集。

    该函数通过模拟在VEC环境中运行启发式卸载策略来创建一系列 (状态, 动作) 对。
    状态是任务DAG的图表示，动作是启发式策略为该任务选择的卸载决策和资源分配。

    流程:
    1. 克隆一个基础环境原型 (`env_prototype`) 以避免修改原始环境。
    2. 对于指定数量的样本 (`num_samples`):
        a. 更新临时环境的信道条件。
        b. 随机选择一个有效的车辆来发起一个新任务 (`TaskDAG`)。
        c. 使用 `utils.generate_expert_policy_heuristic` 函数为该任务生成专家决策。
        d. 如果启发式策略成功生成决策，则将任务的图表示 (`task.graph`) 和专家决策
           存储为一个元组，并添加到数据集中。
    3. 打印生成进度和最终生成的数据集大小。

    参数:
        num_samples (int): 需要生成的专家样本数量。
                           对应于 `config.B_B_EXPERT_SAMPLES`。
        env_prototype (VECEnvironment): 一个 `VECEnvironment` 实例，用作生成
                                        专家数据时环境的基础。

    返回:
        list: 一个包含 `(task_graph, expert_decision)` 元组的列表。
              `task_graph` 是一个 `networkx.DiGraph` 对象。
              `expert_decision` 是一个包含 'offload_target', 'cpu_alloc_request',
              'bw_alloc_request' 的字典。
    """
    print("Generating expert dataset using heuristic...")
    dataset = []; temp_env = copy.deepcopy(env_prototype)
    for i in range(num_samples):
        if i % 100 == 0: print(f" Generating sample {i}/{num_samples}") # Print less often
        temp_env.update_channel_conditions(); valid_vehicle_ids = [v.id for v in temp_env.vehicles]
        if not valid_vehicle_ids: print("Error: No vehicles."); continue
        task_vehicle_id = np.random.choice(valid_vehicle_ids); task = TaskDAG(id=f"expert_task_{i}", vehicle_id=task_vehicle_id)
        expert_decision = generate_expert_policy_heuristic(task, temp_env)
        if expert_decision: dataset.append((task.graph, expert_decision))
        # else: print(f"Warning: Heuristic failed {task.id}") # Optionally suppress warning for cleaner output
    print(f"Expert dataset generated: {len(dataset)} samples."); return dataset


# --- Simulation Function (Returns avg_dj) --- # REVERTED
def run_simulation(algorithm_instance, env_prototype, name_suffix="", sim_params=None):
    """
    为给定的算法实例和特定参数运行一次完整的仿真。

    该函数负责设置仿真环境，根据需要应用特定的仿真参数（如RSU的计算能力`fj`，
    初始桶宽`d0`，或复用相似性阈值`delta`），然后通过多个时间槽运行算法，
    收集性能结果，并计算在本次仿真运行期间RSU计算出的平均自适应桶宽 `dj`。

    流程:
    1.  深度复制 `env_prototype` 以创建一个本次仿真专用的环境实例。
    2.  将此环境实例与 `algorithm_instance` 相关联。
    3.  保存 `config` 中可能被修改的原始值（如 `delta`, `d0`）。
    4.  如果提供了 `sim_params`：
        -   根据 `sim_params['fj']` 修改环境中所有RSU的 `total_cpu_frequency`。
        -   根据 `sim_params['d0']` 修改全局配置 `config.INITIAL_BUCKET_WIDTH_D0`。
        -   根据 `sim_params['delta']` 修改全局配置 `config.REUSE_SIMILARITY_THRESHOLD`。
    5.  重置算法实例内部的结果收集器和 `all_djs_this_run` 列表。
    6.  循环 `config.SIMULATION_SLOTS` 个时间槽：
        a.  调用 `env.step()` 推进环境，获取当前时隙的新任务。
            `env.step()` 内部会调用RSU的 `reset_loads()`，清空上一个时隙的负载和 `dj` 历史。
        b.  如果无任务，则记录默认值（如0或NaN）到算法的结果中。
        c.  否则，调用 `algorithm_instance.run_slot(tasks)` 处理当前时隙的任务。
            `run_slot` 内部会为每个任务调用算法的 `decide_and_execute` 方法，
            该方法内部会调用RSU的 `get_adaptive_bucket_width`，从而记录 `dj` 值。
    7.  仿真循环结束后，从 `algorithm_instance.all_djs_this_run` 中收集所有有效的 `dj` 值，
        并计算其平均值 (`avg_dj_for_run`)。
    8.  恢复在步骤3中保存的原始 `config` 值。
    9.  获取算法的最终性能结果字典，并对结果列表进行填充，确保长度与 `config.SIMULATION_SLOTS` 一致。

    参数:
        algorithm_instance (BaseAlgorithm subclass): 要进行仿真的算法类的实例。
        env_prototype (VECEnvironment): VEC环境的一个原型实例，将被复制用于本次仿真。
        name_suffix (str, optional): 附加到算法名称后的字符串，用于日志和调试。默认为空。
        sim_params (dict, optional): 一个字典，包含特定于本次仿真的参数。
                                     可能的键包括 'fj', 'd0', 'delta'。默认为 `None`。

    返回:
        tuple:
            - results (dict): 一个字典，包含算法在仿真过程中的性能指标时间序列数据。
                              键如 'objective', 'latency', 'energy', 'cache_hits'。
                              每个键对应的值是一个列表，长度为 `config.SIMULATION_SLOTS`。
            - avg_dj_for_run (float): 在本次仿真运行期间计算出的所有有效 `dj` 值的平均值。
                                      如果没有收集到有效的 `dj` 值，则为 `np.nan`。
    """
    algo_name = f"{algorithm_instance.name}{name_suffix}"
    print(f"\n--- Running Simulation for {algo_name} ---")
    env = copy.deepcopy(env_prototype)
    algorithm_instance.env = env # Associate fresh env with the instance

    # --- 保存和恢复原始配置值 ---
    original_delta_cfg = config.REUSE_SIMILARITY_THRESHOLD
    original_d0_cfg = config.INITIAL_BUCKET_WIDTH_D0
    # ------------------------

    # --- 清理旧的补丁逻辑, 应用新的 sim_params ---
    if sim_params:
        print(f"  Applying sim_params: {sim_params}")
        if 'fj' in sim_params: # 如果参数中包含 'fj' (RSU计算能力)
            target_fj = sim_params['fj']
            print(f"  Setting RSU fj = {target_fj:.2e}")
            for rsu_instance in env.rsus: # 遍历环境中的所有RSU
                if hasattr(rsu_instance, 'total_cpu_frequency'):
                     rsu_instance.total_cpu_frequency = max(1e3, target_fj) # 设置RSU的CPU频率，确保为正
                else:
                     print(f"Warning: Cannot set 'total_cpu_frequency' on RSU {getattr(rsu_instance, 'id', 'N/A')}")

        if 'd0' in sim_params: # 如果参数中包含 'd0' (初始桶宽)
            target_d0 = sim_params['d0']
            print(f"  Setting config.INITIAL_BUCKET_WIDTH_D0 = {target_d0}")
            config.INITIAL_BUCKET_WIDTH_D0 = target_d0 # 设置全局的初始桶宽配置

        if 'delta' in sim_params: # 如果参数中包含 'delta' (复用相似性阈值)
             config.REUSE_SIMILARITY_THRESHOLD = sim_params['delta'] # 设置全局的delta配置
             print(f"  Setting config.REUSE_SIMILARITY_THRESHOLD = {config.REUSE_SIMILARITY_THRESHOLD}")

    # --- 重置算法状态和结果收集器 ---
    algorithm_instance.results = {'objective': [], 'latency': [], 'energy': [], 'cache_hits': []} # 清空结果记录
    if hasattr(algorithm_instance, 'all_djs_this_run'):
         algorithm_instance.all_djs_this_run = [] # 清空本次运行的dj记录
    else:
         print(f"Warning: Algorithm instance {algo_name} missing 'all_djs_this_run'. Creating attribute.")
         algorithm_instance.all_djs_this_run = [] # 如果属性不存在则创建
    # --------------------------------

    print(f" Starting simulation loop for {config.SIMULATION_SLOTS} slots...")
    # --- 主仿真循环 ---
    for slot in range(1, config.SIMULATION_SLOTS + 1): # 时隙从1开始计数
        try:
            tasks = env.step() # 环境推进一个时隙，返回当前时隙生成的任务列表
                               # env.step() 内部会重置RSU的负载和dj历史

            if not tasks: # 如果当前时隙没有任务
                 # 为各项性能指标记录默认值（通常是0或表示无活动的占位符）
                 algorithm_instance.results['objective'].append(0.0)
                 algorithm_instance.results['latency'].append(0.0)
                 algorithm_instance.results['energy'].append(0.0)
                 algorithm_instance.results['cache_hits'].append(0)
                 continue # 继续下一个时隙

            # 如果有任务，则调用算法的run_slot方法处理这些任务
            # run_slot 会调用 decide_and_execute, 内部会记录dj值
            algorithm_instance.run_slot(tasks)
        except Exception as e: # 捕获仿真时隙中可能发生的任何错误
            print(f"FATAL: Error during simulation slot {env.current_slot} for {algo_name}: {e}")
            traceback.print_exc()
            # 如果发生错误，为各项性能指标记录NaN或表示失败的值
            algorithm_instance.results['objective'].append(np.nan)
            algorithm_instance.results['latency'].append(np.nan)
            algorithm_instance.results['energy'].append(np.nan)
            algorithm_instance.results['cache_hits'].append(0) # 或 np.nan

    print(f" Simulation loop finished for {algo_name}.")

    # --- 计算本次运行的平均 dj ---
    avg_dj_for_run = np.nan # 初始化平均dj为NaN
    if hasattr(algorithm_instance, 'all_djs_this_run'):
         all_djs = algorithm_instance.all_djs_this_run # 获取本次运行记录的所有dj值
         # 过滤掉非数值 (NaN, Inf) 的dj值
         valid_djs = [d for d in all_djs if isinstance(d, (int, float)) and np.isfinite(d)]
         if valid_djs: # 如果存在有效的dj值
             avg_dj_for_run = np.mean(valid_djs) # 计算平均值
             print(f"  Avg dj calculated during run: {avg_dj_for_run:.4f} (from {len(valid_djs)} valid samples)")
         else:
             print("  No valid dj values collected during the run.")
    else:
        print("  Cannot calculate avg_dj: 'all_djs_this_run' attribute missing.")
    # -------------------------

    # --- 恢复原始 config 值 ---
    # 确保修改的全局配置恢复到仿真前的状态，以避免影响后续不相关的仿真
    config.REUSE_SIMILARITY_THRESHOLD = original_delta_cfg
    config.INITIAL_BUCKET_WIDTH_D0 = original_d0_cfg
    # ------------------------

    # --- 准备返回结果 ---
    results = algorithm_instance.get_results() # 获取算法记录的原始结果
    # 确保所有结果列表都填充到与仿真时隙数相同的长度
    for key in ['objective', 'latency', 'energy', 'cache_hits']:
         results[key] = pad_list(results.get(key, []), config.SIMULATION_SLOTS)

    # --- 返回结果字典和平均 dj 值 ---
    return results, avg_dj_for_run
# --- END of modified run_simulation ---


# --- Helper function to calculate interval averages ---
# (This function remains unchanged)
def calculate_interval_averages(slots, values, interval_size=10, num_slots=config.SIMULATION_SLOTS):
    """
    计算时间序列数据在指定间隔内的平均值。

    该函数将整个仿真时隙 (`num_slots`) 分为若干个等长的间隔 (`interval_size`)。
    对于每个间隔，它会计算落在该间隔内的 `values` 的平均值。
    返回的x坐标是每个间隔的结束时隙。

    主要用于对波动较大的原始仿真结果进行平滑处理，以便在图表中更清晰地展示趋势。

    参数:
        slots (list or np.ndarray): 包含时间槽（或x轴坐标）的列表或数组。
        values (list or np.ndarray): 包含对应于 `slots` 的数值的列表或数组。
        interval_size (int, optional): 每个平均间隔的大小（时隙数量）。默认为10。
        num_slots (int, optional): 仿真的总时隙数。默认为 `config.SIMULATION_SLOTS`。

    返回:
        tuple:
            - plot_x_coords (list): 计算得到的每个间隔的结束时隙，用作绘图的x坐标。
            - plot_y_values (list): 对应于 `plot_x_coords` 的每个间隔的平均值。
                                    如果在某个间隔内没有有效数据（例如，所有值都是NaN），
                                    则该间隔的平均值也会是NaN，并且不会包含在返回结果中。
                                    如果输入数据无效或为空，则返回两个空列表。
    """
    if not isinstance(slots, (list, np.ndarray)): slots = list(slots)
    if not isinstance(values, (list, np.ndarray)): values = list(values)

    if len(slots) != len(values) or len(slots) == 0:
        return [], []

    valid_data = [(s, v) for s, v in zip(slots, values) if np.isfinite(v)]
    if not valid_data:
        return [], []

    valid_slots, valid_values = zip(*valid_data)
    valid_slots = np.array(valid_slots)
    valid_values = np.array(valid_values)

    interval_ends = np.arange(interval_size, num_slots + interval_size, interval_size)
    averaged_values = []

    last_end = 0
    for end in interval_ends:
        mask = (valid_slots > last_end) & (valid_slots <= end)
        interval_values = valid_values[mask]
        if len(interval_values) > 0:
            averaged_values.append(np.mean(interval_values))
        else:
            averaged_values.append(np.nan)
        last_end = end

    plot_x_coords = []
    plot_y_values = []
    for i, avg in enumerate(averaged_values):
        if np.isfinite(avg):
            plot_x_coords.append(interval_ends[i])
            plot_y_values.append(avg)

    return plot_x_coords, plot_y_values


# --- Plotting Functions ---

# --- plot_figure_1 (remains unchanged) ---
def plot_figure_1(model_il_noreuse, model_reuse_il): # Accept model objects
    """
    绘制并保存图1：模仿学习（IL）模型的训练损失与训练周期（Epoch）的关系图。

    该图比较了 `REUSE-GAT-IL` 模型的训练收敛情况。
    它从提供的模型对象中获取训练损失历史数据，进行必要的过滤（移除NaN/Inf），
    然后绘制损失曲线。

    参数:
        model_il_noreuse (GAT_IL_NoReuse): 经过训练的 `GAT_IL_NoReuse` 模型实例。
                                           此参数当前未在图1的典型版本中使用，
                                           但保留可能是为了未来的扩展或对比。
                                           当前主要使用 `model_reuse_il`。
        model_reuse_il (GAT_REUSE_IL): 经过训练的 `GAT_REUSE_IL` 模型实例。
                                       其 `train_loss_history` 属性将被用于绘图。

    副作用:
        - 生成一个名为 "fig1_loss_vs_epoch.png" 的图像文件并保存在当前工作目录。
        - 如果模型没有训练损失数据或数据无效，图中会显示相应的提示信息。
        - 打印绘图过程中的日志信息和潜在错误。
    """
    filename="fig1_loss_vs_epoch.png"; print(f"Plotting Figure 1 ({filename})..."); plt.figure(figsize=(8, 5))
    try:
        loss_reuse_il = getattr(model_reuse_il, 'train_loss_history', [])
        max_len = len(loss_reuse_il) if loss_reuse_il else 0
        if max_len == 0:
            print(" Warning: No training loss data found in REUSE-GAT-IL model attributes."); # 更新警告信息
            plt.text(0.5, 0.5, 'No REUSE-GAT-IL training data found', ha='center', va='center')
        else:
            epochs = np.arange(1, max_len + 1);
            loss_reuse_il_padded = pad_list(loss_reuse_il, max_len)
            valid_epochs_reuse_il, valid_loss_reuse_il = filter_plot_data(epochs, loss_reuse_il_padded)
            plotted = False
            if valid_epochs_reuse_il and valid_loss_reuse_il:
                plt.plot(valid_epochs_reuse_il, valid_loss_reuse_il, linestyle='--', label='REUSE-GAT-IL')
                plotted = True

            if not plotted:
                 plt.text(0.5, 0.5, 'No valid REUSE-GAT-IL training data to plot', ha='center', va='center'); # 更新警告信息
                 print(" Warning: No valid points after filtering Fig 1.")
            else:
                 plt.xlabel("Epoch"); plt.ylabel("Training Loss"); plt.title("Figure 1: IL Training Convergence"); plt.legend(); plt.grid(True)

        print(f"  Attempting to save {filename}..."); plt.savefig(filename); print(f"  Successfully saved {filename}.")
    except Exception as e: print(f"  Error saving plot {filename}: {e}"); traceback.print_exc()
    finally: plt.close()

# --- plot_figure_2 (remains unchanged) ---
def plot_figure_2(results_il, results_reuse_il, results_drl):
    """
    绘制并保存图2：不同卸载算法的平均系统成本（目标值）与时间槽的关系对比图。

    该图比较了三种主要算法的性能：
    - GAT-IL (无复用)
    - REUSE-GAT-IL (带复用机制的模仿学习)
    - REUSE-GAT-DRL (带复用机制的深度强化学习)

    它从提供的仿真结果字典中提取每种算法在各个时间槽的系统成本（目标值）数据，
    使用 `calculate_interval_averages` 函数计算10个时间槽的平均值以平滑曲线，
    然后绘制这些平均成本随时间变化的曲线。

    参数:
        results_il (tuple or dict): `GAT_IL_NoReuse` 算法的仿真结果。
                                    如果是元组，则第一个元素是包含 'objective' 列表的结果字典。
                                    如果是字典，则直接包含 'objective' 列表。
        results_reuse_il (tuple or dict): `REUSE_GAT_IL` 算法的仿真结果，格式同上。
        results_drl (tuple or dict): `REUSE_GAT_DRL` 算法的仿真结果，格式同上。

    副作用:
        - 生成一个名为 "fig2_objective_vs_slot.png" 的图像文件并保存在当前工作目录。
        - 如果某些算法的仿真结果数据不足或无效，对应的曲线可能不会被绘制。
        - 打印绘图过程中的日志信息和潜在错误。
    """
    filename="fig2_objective_vs_slot.png"; print(f"Plotting Figure 2 ({filename})..."); plt.figure(figsize=(8, 5))
    try:
        slots = np.arange(1, config.SIMULATION_SLOTS + 1)
        # Unpack results (ignore avg_dj if present)
        obj_il_padded = pad_list(results_il[0].get('objective', []) if isinstance(results_il, tuple) else results_il.get('objective',[]), config.SIMULATION_SLOTS)
        obj_reuse_il_padded = pad_list(results_reuse_il[0].get('objective', []) if isinstance(results_reuse_il, tuple) else results_reuse_il.get('objective',[]), config.SIMULATION_SLOTS)
        obj_drl_padded = pad_list(results_drl[0].get('objective', []) if isinstance(results_drl, tuple) else results_drl.get('objective',[]), config.SIMULATION_SLOTS)

        avg_slots_il, avg_obj_il = calculate_interval_averages(slots, obj_il_padded, 10, config.SIMULATION_SLOTS)
        avg_slots_reuse_il, avg_obj_reuse_il = calculate_interval_averages(slots, obj_reuse_il_padded, 10, config.SIMULATION_SLOTS)
        avg_slots_drl, avg_obj_drl = calculate_interval_averages(slots, obj_drl_padded, 10, config.SIMULATION_SLOTS)

        plotted = False
        delta_val_fig2 = config.REUSE_SIMILARITY_THRESHOLD # Get current delta

        if avg_slots_il and avg_obj_il:
            plt.plot(avg_slots_il, avg_obj_il, marker='o', linestyle='-', label='GAT-IL(noreuse)')
            plotted = True
        if avg_slots_reuse_il and avg_obj_reuse_il:
            plt.plot(avg_slots_reuse_il, avg_obj_reuse_il, marker='s', linestyle='--', label=f'REUSE-GAT-IL (δ={delta_val_fig2})')
            plotted = True
        if avg_slots_drl and avg_obj_drl:
            plt.plot(avg_slots_drl, avg_obj_drl, marker='^', linestyle=':', label=f'REUSE-GAT-DRL (δ={delta_val_fig2})')
            plotted = True

        if not plotted:
            print(" Warning: No valid points after averaging Fig 2.")
            plt.text(0.5, 0.5, 'No valid averaged objective data', ha='center', va='center')
        else:
            plt.xlabel("Time Slot")
            plt.ylabel("Avg System Cost")
            plt.title("Figure 2: Performance Comparison (10-slot average)")
            plt.legend()
            plt.grid(True)
            plt.xticks(np.arange(10, config.SIMULATION_SLOTS + 1, 10))

            all_valid_avg_obj = [o for o in avg_obj_il + avg_obj_reuse_il + avg_obj_drl if np.isfinite(o)]
            if all_valid_avg_obj:
                 min_val = min([o for o in all_valid_avg_obj if o >= 0], default=0)
                 max_val = max(all_valid_avg_obj, default=1)
                 y_bottom = max(0, min_val * 0.9)
                 y_top = max_val * 1.1 if max_val > 0 else 1.0
                 if y_top <= y_bottom: y_top = y_bottom + 0.1
                 plt.ylim(bottom=y_bottom, top=y_top)
            else:
                 plt.ylim(bottom=0)

        print(f"  Attempting to save {filename}..."); plt.savefig(filename); print(f"  Successfully saved {filename}.")
    except Exception as e: print(f"  Error saving plot {filename}: {e}"); traceback.print_exc()
    finally: plt.close()

# --- plot_figure_3 (remains unchanged) ---
def plot_figure_3(results_delta, fixed_d0_used):
    """
    绘制并保存图3：不同相似性阈值 `δ` (delta) 对 `REUSE-GAT-IL` 算法平均系统成本的影响图。

    该图展示了在固定初始桶宽 `d0` 的条件下，改变复用机制中的相似性阈值 `δ` 时，
    `REUSE-GAT-IL` 算法的平均系统成本（目标值）随时间槽的变化情况。
    图中会为每个测试的 `δ` 值绘制一条曲线。

    与图2类似，它也使用 `calculate_interval_averages` 对原始数据进行平滑处理。

    参数:
        results_delta (dict): 一个字典，键是不同的 `δ` (delta) 值，
                              值是对应 `δ` 值下 `REUSE-GAT-IL` 算法的仿真结果元组
                              `(results_dict, avg_dj_for_run)`。
                              `results_dict` 中包含 'objective' 列表。
        fixed_d0_used (float): 在进行这些 `δ` 值对比仿真时，所使用的固定的初始桶宽 `d0` 的值。
                               这个值会显示在图表的标题中。

    副作用:
        - 生成一个名为 "fig3_delta_comparison.png" 的图像文件并保存在当前工作目录。
        - 如果某些 `δ` 值的仿真结果数据不足或无效，对应的曲线可能不会被绘制。
        - 打印绘图过程中的日志信息和潜在错误。
    """
    filename = "fig3_delta_comparison.png"; print(f"Plotting Figure 3 ({filename})..."); plt.figure(figsize=(8, 5))
    try:
        slots = np.arange(1, config.SIMULATION_SLOTS + 1)
        plotted = False
        all_avg_objectives = []

        for delta, (results, _) in results_delta.items(): # Unpack results tuple, ignore avg_dj
            obj_list_padded = pad_list(results.get('objective', []), config.SIMULATION_SLOTS)
            avg_slots, avg_obj = calculate_interval_averages(slots, obj_list_padded, 10, config.SIMULATION_SLOTS)

            if avg_slots and avg_obj:
                print(f"  Plotting {len(avg_slots)} averaged points delta={delta}.")
                plt.plot(avg_slots, avg_obj, marker='s', linestyle='--', label=f'REUSE-GAT-IL (δ={delta})')
                all_avg_objectives.extend(avg_obj)
                plotted = True
            else:
                print(f" Warning: No valid averaged data delta={delta} Fig 3.")

        if plotted:
            plt.xlabel("Time Slot")
            plt.ylabel("Avg System Cost")
            title = f"Figure 3: Impact of Similarity Threshold (δ) (d0={fixed_d0_used} fixed, 10-slot avg)"
            plt.title(title)
            plt.legend()
            plt.grid(True)
            plt.xticks(np.arange(10, config.SIMULATION_SLOTS + 1, 10))

            if all_avg_objectives:
                 finite_objectives = [o for o in all_avg_objectives if np.isfinite(o)]
                 if finite_objectives:
                     min_val = min([o for o in finite_objectives if o >= 0], default=0)
                     max_val = max(finite_objectives, default=1)
                     y_bottom = max(0, min_val * 0.9)
                     y_top = max_val * 1.1 if max_val > 0 else 1.0
                     if y_top <= y_bottom: y_top = y_bottom + 0.1
                     plt.ylim(bottom=y_bottom, top=y_top)
                 else:
                     plt.ylim(bottom=0)
            else:
                 plt.ylim(bottom=0)
        else:
            plt.text(0.5, 0.5, 'No valid averaged data Fig 3', ha='center', va='center')

        print(f"  Attempting to save {filename}..."); plt.savefig(filename); print(f"  Successfully saved {filename}.")
    except Exception as e: print(f"  Error saving plot {filename}: {e}"); traceback.print_exc()
    finally: plt.close()


# --- Reverted plot_figure_4 (Average dj Plot) --- # REVERTED
def plot_figure_4(results_dynamic_load):
    """
    绘制并保存图4：在动态负载条件下，平均自适应桶宽 `dj` 与 RSU 计算能力 `fj` 的关系图。

    该图展示了对于不同的初始桶宽 `d0` (来自 `config.D0_VALUES_FIG4_5`)，
    当RSU的总计算能力 `fj` (x轴，来自 `config.F_J_VALUES_FIG4_5`) 变化时，
    `REUSE-GAT-IL` 算法在整个仿真过程中计算出的平均 `dj` 值是如何变化的。
    图中会为每个测试的 `d0` 值绘制一条曲线。

    数据来源 `results_dynamic_load` 是一个嵌套字典，结构为：
    `{fj_value: {d0_value: {'avg_latency': ..., 'avg_dj': ...}}}`。
    此函数提取其中的 `'avg_dj'` 值进行绘图。

    参数:
        results_dynamic_load (dict): 一个嵌套字典，包含了在不同 `fj` 和 `d0` 组合下，
                                     `REUSE-GAT-IL` 算法的仿真结果，
                                     特别是每次仿真运行的平均 `dj` (`avg_dj`)。

    副作用:
        - 生成一个名为 "fig4_dj_vs_fj_dynamic.png" 的图像文件并保存在当前工作目录。
        - 如果某些 `d0` 值的仿真结果数据不足或无效，对应的曲线可能不会被绘制。
        - 打印绘图过程中的日志信息和潜在错误。
    """
    filename="fig4_dj_vs_fj_dynamic.png"; # Original filename
    print(f"Plotting Figure 4 ({filename}) using dynamic load data (Average dj)..."); # Changed description
    plt.figure(figsize=(8, 5))
    try:
        if not results_dynamic_load:
            print(" Warning: No data provided for Figure 4."); raise ValueError("Empty data for Figure 4")

        try:
            fj_values_numeric = sorted([float(fj) for fj in results_dynamic_load.keys()])
        except ValueError:
            print("Warning: Non-numeric keys found for fj in Figure 4 data.");
            fj_values_numeric = sorted(list(results_dynamic_load.keys()))

        plotted = False
        all_plotted_djs = [] # Renamed from all_avg_djs for clarity
        for d0_val in config.D0_VALUES_FIG4_5:
            dj_values = [] # Stores avg_dj for this d0
            valid_fj = []
            for fj_num in fj_values_numeric:
                 fj_key = float(fj_num) # Use float key
                 if fj_key in results_dynamic_load and d0_val in results_dynamic_load[fj_key]:
                     # Retrieve the pre-calculated average dj
                     avg_dj = results_dynamic_load[fj_key][d0_val].get('avg_dj', np.nan)
                     if np.isfinite(avg_dj): # Check finiteness
                         dj_values.append(avg_dj)
                         valid_fj.append(fj_key)

            if dj_values:
                print(f"  Plotting {len(dj_values)} avg_dj points for d0={d0_val}.")
                plt.plot(valid_fj, dj_values, marker='o', linestyle='-', label=f'd0={d0_val}') # Plot avg dj
                all_plotted_djs.extend(dj_values)
                plotted = True
            else:
                print(f" Warning: No valid avg_dj data found for d0={d0_val} in Figure 4.")

        if plotted:
            plt.xlabel(r'$f_j$ [Hz]')
            plt.ylabel(r'Average $d_j$ (during simulation)') # Y label is average dj
            plt.title(f"Figure 4: Average Bucket Width vs RSU Capacity (Dynamic Load)")
            plt.legend()
            plt.grid(True)
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            if all_plotted_djs:
                 valid_djs_for_limits = [d for d in all_plotted_djs if np.isfinite(d)]
                 if valid_djs_for_limits:
                     min_dj = min(valid_djs_for_limits)
                     max_dj = max(valid_djs_for_limits)
                     y_bottom = max(0, min_dj * 0.9)
                     y_top = max_dj * 1.1 if max_dj > 0 else 0.1
                     if y_top <= y_bottom: y_top = y_bottom + 0.1
                     plt.ylim(bottom=y_bottom, top=y_top)
                 else:
                      plt.ylim(bottom=0)
            else:
                 plt.ylim(bottom=0)
        else:
            plt.text(0.5, 0.5, 'No valid data available for plotting', ha='center', va='center')

        print(f"  Attempting to save {filename}..."); plt.savefig(filename); print(f"  Successfully saved {filename}.")
    except Exception as e:
        print(f"  Error saving plot {filename}: {e}"); traceback.print_exc()
    finally:
        plt.close()
# --- END of reverted plot_figure_4 ---


# --- Modified plot_figure_5 (Different Linestyles) --- # MODIFIED
def plot_figure_5(results_dynamic_load):
    """
    绘制并保存图5：在动态负载条件下，平均任务执行延迟与RSU计算能力 `fj` 的关系图。

    该图展示了对于不同的初始桶宽 `d0` (来自 `config.D0_VALUES_FIG4_5`)，
    当RSU的总计算能力 `fj` (x轴，来自 `config.F_J_VALUES_FIG4_5`，单位转换为GHz) 变化时，
    `REUSE-GAT-IL` 算法实现的平均任务执行延迟是如何变化的。
    图中会为每个测试的 `d0` 值绘制一条曲线，并使用不同的标记样式区分。

    数据来源 `results_dynamic_load` 是一个嵌套字典，结构为：
    `{fj_value: {d0_value: {'avg_latency': ..., 'avg_dj': ...}}}`。
    此函数提取其中的 `'avg_latency'` 值进行绘图。

    参数:
        results_dynamic_load (dict): 一个嵌套字典，包含了在不同 `fj` 和 `d0` 组合下，
                                     `REUSE-GAT-IL` 算法的仿真结果，
                                     特别是每次仿真运行的平均任务执行延迟 (`avg_latency`)。

    副作用:
        - 生成一个名为 "fig5_delay_vs_fj_dynamic.png" 的图像文件并保存在当前工作目录。
        - 如果某些 `d0` 值的仿真结果数据不足或无效，对应的曲线可能不会被绘制。
        - 打印绘图过程中的日志信息和潜在错误。
    """
    filename="fig5_delay_vs_fj_dynamic.png";
    print(f"Plotting Figure 5 ({filename}) using dynamic load data (Avg Latency)...");
    plt.figure(figsize=(8, 5))
    try:
        if not results_dynamic_load:
            print(" Warning: No data provided for Figure 5."); raise ValueError("Empty data for Figure 5")

        try:
            fj_values_numeric = sorted([float(fj) for fj in results_dynamic_load.keys()])
        except ValueError:
            print("Warning: Non-numeric keys found for fj in Figure 5 data.");
            fj_values_numeric = sorted(list(results_dynamic_load.keys()))

        plotted = False
        all_delays = []
        markers = ['o', '^', 's'] # 定义标记形状列表
        marker_idx = 0 # 初始化标记索引

        for d0_val in config.D0_VALUES_FIG4_5:
             delay_values = []
             valid_fj = []
             for fj_num in fj_values_numeric:
                 fj_key = float(fj_num) # Use float key
                 if fj_key in results_dynamic_load and d0_val in results_dynamic_load[fj_key]:
                     avg_latency = results_dynamic_load[fj_key][d0_val].get('avg_latency', np.nan)
                     if np.isfinite(avg_latency):
                         delay_values.append(avg_latency)
                         valid_fj.append(fj_key)

             if delay_values:
                 print(f"  Plotting {len(delay_values)} avg_latency points for d0={d0_val}.")
                 marker_shape = markers[marker_idx % len(markers)] # 选择标记形状
                 # 将 fj 转换为 GHz
                 valid_fj_ghz = [f / 1e9 for f in valid_fj]
                 plt.plot(valid_fj_ghz, delay_values, marker=marker_shape, linestyle='-', label=rf'$d_0={d0_val}$') # 修改图例标签，使用GHz数据
                 all_delays.extend(delay_values)
                 plotted = True
                 marker_idx += 1 # 更新标记索引
             else:
                 print(f" Warning: No valid avg_latency data found for d0={d0_val} in Figure 5.")

        if plotted:
            plt.xlabel(r'Computation Capacity of R$_j$  $f_j$ (GHz)') # 修改横坐标标签
            plt.ylabel("Avg Task Execution Latency (s)")
            plt.title(f"Figure 5: Average Latency vs RSU Capacity (Dynamic Load)")
            plt.legend()
            plt.grid(True)
            # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) # 移除科学计数法格式
            if all_delays:
                valid_positive_delays = [d for d in all_delays if np.isfinite(d) and d >= 0]
                if valid_positive_delays:
                    min_delay = min(valid_positive_delays)
                    max_delay = max(valid_positive_delays)
                    y_bottom = max(0, min_delay * 0.9)
                    y_top = max_delay * 1.1 if max_delay > 0 else 1.0
                    if y_top <= y_bottom: y_top = y_bottom + 0.1
                    plt.ylim(bottom=y_bottom, top=y_top)
                else:
                    plt.ylim(bottom=0)
            else:
                plt.ylim(bottom=0)
        else:
            plt.text(0.5, 0.5, 'No valid data available for plotting', ha='center', va='center')

        print(f"  Attempting to save {filename}..."); plt.savefig(filename); print(f"  Successfully saved {filename}.")
    except Exception as e:
        print(f"  Error saving plot {filename}: {e}"); traceback.print_exc()
    finally:
        plt.close()
# --- END of modified plot_figure_5 ---

# --- FUNCTION DEFINITIONS END ---


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting main execution ---")
    # --- Initialize Environment ---
    # 尝试初始化VEC环境
    try:
        base_env = VECEnvironment() # 创建VECEnvironment的实例
        # 检查环境是否成功创建了车辆和RSU
        if not base_env.vehicles or not base_env.rsus:
            raise RuntimeError("Environment initialization failed: No vehicles or RSUs created.")
        print(f"Environment initialized with {len(base_env.vehicles)} vehicles and {len(base_env.rsus)} RSUs.")
    except Exception as e:
        # 如果环境初始化失败，打印错误并退出
        print(f"FATAL: Environment initialization error: {e}")
        traceback.print_exc()
        exit()

    # --- Store Original Config Values ---
    # 保存一些可能在仿真过程中被修改的原始配置值，以便后续恢复
    main_original_d0 = config.INITIAL_BUCKET_WIDTH_D0      # 保存原始的初始桶宽d0
    main_original_delta = config.REUSE_SIMILARITY_THRESHOLD # 保存原始的复用相似性阈值delta
    main_original_sim_slots = config.SIMULATION_SLOTS     # 保存原始的仿真时隙数

    # --- Generate Expert Data ---
    # 为模仿学习（IL）模型生成专家数据集
    expert_dataset = [] # 初始化专家数据集列表
    try:
        # 调用函数生成专家数据，使用配置中定义的样本数量和基础环境
        expert_dataset = generate_il_expert_dataset(config.B_B_EXPERT_SAMPLES, base_env)
        if not expert_dataset:
            # 如果未能生成任何专家样本，则打印警告
            print("Warning: Failed to generate any expert dataset samples. IL models will not train.")
        else:
            print(f"Successfully generated {len(expert_dataset)} expert samples.")
    except Exception as e:
        # 如果专家数据生成过程中发生错误，打印错误信息
        print(f"FATAL: Error during expert data generation: {e}")
        traceback.print_exc()
        # exit() # 可以选择在此处退出脚本

    # --- Instantiate and Train IL Models ONCE ---
    # 实例化并（如果专家数据可用）训练模仿学习模型
    # 这些模型在脚本开始时训练一次，然后在后续的多次仿真中复用其训练好的参数（除非特定仿真需要重新训练或不同模型）
    print("\n--- Initializing and Training IL Models ---")
    # 实例化GAT-IL (无复用) 模型
    gat_il_noreuse_model = GAT_IL_NoReuse(base_env)
    # 实例化REUSE-GAT-IL (带复用) 模型
    gat_reuse_il_model = GAT_REUSE_IL(base_env)

    if expert_dataset: # 仅当专家数据集非空时才进行训练
        try:
            print("Training GAT-IL (no reuse)...")
            gat_il_noreuse_model.train(expert_dataset) # 训练GAT-IL模型
            print("Training GAT-IL (no reuse) finished.")
        except Exception as e:
            print(f"Error training GAT_IL_NoReuse: {e}") # 打印训练GAT_IL_NoReuse时的错误
            traceback.print_exc()
        try:
            print("Training REUSE-GAT-IL...")
            gat_reuse_il_model.train(expert_dataset) # 训练REUSE-GAT-IL模型
            print("Training REUSE-GAT-IL finished.")
        except Exception as e:
            print(f"Error training GAT_REUSE_IL: {e}") # 打印训练GAT_REUSE_IL时的错误
            traceback.print_exc()
    else:
        # 如果专家数据为空，则跳过IL模型训练
        print("Skipping IL training due to empty or failed expert dataset generation.")


    # --- Run Baseline Simulations (Fig 1 & 2) ---
    # 运行基线仿真，主要用于生成图1（IL训练损失）和图2（算法性能对比）所需的数据
    print("\n--- Running Simulations for Fig 1 & 2 ---")
    # 初始化结果变量，用于存储仿真结果字典和平均dj值。
    # GAT_IL_NoReuse的结果也用于图1的损失曲线（如果适用，尽管当前图1主要关注REUSE-GAT-IL）。
    results_gat_il_noreuse_fig12 = ({}, np.nan) # (结果字典, 平均dj)
    results_gat_reuse_il_fig2 = ({}, np.nan)    # (结果字典, 平均dj)
    results_gat_reuse_drl_fig2 = ({}, np.nan)   # (结果字典, 平均dj)
    gat_reuse_drl_instance_fig2 = None          # 为图2的DRL仿真创建一个特定的实例，以隔离其训练状态

    try:
        # 确保基线仿真使用原始（或预期的）配置值
        config.REUSE_SIMILARITY_THRESHOLD = main_original_delta # 恢复/设置复用相似性阈值
        config.INITIAL_BUCKET_WIDTH_D0 = main_original_d0       # 恢复/设置初始桶宽
        config.SIMULATION_SLOTS = main_original_sim_slots       # 恢复/设置仿真时隙数

        print("Running GAT-IL (noreuse) for Fig 1/2 baseline...")
        # 运行GAT-IL (无复用) 算法的仿真
        # `gat_il_noreuse_model` 是之前训练好的（如果专家数据可用）
        results_gat_il_noreuse_fig12 = run_simulation(
            algorithm_instance=gat_il_noreuse_model,
            env_prototype=base_env,
            name_suffix="_Fig1_2_Base" # 日志后缀
        )

        print("Running REUSE-GAT-IL for Fig 2 baseline...")
        # 运行REUSE-GAT-IL算法的仿真
        # `gat_reuse_il_model` 也是之前训练好的
        results_gat_reuse_il_fig2 = run_simulation(
            algorithm_instance=gat_reuse_il_model,
            env_prototype=base_env,
            name_suffix="_Fig2_Base"
        )

        print("Running REUSE-GAT-DRL for Fig 2 baseline...")
        # 为DRL创建一个新的实例进行图2的仿真，以确保其从头开始学习（如果DRL模型会持续学习的话）
        # 或者如果DRL模型是加载预训练参数，则确保加载正确的参数。
        # 当前DRL模型在`run_slot`中训练，所以每次`run_simulation`都是一次新的训练/评估过程。
        gat_reuse_drl_instance_fig2 = GAT_REUSE_DRL(base_env) # 创建新的DRL实例
        results_gat_reuse_drl_fig2 = run_simulation(
            algorithm_instance=gat_reuse_drl_instance_fig2,
            env_prototype=base_env,
            name_suffix="_Fig2_Base"
        )

    except Exception as e:
        # 如果基线仿真过程中出错，打印错误信息
        print(f"FATAL: Error during Fig 1/2 simulation runs: {e}")
        traceback.print_exc()
    finally:
        # 无论成功与否，都尝试恢复原始配置值，以防后续仿真需要它们
        config.REUSE_SIMILARITY_THRESHOLD = main_original_delta
        config.INITIAL_BUCKET_WIDTH_D0 = main_original_d0
        config.SIMULATION_SLOTS = main_original_sim_slots

    # --- Run Simulations for Fig 3 (Varying Delta) ---
    # 运行一系列仿真以评估不同相似性阈值δ (delta)对REUSE-GAT-IL性能的影响（用于图3）
    print("\n--- Running Simulations for Fig 3 (Varying Delta) ---")
    results_delta_comparison = {} # 初始化字典，用于存储不同delta值对应的仿真结果
    fixed_d0_fig3 = 0.2           # 为图3的仿真定义一个固定的初始桶宽d0值

    try:
        print(f"  Fixing d0 = {fixed_d0_fig3} for Fig 3 simulations.")
        config.SIMULATION_SLOTS = main_original_sim_slots # 确保使用正确的仿真时隙长度
        d0_before_fig3_loop = config.INITIAL_BUCKET_WIDTH_D0 # 保存当前的d0值，以便在循环后恢复
        config.INITIAL_BUCKET_WIDTH_D0 = fixed_d0_fig3 # 设置图3实验所需的固定d0值

        # 遍历预定义的delta值列表 (来自config.DELTA_VALUES_FIG3)
        for delta_val in config.DELTA_VALUES_FIG3:
            print(f"  Running simulation for delta = {delta_val}...")
            # 设置此次仿真的特定参数：当前的delta值和固定的d0值
            sim_params = {'delta': delta_val, 'd0': fixed_d0_fig3}
            # 使用REUSE-GAT-IL模型运行仿真
            # run_simulation 返回 (仿真结果字典, 本次运行的平均dj值) 元组
            results_delta_comparison[delta_val] = run_simulation(
                gat_reuse_il_model, # 复用之前训练好的REUSE-GAT-IL模型实例
                base_env,           # 使用基础环境原型
                name_suffix=f"_Fig3_delta{delta_val}", # 为日志添加后缀，区分不同delta的运行
                sim_params=sim_params # 传递特定参数 (delta, d0)
            ) # 将结果元组存储在字典中，以delta值为键

    except Exception as e:
        # 如果图3仿真过程中出错，打印错误信息
        print(f"FATAL: Error during Fig 3 simulation runs: {e}")
        traceback.print_exc()
    finally:
        config.INITIAL_BUCKET_WIDTH_D0 = d0_before_fig3_loop # 无论成功与否，都恢复之前的d0值

    # --- Modified: Data Collection for Fig 4 & 5 --- # REVERTED data storage
    # 收集用于图4（平均dj vs RSU能力）和图5（平均延迟 vs RSU能力）的动态负载数据
    print("\n--- Collecting dynamic load data for Fig 4 & 5 ---")
    # 初始化结果存储字典，第一层键是fj值，第二层键是d0值，值为包含'avg_latency'和'avg_dj'的字典
    results_dynamic_load_fig4_5 = {fj: {} for fj in config.F_J_VALUES_FIG4_5}
    try:
        config.SIMULATION_SLOTS = main_original_sim_slots # 确保使用完整的仿真时隙长度
        original_d0_fig4_5 = config.INITIAL_BUCKET_WIDTH_D0 # 保存当前的d0值，以便后续恢复 (变量名修正为 original_d0_fig4_5 更清晰)

        print(f" Running full simulations for Fig 4/5 data ({config.SIMULATION_SLOTS} slots each)...")
        # 外层循环：遍历config中为图4/5定义的每个RSU计算能力fj值 (F_J_VALUES_FIG4_5)
        for fj_val in config.F_J_VALUES_FIG4_5:
            fj_key = float(fj_val) # 将fj值转换为float，用作字典的键
            print(f"  Processing fj = {fj_key:.2e}...")
            results_dynamic_load_fig4_5[fj_key] = {} # 为当前的fj值在结果字典中创建一个新的空字典
            # 内层循环：对于每个fj值，遍历config中为图4/5定义的每个初始桶宽d0值 (D0_VALUES_FIG4_5)
            for d0_val in config.D0_VALUES_FIG4_5:
                print(f"    Processing d0 = {d0_val}...")
                # 设置此次仿真的特定参数：当前的RSU计算能力fj和初始桶宽d0
                sim_params = {'fj': fj_key, 'd0': d0_val}

                # 使用REUSE-GAT-IL模型运行仿真
                # run_simulation 函数返回一个元组: (仿真结果字典, 本次运行中计算出的平均dj值)
                sim_results_dict, avg_dj_value = run_simulation(
                    gat_reuse_il_model, # 使用之前训练好的REUSE-GAT-IL模型实例
                    base_env,           # 使用基础环境原型
                    name_suffix=f"_Fig45_dyn_fj{fj_key:.1e}_d0{d0_val}", # 为日志添加有意义的后缀
                    sim_params=sim_params # 传递特定参数 (fj, d0)
                )

                # 从仿真结果字典中提取'latency'列表，并过滤掉无效的（NaN或Inf）延迟数据
                valid_latencies = [l for l in sim_results_dict.get('latency', []) if np.isfinite(l)]
                # 计算这些有效延迟的平均值；如果列表中没有有效延迟数据，则平均延迟为NaN
                avg_latency = np.mean(valid_latencies) if valid_latencies else np.nan

                # 将计算得到的平均延迟 (avg_latency) 和 平均dj值 (avg_dj_value) 存储到结果字典中
                # 键的路径是 results_dynamic_load_fig4_5[fj_value][d0_value]
                results_dynamic_load_fig4_5[fj_key][d0_val] = {
                    'avg_latency': avg_latency, # 存储平均延迟
                    'avg_dj': avg_dj_value      # 存储平均dj值
                }
                # 打印当前 (fj, d0) 组合下得到的平均延迟和平均dj
                print(f"      -> Avg Latency: {avg_latency:.4f}, Avg Dj: {avg_dj_value:.4f}")

        config.INITIAL_BUCKET_WIDTH_D0 = original_d0_fig4_5 # 在所有fj和d0的循环结束后，恢复原始的d0值

    except Exception as e:
        # 如果在为图4/5收集数据的过程中发生任何错误，打印错误信息
        print(f"FATAL: Error during dynamic data collection for Fig 4/5: {e}")
        traceback.print_exc()
        config.INITIAL_BUCKET_WIDTH_D0 = original_d0_fig4_5 # 即使发生错误，也尝试恢复原始的d0值
    # --- END of reverted data collection ---


    # --- Plotting (Using reverted plot function for 4, modified 5) ---
    # 生成并保存所有图表
    print("\n--- Generating Plots (Avg dj for Fig 4, styled Fig 5) ---")
    try:
        # 绘制图1：IL模型训练损失 vs. Epoch
        # 只有当两个IL模型实例都存在时才绘制
        if gat_il_noreuse_model and gat_reuse_il_model:
            plot_figure_1(gat_il_noreuse_model, gat_reuse_il_model)
        else:
            print(" Skipping Fig 1 plot due to missing model instances.")

        # 绘制图2：算法性能对比 (平均系统成本 vs. 时间槽)
        # 检查用于图2的所有基线仿真结果是否都有效
        valid_fig2_data = (results_gat_il_noreuse_fig12 and results_gat_il_noreuse_fig12[0] and
                           results_gat_reuse_il_fig2 and results_gat_reuse_il_fig2[0] and
                           results_gat_reuse_drl_fig2 and results_gat_reuse_drl_fig2[0])
        if valid_fig2_data:
             plot_figure_2(results_gat_il_noreuse_fig12, results_gat_reuse_il_fig2, results_gat_reuse_drl_fig2)
        else:
             print(" Skipping Fig 2 plot due to missing baseline simulation data.")

        # 绘制图3：相似性阈值δ对REUSE-GAT-IL性能的影响
        if results_delta_comparison: # 仅当有delta对比结果时才绘制
             plot_figure_3(results_delta_comparison, fixed_d0_used=fixed_d0_fig3)
        else:
             print(" Skipping Fig 3 plot due to missing delta comparison data.")

        # 绘制图4和图5：动态负载下的性能 (平均dj和平均延迟 vs. RSU计算能力)
        if results_dynamic_load_fig4_5: # 仅当有动态负载数据时才绘制
            print("Plotting Figure 4 with dynamic load data (Average dj)...")
            plot_figure_4(results_dynamic_load_fig4_5) # 调用图4的绘制函数

            print("Plotting Figure 5 with dynamic load data (Average Latency, styled)...")
            plot_figure_5(results_dynamic_load_fig4_5) # 调用图5的绘制函数
        else:
            print(" Skipping Fig 4 & 5 plots due to missing dynamic load data.")

    except Exception as e:
        # 如果绘图过程中出错，打印错误信息
        print(f"FATAL: Plotting error: {e}")
        traceback.print_exc()
    # --- END of modified plotting section ---

    # --- Final Restoration (Optional, good practice) ---
    # 脚本结束前，再次确保所有可能在仿真过程中修改的全局配置值恢复到初始状态
    config.INITIAL_BUCKET_WIDTH_D0 = main_original_d0
    config.REUSE_SIMILARITY_THRESHOLD = main_original_delta
    config.SIMULATION_SLOTS = main_original_sim_slots
    print(f"\nRestored original config: d0={config.INITIAL_BUCKET_WIDTH_D0}, delta={config.REUSE_SIMILARITY_THRESHOLD}, slots={config.SIMULATION_SLOTS}")

    print("\n--- Plotting phase complete ---")
    print("--- Script execution finished ---")
