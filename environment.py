"""
此模块定义了车载边缘计算（VEC）仿真环境的组件。

主要包含以下类：

1.  `Vehicle`:
    表示VEC系统中的一辆车。它具有特定的计算能力（CPU频率）以及
    用于本地计算和数据传输的功率消耗参数。

2.  `RSU (Roadside Unit)`:
    表示一个路边单元。RSU拥有计算资源（总CPU频率）、带宽资源，
    并配备了一个缓存机制来存储和复用子任务的计算结果。
    RSU的核心特性之一是 `get_adaptive_bucket_width` 方法，该方法根据
    RSU当前的负载动态调整A-LSH（Adaptive Locality Sensitive Hashing）算法中的桶宽度`dj`，
    这是实现自适应缓存复用的关键。RSU还负责跟踪其当前的CPU和带宽负载，
    并在每个仿真时隙开始时重置这些负载。

3.  `VECEnvironment`:
    模拟整个VEC环境。它管理系统中的所有车辆和RSU实例。
    该类负责：
    - 初始化车辆和RSU。
    - 初始化和更新车辆与RSU之间的信道增益，模拟动态的无线信道条件。
    - 在每个仿真时隙（`step`方法）生成新的计算任务。
    - 提供接口以获取环境中的特定车辆或RSU实例。

该模块与 `config.py` 紧密相关，后者提供了所有环境参数的配置，
例如车辆和RSU的数量、它们的物理属性以及信道模型的参数。
它也与 `task.py` 交互，因为环境会生成 `TaskDAG` 对象。
"""
# environment.py
import numpy as np
import config
from task import TaskDAG
from collections import deque
import traceback

class Vehicle:
    """
    表示车载边缘计算（VEC）系统中的一辆车。

    车辆是任务的发起者，并且具有在本地处理计算任务的能力。
    它也能够将其任务卸载到路边单元（RSU）进行处理。

    属性:
        id (str): 车辆的唯一标识符 (例如, "veh_0", "veh_1", ...)。
        cpu_frequency (float): 车辆本地CPU的处理频率 (单位: Hz)。
                               该值从 `config.VEHICLE_CPU_FREQ` 初始化。
        power_compute (float): 车辆在本地执行计算任务时的功率消耗 (单位: Watts)。
                               该值从 `config.VEHICLE_POWER_COMPUTE` 初始化。
        power_transmit (float): 车辆向RSU传输数据时的功率消耗 (单位: Watts)。
                                该值从 `config.VEHICLE_POWER_TRANSMIT` 初始化。
    """
    def __init__(self, id):
        self.id = id
        self.cpu_frequency = config.VEHICLE_CPU_FREQ # fi
        self.power_compute = config.VEHICLE_POWER_COMPUTE # Pi
        self.power_transmit = config.VEHICLE_POWER_TRANSMIT # Pi,j_trans

class RSU:
    """
    表示一个路边单元（RSU），它是VEC系统中的边缘服务器。

    RSU为车辆提供计算卸载服务。它拥有计算资源（CPU），带宽资源，
    并实现了一个基于A-LSH（Adaptive Locality Sensitive Hashing）的缓存机制，
    用于存储和复用子任务的计算结果，以提高效率。

    核心功能和属性:
    - `id (str)`: RSU的唯一标识符 (例如, "rsu_0", "rsu_1", ...)。
    - `total_cpu_frequency (float)`: RSU可用的总CPU处理频率 (单位: Hz)。
      在初始化时从 `config.RSU_CPU_FREQ_TOTAL_MIN` 和 `config.RSU_CPU_FREQ_TOTAL_MAX`
      之间的均匀分布中随机选取。
    - `total_bandwidth (float)`: RSU可用的总带宽 (单位: Hz)，从 `config.RSU_BANDWIDTH_TOTAL` 初始化。
    - `cache (dict)`: 一个字典，用作子任务结果的缓存。键是LSH哈希值，值是包含
      {'result': ..., 'feature_vector': ..., 'timestamp': ...} 的字典。
    - `cache_order_deque (deque)`: 一个双端队列，用于实现缓存的替换策略（如LRU）。
      存储缓存条目的键（LSH哈希值），按访问顺序排列。
    - `current_cpu_load (float)`: RSU当前已分配或正在使用的CPU频率。
    - `current_bw_load (float)`: RSU当前已分配或正在使用的带宽。
    - `dj_history_this_slot (list)`: 存储当前仿真时隙内，每次调用 `get_adaptive_bucket_width`
      时计算出的 `dj` 值的列表。
    - `reset_loads()`: 在每个仿真时隙开始时调用，用于重置 `current_cpu_load`,
      `current_bw_load` 和 `dj_history_this_slot`。
    - `cache_lookup(hash_value)`: 根据LSH哈希值在缓存中查找条目。
    - `cache_add(hash_value, result, feature_vector)`: 将新的子任务结果添加到缓存中。
      如果缓存已满，则根据 `cache_order_deque` 的顺序（LRU）驱逐旧条目。
    - `cache_evict()`: 从缓存中驱逐最久未使用的条目。
    - `get_adaptive_bucket_width(d0=None)`: 动态计算A-LSH算法的桶宽度 `dj`。
      该宽度根据RSU的当前CPU负载 `fj_used` 和总CPU能力 `fj` 进行调整，
      并受到配置参数 `d0` (初始桶宽), `μ` (负载敏感度) 和 `ψ` (最大宽度调整因子) 的影响。
      这个自适应机制是实现高效缓存复用的关键，因为它允许LSH的粒度随RSU负载而变化。

    RSU的缓存和自适应桶宽机制旨在通过复用相似子任务的计算结果来减少重复计算，
    从而降低任务处理延迟。
    """
    def __init__(self, id):
        self.id = id
        self.total_cpu_frequency = np.random.uniform(config.RSU_CPU_FREQ_TOTAL_MIN, config.RSU_CPU_FREQ_TOTAL_MAX) # fj
        self.total_bandwidth = config.RSU_BANDWIDTH_TOTAL # bj
        self.cache = {}
        self.cache_order_deque = deque(maxlen=config.RSU_CACHE_CAPACITY)
        self.current_cpu_load = 0
        self.current_bw_load = 0
        self.dj_history_this_slot = []

        if self.total_cpu_frequency <= 0:
             self.total_cpu_frequency = config.RSU_CPU_FREQ_TOTAL_MIN

    def reset_loads(self):
        """
        重置RSU在当前仿真时隙的负载信息和 `dj` 历史记录。

        此方法通常在每个新的仿真时隙开始时被 `VECEnvironment.step()` 调用。
        它将RSU的当前CPU负载 (`current_cpu_load`) 和带宽负载 (`current_bw_load`)
        都重置为0，并清空存储本时隙内计算出的所有 `dj` 值的列表
        (`dj_history_this_slot`)。
        这确保了每个时隙的负载计算和 `dj` 值收集都是从一个干净的状态开始。
        """
        self.current_cpu_load = 0
        self.current_bw_load = 0
        self.dj_history_this_slot = []

    def cache_lookup(self, hash_value):
        """
        根据LSH哈希值在RSU的缓存中查找条目。

        参数:
            hash_value (tuple): 要查找的子任务的LSH哈希值。

        返回:
            dict or None: 如果在缓存中找到匹配的哈希值，则返回对应的缓存条目字典。
                          该字典通常包含 {'result': ..., 'feature_vector': ..., 'timestamp': ...}。
                          如果未找到匹配项，则返回 `None`。
        """
        return self.cache.get(hash_value)

    def cache_add(self, hash_value, result, feature_vector):
        """
        将一个新的子任务计算结果及其相关信息添加到RSU的缓存中。

        如果具有相同 `hash_value` 的条目已存在于缓存中，则更新该条目的
        `result` 和 `timestamp`，并将其移到 `cache_order_deque` (用于LRU策略) 的末尾，
        表示最近被访问。

        如果 `hash_value` 不在缓存中：
        1.  检查缓存是否已达到其容量 (`config.RSU_CACHE_CAPACITY`)。
        2.  如果缓存已满，则调用 `self.cache_evict()` 来移除最久未使用的条目。
        3.  创建一个新的缓存条目，包含 `result`、`feature_vector` 和当前时间戳。
        4.  将新的 `hash_value` 添加到 `cache_order_deque` 的末尾。

        参数:
            hash_value (tuple): 要添加的子任务的LSH哈希值，作为缓存的键。
            result (any): 子任务的计算结果（或其模拟表示）。
            feature_vector (np.ndarray): 子任务的特征向量，与结果一起存储，
                                         用于后续可能的精确比较（如果需要）。
        """
        if hash_value in self.cache:
            self.cache[hash_value]['result'] = result
            self.cache[hash_value]['timestamp'] = np.datetime64('now')
            try:
                self.cache_order_deque.remove(hash_value)
            except ValueError: pass
            self.cache_order_deque.append(hash_value)
        else:
            if len(self.cache) >= config.RSU_CACHE_CAPACITY:
                self.cache_evict()
            timestamp = np.datetime64('now')
            self.cache[hash_value] = {'result': result, 'feature_vector': feature_vector, 'timestamp': timestamp}
            self.cache_order_deque.append(hash_value)

    def cache_evict(self):
        """
        从RSU的缓存中驱逐一个条目。

        此方法实现了基于LRU（Least Recently Used，最近最少使用）的缓存替换策略。
        它从 `cache_order_deque` 的左端（即最早添加或最久未被更新的条目）
        弹出一个键 (`evict_key`)，然后在主缓存字典 `self.cache` 中删除
        与该键对应的条目。

        如果 `cache_order_deque` 为空（即缓存为空），则此方法不执行任何操作。
        """
        if not self.cache_order_deque:
            return
        evict_key = self.cache_order_deque.popleft()
        if evict_key in self.cache:
            del self.cache[evict_key]

    # --- !! 使用原始公式计算 dj !! ---
    def get_adaptive_bucket_width(self, d0=None):
        """
        计算并返回A-LSH（Adaptive Locality Sensitive Hashing）的自适应桶宽度 `dj`。

        桶宽度 `dj` 的自适应调整是根据RSU当前的CPU负载情况动态进行的，
        旨在优化缓存复用的效果。当RSU负载较低时，桶宽度可以较大，允许更多不完全
        相同的子任务被哈希到同一个桶中，从而增加复用机会。当RSU负载较高时，
        桶宽度会减小，使得哈希更具区分性，减少错误复用的可能性。

        计算公式（基于论文中的Eq 6，但具体实现可能略有调整）：
        `dj = d0 * (1 + ψ / (1 + exp(μ * (fj - fj_used))))`
        其中:
        - `d0`: 初始（或基础）桶宽度。如果未提供，则使用 `config.INITIAL_BUCKET_WIDTH_D0`。
        - `ψ` (psi): 最大宽度调整因子 (`config.MAX_WIDTH_ADJUST_FACTOR`)。
        - `μ` (mu): 负载阈值敏感度 (`config.LOAD_THRESHOLD_SENSITIVITY`)。
        - `fj`: RSU的总CPU计算能力 (`self.total_cpu_frequency`)。
        - `fj_used`: RSU当前已分配或正在使用的CPU计算能力 (`self.current_cpu_load`)。

        该方法还包括一些数值稳定性处理，例如限制指数的范围以防止溢出，
        以及在计算结果异常时提供回退逻辑。计算出的 `dj` 值会被记录在
        `self.dj_history_this_slot` 列表中，并确保返回的 `dj` 值有一个最小下限（例如1e-6）。

        参数:
            d0 (float, optional): 初始桶宽度。如果为 `None`，则使用配置文件中的默认值。

        返回:
            float: 计算得到的自适应桶宽度 `dj`。
        """
        if d0 is None:
            d0 = config.INITIAL_BUCKET_WIDTH_D0

         fj_used = self.current_cpu_load
         fj = self.total_cpu_frequency
         psi = config.MAX_WIDTH_ADJUST_FACTOR
         # --- 使用原始的 mu ---
         mu = config.LOAD_THRESHOLD_SENSITIVITY # 从 config 获取 mu
         # -------------------
         dj = d0

         # RSU 容量小于等于0时，直接返回d0
         if fj <= 1e-9:
             # print(f"Warning: RSU {self.id} total_cpu_frequency is non-positive ({fj}). Using default dj={d0}.")
             if np.isfinite(dj): self.dj_history_this_slot.append(dj)
             return max(1e-6, dj)

         # --- 使用原始指数项 (绝对差值) ---
         exponent = mu * (fj - fj_used)
         # ------------------------------

         # 限制指数范围防止计算溢出
         MAX_EXPONENT = 700.0 # exp(709) 接近 float64 上限
         MIN_EXPONENT = -700.0
         clamped_exponent = np.clip(exponent, MIN_EXPONENT, MAX_EXPONENT)

         try:
             adaptive_term = 1 + np.exp(clamped_exponent)
             # Fallback 逻辑处理 exp() 结果异常或 adaptive_term 接近 0 的情况
             if adaptive_term <= 1e-9 or np.isinf(adaptive_term) or np.isnan(adaptive_term):
                 # print(f"DEBUG: Fallback triggered. exponent={exponent:.4f}, clamped={clamped_exponent:.4f}, term={adaptive_term}")
                 # Fallback 基于原始（未钳位）的 exponent 符号
                 dj = d0 if exponent > 0 else d0 * (1 + psi)
             else:
                 dj = d0 * (1 + psi / adaptive_term)
         except Exception as e:
             print(f"Warning: Numerical issue in get_adaptive_bucket_width (exponent={exponent:.4f}, fj={fj:.2e}, fj_used={fj_used:.2e}): {e}. Falling back.")
             traceback.print_exc()
             # Fallback 基于原始 exponent 符号
             dj = d0 if exponent > 0 else d0 * (1 + psi)

         # 记录计算出的 dj 值
         final_dj = max(1e-6, dj) # 保证 dj > 0
         if np.isfinite(final_dj):
             self.dj_history_this_slot.append(final_dj)
             # 可选调试打印
             # print(f"DEBUG: RSU {self.id} calculated dj={final_dj:.4f} (load={fj_used:.2e}, fj={fj:.2e}, d0={d0}, mu={mu:.1e})")
         return final_dj
    # --- 结束修改 ---

class VECEnvironment:
    """
    模拟整个车载边缘计算（VEC）环境。

    该类负责管理VEC系统中的所有实体（车辆和RSU），模拟动态变化的
    环境条件（如无线信道），并在每个仿真时隙生成新的计算任务。

    属性:
        vehicles (list[Vehicle]): 环境中所有 `Vehicle` 对象的列表。
        rsus (list[RSU]): 环境中所有 `RSU` 对象的列表。
        channel_gains (np.ndarray): 一个二维NumPy数组，存储每对车辆-RSU之间的
                                    信道增益 `h_ij`。这些增益会随时间动态变化。
        current_slot (int): 当前仿真的时间槽计数器。

    主要方法:
        _initialize_channel_gains(): 在环境初始化时，随机生成车辆与RSU之间的初始信道增益。
        update_channel_conditions(): 模拟信道条件的动态变化。在每个时间槽，
                                     会给当前的信道增益添加随机噪声。
        get_channel_gain(vehicle_idx, rsu_idx): 获取特定车辆和RSU之间的当前信道增益。
        get_rsu(rsu_idx)` / `get_rsu_by_id(rsu_id)`: 根据索引或ID获取RSU实例。
        get_vehicle(vehicle_idx)` / `get_vehicle_by_id(vehicle_id)`: 根据索引或ID获取车辆实例。
        generate_tasks(): 在当前时间槽为随机选择的车辆生成一批新的计算任务 (`TaskDAG` 对象)。
                          任务数量遵循泊松分布，平均值为 `config.NUM_TASKS_PER_SLOT`。
        step(): 将仿真推进一个时间槽。这个方法会：
                1. 增加 `current_slot` 计数器。
                2. 调用 `update_channel_conditions()` 更新信道。
                3. 调用每个RSU的 `reset_loads()` 方法来清除它们上一个时隙的负载信息和 `dj` 历史。
                4. 调用 `generate_tasks()` 生成当前时隙的新任务。
                5. 返回新生成的任务列表。

    `VECEnvironment` 是整个仿真的核心驱动者，它为各种卸载算法提供了一个动态的、
    可交互的测试平台。
    """
    def __init__(self):
        self.vehicles = [Vehicle(id=f"veh_{i}") for i in range(config.NUM_VEHICLES)]
        self.rsus = [RSU(id=f"rsu_{j}") for j in range(config.NUM_RSUS)]
        for rsu in self.rsus:
             if rsu.total_cpu_frequency <= 0:
                  rsu.total_cpu_frequency = config.RSU_CPU_FREQ_TOTAL_MIN
        self.channel_gains = self._initialize_channel_gains() # h_ij matrix
        self.current_slot = 0

    def _initialize_channel_gains(self):
        """
        初始化车辆与RSU之间的信道增益矩阵。

        信道增益 `h_ij` 反映了从车辆 `i` 到RSU `j` 的无线信道质量。
        此方法在VEC环境设置时被调用一次，为每对车辆-RSU生成一个初始的信道增益值。
        这些值是从一个均匀分布中随机抽取的，范围由 `config.CHANNEL_GAIN_MIN`
        和 `config.CHANNEL_GAIN_MAX` 定义。
        确保增益值是正的，并有一个小的最小下限。

        返回:
            np.ndarray: 一个形状为 `(NUM_VEHICLES, NUM_RSUS)` 的NumPy数组，
                        其中包含了初始的信道增益值。
        """
        min_gain = max(1e-9, config.CHANNEL_GAIN_MIN)
        max_gain = max(min_gain + 1e-9, config.CHANNEL_GAIN_MAX)
        gains = np.random.uniform(min_gain, max_gain,
                                   size=(config.NUM_VEHICLES, config.NUM_RSUS))
        return gains

    def update_channel_conditions(self):
        """
        模拟并更新车辆与RSU之间无线信道条件的动态变化。

        此方法在每个仿真时隙 (`self.step()` 中) 被调用，以反映信道随时间波动的情况。
        它通过在当前的信道增益矩阵 `self.channel_gains` 上添加一个从正态分布
        (均值为0，标准差为 `config.CHANNEL_NOISE_STDDEV`) 中采样的随机噪声来实现。
        更新后的信道增益值会被裁剪，以确保它们保持正值且不小于一个小的下限 (1e-9)。

        副作用:
            - 修改 `self.channel_gains` 矩阵。
        """
        noise = np.random.normal(0, config.CHANNEL_NOISE_STDDEV,
                                 size=(config.NUM_VEHICLES, config.NUM_RSUS))
        self.channel_gains += noise
        self.channel_gains = np.maximum(1e-9, self.channel_gains)

    def get_channel_gain(self, vehicle_idx, rsu_idx):
        if 0 <= vehicle_idx < config.NUM_VEHICLES and 0 <= rsu_idx < config.NUM_RSUS:
             return self.channel_gains[vehicle_idx, rsu_idx]
        else:
             print(f"Warning: Invalid index for get_channel_gain ({vehicle_idx}, {rsu_idx})")
             return 1e-9

    def get_rsu(self, rsu_idx):
        if 0 <= rsu_idx < len(self.rsus):
            return self.rsus[rsu_idx]
        return None

    def get_vehicle(self, vehicle_idx):
         if 0 <= vehicle_idx < len(self.vehicles):
            return self.vehicles[vehicle_idx]
         return None

    def generate_tasks(self):
        """
        为当前仿真时隙生成一批新的计算任务。

        任务的数量是根据泊松分布随机生成的，其平均值由 `config.NUM_TASKS_PER_SLOT` 控制。
        如果 `config.NUM_TASKS_PER_SLOT` 小于或等于0，或者环境中没有车辆，则不生成任务。
        每个生成的任务都是一个 `TaskDAG` 对象，它会被随机分配给环境中的一个有效车辆作为发起者。
        任务ID会包含当前的时隙号和任务在本时隙的序号，以确保唯一性。

        返回:
            list[TaskDAG]: 一个包含新生成的 `TaskDAG` 对象的列表。
                           如果无法生成任务（例如，没有车辆或任务生成数量为0），
                           则返回空列表。
        """
        tasks = []
        if config.NUM_TASKS_PER_SLOT <= 0: return tasks
        num_tasks = np.random.poisson(max(1, config.NUM_TASKS_PER_SLOT))
        if num_tasks == 0: return tasks
        vehicle_ids = [v.id for v in self.vehicles]
        if not vehicle_ids: return tasks
        for k in range(num_tasks):
            veh_id = np.random.choice(vehicle_ids)
            task_id = f"task_{self.current_slot}_{k}"
            tasks.append(TaskDAG(id=task_id, vehicle_id=veh_id))
        return tasks

    def step(self):
        """
        将VEC仿真环境推进一个时间槽。

        此方法是仿真循环的核心驱动步骤，它执行以下操作：
        1.  增加当前时间槽计数器 (`self.current_slot`)。
        2.  调用 `self.update_channel_conditions()` 来模拟无线信道的变化。
        3.  遍历环境中的所有RSU，并调用每个RSU的 `reset_loads()` 方法。
            这将清除RSU上一个时隙的CPU和带宽负载记录，以及它们的 `dj` 历史，
            为当前新时隙的计算做准备。
        4.  调用 `self.generate_tasks()` 来生成当前时隙到达的新计算任务。
        5.  返回新生成的任务列表。

        返回:
            list[TaskDAG]: 在这个新的仿真时隙中生成的一批 `TaskDAG` 对象。
        """
        self.current_slot += 1
        self.update_channel_conditions()
        for rsu in self.rsus:
            rsu.reset_loads()
        new_tasks = self.generate_tasks()
        return new_tasks

    def get_rsu_by_id(self, rsu_id):
        for rsu in self.rsus:
            if rsu.id == rsu_id:
                return rsu
        return None

    def get_vehicle_by_id(self, vehicle_id):
        for veh in self.vehicles:
            if veh.id == vehicle_id:
                return veh
        return None