# environment.py
import numpy as np
import config
from task import TaskDAG
from collections import deque
import traceback

class Vehicle:
    """Represents a vehicle in the VEC system."""
    def __init__(self, id):
        self.id = id
        self.cpu_frequency = config.VEHICLE_CPU_FREQ # fi
        self.power_compute = config.VEHICLE_POWER_COMPUTE # Pi
        self.power_transmit = config.VEHICLE_POWER_TRANSMIT # Pi,j_trans

class RSU:
    """Represents a Roadside Unit."""
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
        self.current_cpu_load = 0
        self.current_bw_load = 0
        self.dj_history_this_slot = []

    def cache_lookup(self, hash_value):
        return self.cache.get(hash_value)

    def cache_add(self, hash_value, result, feature_vector):
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
        if not self.cache_order_deque:
            return
        evict_key = self.cache_order_deque.popleft()
        if evict_key in self.cache:
            del self.cache[evict_key]

    # --- !! 使用原始公式计算 dj !! ---
    def get_adaptive_bucket_width(self, d0=None):
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
    """Simulates the Vehicular Edge Computing environment."""
    def __init__(self):
        self.vehicles = [Vehicle(id=f"veh_{i}") for i in range(config.NUM_VEHICLES)]
        self.rsus = [RSU(id=f"rsu_{j}") for j in range(config.NUM_RSUS)]
        for rsu in self.rsus:
             if rsu.total_cpu_frequency <= 0:
                  rsu.total_cpu_frequency = config.RSU_CPU_FREQ_TOTAL_MIN
        self.channel_gains = self._initialize_channel_gains() # h_ij matrix
        self.current_slot = 0

    def _initialize_channel_gains(self):
        min_gain = max(1e-9, config.CHANNEL_GAIN_MIN)
        max_gain = max(min_gain + 1e-9, config.CHANNEL_GAIN_MAX)
        gains = np.random.uniform(min_gain, max_gain,
                                   size=(config.NUM_VEHICLES, config.NUM_RSUS))
        return gains

    def update_channel_conditions(self):
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