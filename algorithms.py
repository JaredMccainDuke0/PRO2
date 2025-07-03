"""
此模块实现了车载边缘计算（VEC）环境中的各种任务卸载算法。

主要包括以下几个部分：
1.  `ReplayBuffer`: 一个用于深度强化学习（DRL）的经验回放缓冲区。
2.  `BaseAlgorithm`: 一个抽象基类，定义了所有卸载算法的通用接口和基本功能，
    例如结果收集和在仿真时隙中运行任务。
3.  `ExpertDataset`: 一个 PyTorch `Dataset` 类，用于处理和加载专家演示数据，
    供模仿学习（IL）模型训练使用。
4.  `GAT_REUSE_IL`: 基于图注意力网络（GAT）的模仿学习算法，集成了子任务结果复用机制。
    它通过学习专家策略来进行卸载决策和资源分配。
5.  `GAT_IL_NoReuse`: `GAT_REUSE_IL` 的一个变体，不使用子任务结果复用机制，
    作为对比基准。
6.  `GAT_REUSE_DRL`: 基于GAT的深度强化学习（Actor-Critic）算法，同样集成了子任务复用机制。
    它通过与环境交互学习最优策略，旨在最小化系统成本（如延迟和能耗）。

这些算法旨在决定计算任务是在车辆本地执行还是卸载到附近的RSU执行，
并确定相应的资源分配（如CPU频率、带宽）。
复用机制允许RSU缓存和重用之前计算过的子任务结果，以减少计算延迟。
"""
# algorithms.py
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from collections import deque
from torch_geometric.data import Batch, Data # Import Batch and Data
# --- CORRECTED: Import PyG DataLoader ---
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import Dataset # Import torch Dataset
import traceback # For detailed error printing

import config
from task import TaskDAG, SubTask
from environment import VECEnvironment, RSU, Vehicle # Import RSU here
from gat_model import GATNetwork, ILDecisionModel, ActorNetwork, CriticNetwork, dag_to_pyg_data
import utils

# --- ReplayBuffer ---
class ReplayBuffer:
    """
    经验回放缓冲区，用于存储和采样深度强化学习（DRL）中的经验。

    经验通常以 (状态, 动作, 奖励, 下一个状态, 完成标志) 的元组形式存储。
    缓冲区具有固定容量，当缓冲区满时，添加新的经验会替换掉最旧的经验（FIFO）。
    `sample` 方法允许从缓冲区中随机抽取一批经验用于训练DRL智能体。

    属性:
        buffer (deque): 一个双端队列，用于存储经验元组。
        capacity (int): 缓冲区的最大容量。
    """
    def __init__(self, capacity=config.DRL_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)
    def add(self, state_data, action_details_train, reward, next_state_data, done):
        """
        向经验回放缓冲区中添加一条新的经验。

        经验元组包含状态、为训练准备的动作细节、奖励、下一个状态和完成标志。
        在存入缓冲区之前，状态数据 (`state_data`) 会被转移到CPU。

        参数:
            state_data (torch_geometric.data.Data): 当前状态的PyG图数据对象。
            action_details_train (dict): 一个包含用于训练的动作张量（例如，离散动作索引，
                                         连续动作值，可能还有log_probs）的字典。
                                         这些张量应该是已经 .cpu().detach() 过的。
            reward (float): 在该状态下执行动作后获得的奖励。
            next_state_data (torch_geometric.data.Data or None): 转移到的下一个状态的PyG图数据对象。
                                                                 如果当前状态是终止状态，则为 None。
            done (bool): 一个布尔标志，指示当前episode是否在此步骤结束。
        """
        # Ensure state_data is on CPU before adding
        self.buffer.append((state_data.cpu(), action_details_train, reward, next_state_data, done))
    def sample(self, batch_size, device):
        """
        从经验回放缓冲区中随机采样一批经验。

        如果缓冲区中的经验数量少于 `batch_size`，则不进行采样并返回None。
        采样得到的各个部分（状态、动作、奖励等）会被转换成PyTorch张量，
        并且状态数据会被批量处理 (`torch_geometric.data.Batch.from_data_list`)
        并转移到指定的 `device`。

        参数:
            batch_size (int): 要采样的经验数量。
            device (torch.device): 采样出的张量（尤其是状态批次）应转移到的设备 (CPU 或 CUDA)。

        返回:
            tuple: 包含以下元素的元组：
                - `state_batch (torch_geometric.data.Batch or None)`:
                  批量处理后的状态图数据，已转移到 `device`。如果采样失败则为None。
                - `actions_batch (dict or None)`:
                  一个字典，包含批量处理后的动作张量（例如 'offload', 'cpu', 'bw'），
                  已转移到 `device`。如果采样失败则为None。
                - `rewards_tensor (torch.Tensor or None)`:
                  奖励的张量，形状为 `[batch_size, 1]`，已转移到 `device`。如果采样失败则为None。
                - `next_state_batch (None)`:
                  当前实现中，下一个状态批次总是返回None（因为DRL逻辑假设每个任务完成即episode结束）。
                - `dones_tensor (torch.Tensor or None)`:
                  完成标志的张量，形状为 `[batch_size, 1]`，已转移到 `device`。如果采样失败则为None。
        """
        if len(self.buffer) < batch_size: return None, None, None, None, None
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states_pyg_cpu, actions_details_list, rewards, next_states_pyg, dones = zip(*[self.buffer[i] for i in indices])
        valid_states = [s for s in states_pyg_cpu if s is not None]
        if len(valid_states) != batch_size: print(f"Warning: Sampled {len(valid_states)} valid states, expected {batch_size}."); return None, None, None, None, None
        try:
             # Ensure list contains only Data objects before batching
             if not all(isinstance(s, Data) for s in valid_states):
                 print(f"Error: Invalid items in states list for batching: {[type(s) for s in valid_states]}")
                 return None, None, None, None, None
             state_batch = Batch.from_data_list(valid_states).to(device)
        except Exception as e: print(f"Error batching PyG states: {e}"); traceback.print_exc(); return None, None, None, None, None
        rewards_tensor = torch.tensor(rewards, dtype=torch.float).unsqueeze(1).to(device)
        dones_tensor = torch.tensor(dones, dtype=torch.float).unsqueeze(1).to(device)
        # Stack actions ONLY (log_prob removed)
        offload_actions = torch.stack([details['offload_action'].to(device) for details in actions_details_list])
        cpu_actions = torch.stack([details['cpu_action'].to(device) for details in actions_details_list])
        bw_actions = torch.stack([details['bw_action'].to(device) for details in actions_details_list])
        actions_batch = {'offload': offload_actions, 'cpu': cpu_actions, 'bw': bw_actions}
        return state_batch, actions_batch, rewards_tensor, None, dones_tensor # Returning None for next_state
    def __len__(self): return len(self.buffer)


# --- BaseAlgorithm ---
class BaseAlgorithm:
    """
    所有卸载算法的抽象基类。

    该类定义了卸载算法所需的基本结构和通用功能，包括：
    - 初始化算法实例，关联VEC环境。
    - `decide_and_execute(task)`: 一个抽象方法，需要在子类中实现，
      用于对给定的任务做出卸载决策并模拟执行。
    - `run_slot(tasks)`: 处理单个仿真时隙中的所有任务。它会为每个任务调用
      `decide_and_execute`，收集并聚合该时隙的性能统计数据（如目标值、
      延迟、能耗、缓存命中次数），并收集RSU计算出的所有 `dj` (自适应桶宽) 值。
    - `get_results()`: 返回算法在整个仿真过程中收集到的累积性能结果。

    属性:
        env (VECEnvironment): 算法运行所在的VEC环境实例。
        name (str): 算法的名称。
        results (dict): 存储仿真结果的字典，键包括 'objective', 'latency',
                        'energy', 'cache_hits'。
        all_djs_this_run (list): 存储在当前仿真运行中，所有RSU在所有时隙
                                 计算出的 `dj` 值的列表。
    """
    def __init__(self, env: VECEnvironment, name="Base"):
        self.env = env
        self.name = name
        self.results = {'objective': [], 'latency': [], 'energy': [], 'cache_hits': []}
        # --- 存储本次仿真运行的所有 dj 值 ---
        self.all_djs_this_run = []
        # ---------------------------------------

    def decide_and_execute(self, task: TaskDAG):
        raise NotImplementedError

    # --- run_slot (收集 dj 值) ---
    def run_slot(self, tasks):
        """
        处理当前仿真时隙（slot）内到达的一批任务。

        对于每个任务，此方法会调用子类实现的 `decide_and_execute` 方法来获取
        卸载决策并模拟任务执行。然后，它会收集并聚合该时隙内所有已处理任务的
        性能统计数据，包括平均目标值、平均延迟、平均能耗和总缓存命中次数。

        此外，此方法还负责从环境中的所有RSU收集在本时隙内计算出的所有
        自适应桶宽 `dj` 值，并将它们存储在 `self.all_djs_this_run` 列表中，
        用于后续计算平均 `dj`。

        参数:
            tasks (list[TaskDAG]): 当前仿真时隙需要处理的任务DAG对象列表。

        副作用:
            - 更新 `self.results` 字典，追加当前时隙的聚合性能指标。
            - 更新 `self.all_djs_this_run` 列表，追加当前时隙收集到的所有 `dj` 值。
            - 打印当前时隙的性能摘要信息（周期性打印，例如每10个时隙）。
            - RSU的内部状态（如 `dj_history_this_slot`）会在 `decide_and_execute`
              调用 `rsu.get_adaptive_bucket_width()` 时被修改。
              注意：RSU的CPU和带宽负载以及 `dj_history_this_slot` 通常在
              `env.step()` 方法中被重置，而不是在此方法中。
        """
        slot_objective = 0; slot_latency = 0; slot_energy = 0; slot_cache_hits = 0; processed_tasks = 0; task_results = []
        # RSU 负载和 dj 历史在 env.step() 中通过 RSU.reset_loads() 重置

        # --- 处理本 slot 的所有任务 ---
        for task in tasks:
            decision, stats = self.decide_and_execute(task) # decide_and_execute 内部会调用 RSU.get_adaptive_bucket_width
            if stats:
                # Ensure stats dict contains expected keys with finite values before appending
                if all(np.isfinite(stats.get(k, np.inf)) for k in ['latency', 'energy', 'objective']):
                     task_results.append(stats)
                     processed_tasks += 1
                else:
                     print(f"Warning: Skipping task {task.id} due to non-finite stats: {stats}")


        # --- 计算本 slot 的统计数据 ---
        if processed_tasks > 0:
            valid_objectives = [s.get('objective', np.inf) for s in task_results if np.isfinite(s.get('objective', np.inf))]
            valid_latencies = [s.get('latency', np.inf) for s in task_results if np.isfinite(s.get('latency', np.inf))]
            valid_energies = [s.get('energy', np.inf) for s in task_results if np.isfinite(s.get('energy', np.inf))]

            slot_objective = np.mean(valid_objectives) if valid_objectives else 0.0
            slot_latency = np.mean(valid_latencies) if valid_latencies else 0.0
            slot_energy = np.mean(valid_energies) if valid_energies else 0.0
            # Cache hits can be summed even if other stats are inf
            slot_cache_hits = np.sum([s.get('cache_hit_count', 0) for s in task_results])

            if np.isnan(slot_objective): slot_objective = 0.0
            if np.isnan(slot_latency): slot_latency = 0.0
            if np.isnan(slot_energy): slot_energy = 0.0

        self.results['objective'].append(slot_objective)
        self.results['latency'].append(slot_latency)
        self.results['energy'].append(slot_energy)
        self.results['cache_hits'].append(slot_cache_hits)

        # --- 收集本 slot 计算出的所有 dj 值 ---
        slot_djs = []
        if hasattr(self.env, 'rsus'): # Ensure env has rsus attribute
            for rsu in self.env.rsus:
                if hasattr(rsu, 'dj_history_this_slot'): # Ensure rsu has the attribute
                    # dj_history_this_slot 包含了本 slot 内所有对 get_adaptive_bucket_width 的调用结果
                    valid_slot_djs = [d for d in rsu.dj_history_this_slot if np.isfinite(d)]
                    if valid_slot_djs:
                        slot_djs.extend(valid_slot_djs)
                else:
                     print(f"Warning: RSU {getattr(rsu, 'id', 'N/A')} missing 'dj_history_this_slot'")

        if slot_djs:
             # 将本 slot 的 dj 值添加到整个运行的记录中
             if hasattr(self, 'all_djs_this_run'): # Check attribute exists
                 self.all_djs_this_run.extend(slot_djs)
             else:
                  print(f"Warning: Algorithm instance {self.name} missing 'all_djs_this_run'")
        # -----------------------------------------

        # --- 打印信息 (包括 dj) ---
        if self.env.current_slot % 10 == 1 or self.env.current_slot == config.SIMULATION_SLOTS:
             avg_dj_so_far_val = np.nan
             if hasattr(self, 'all_djs_this_run'):
                 valid_djs_so_far = [d for d in self.all_djs_this_run if np.isfinite(d)]
                 if valid_djs_so_far:
                     avg_dj_so_far_val = np.mean(valid_djs_so_far)

             print(f"[{self.name} Slot {self.env.current_slot}] Avg Objective: {slot_objective:.4f}, Avg Latency: {slot_latency:.4f}, Avg Energy: {slot_energy:.4f}, Cache Hits: {slot_cache_hits}, Avg dj so far: {avg_dj_so_far_val:.4f}")

    # --- get_results (保持不变) ---
    def get_results(self):
        return self.results


# --- Dataset Class for Expert Data ---
class ExpertDataset(Dataset):
    """
    一个 PyTorch `Dataset` 类，用于封装和处理模仿学习所需的专家演示数据。

    专家数据通常由 (状态, 动作) 对组成，其中状态是任务DAG的图表示，
    动作是专家对该任务做出的卸载决策和资源分配请求。
    此类负责将原始的专家数据列表（通常是 NetworkX 图和决策字典）
    转换为 PyTorch Geometric `Data` 对象，并附加目标（专家动作）作为标签。
    转换后的 `Data` 对象可以直接被 `torch_geometric.loader.DataLoader` 用于批量加载
    和模型训练。

    在初始化过程中，它会：
    1. 遍历原始专家数据列表。
    2. 对每个样本，使用 `dag_to_pyg_data` 函数将任务DAG转换为 PyG `Data` 对象。
    3. 从专家决策中提取卸载目标、CPU分配请求和带宽分配请求。
    4. 将这些目标作为张量附加到 PyG `Data` 对象的 `y_offload`, `y_cpu`, `y_bw` 属性上。
    5. 过滤掉无效或转换失败的样本。

    `__len__` 方法返回数据集中有效样本的数量。
    `__getitem__` 方法根据索引返回一个处理好的 PyG `Data` 对象。
    """
    def __init__(self, expert_data_list):
        """
        初始化 ExpertDataset。

        此构造函数接收一个原始专家数据列表，并将其处理成适用于PyTorch Geometric模型训练的格式。
        每个专家样本（通常是 NetworkX 图和专家决策字典）被转换为一个
        `torch_geometric.data.Data` 对象。专家决策（卸载目标、CPU分配、带宽分配）
        作为目标标签（y_offload, y_cpu, y_bw）附加到每个 `Data` 对象上。

        处理步骤：
        1. 遍历 `expert_data_list` 中的每个 `(graph_nx, decision)` 元组。
        2. 检查 `decision` 是否有效且包含 'offload_target'。
        3. 使用 `dag_to_pyg_data` 将 `graph_nx`（NetworkX图）转换为 PyG `Data` 对象。
        4. 验证转换后的 `pyg_data` 对象及其节点特征 `x` 和边索引 `edge_index` 是否有效。
        5. 检查节点特征维度是否符合 `config.FEATURE_DIM + 1`。
        6. 从 `decision` 字典中提取卸载目标、CPU分配请求和带宽分配请求，
           并将它们转换为PyTorch张量，分别存储在 `pyg_data.y_offload`,
           `pyg_data.y_cpu`, `pyg_data.y_bw` 中。
        7. 将处理好的 `pyg_data` 对象添加到内部的 `self.data` 列表中。
        8. 统计并打印成功处理的样本数和因错误或无效数据而被跳过的样本数。

        参数:
            expert_data_list (list): 一个列表，其中每个元素是一个元组
                                     `(graph_nx, decision)`。
                                     `graph_nx` 是任务DAG的NetworkX图表示。
                                     `decision` 是一个字典，包含专家的卸载决策
                                     （例如，{'offload_target': int,
                                     'cpu_alloc_request': float,
                                     'bw_alloc_request': float}）。
        """
        self.data = []
        print("Processing expert data for Dataset...")
        count = 0
        skipped = 0
        for graph_nx, decision in expert_data_list:
            if decision is None or 'offload_target' not in decision:
                skipped += 1
                continue
            try:
                # Pass subtask map as None, assuming features are in graph nodes
                pyg_data = dag_to_pyg_data(graph_nx, None)
                if pyg_data is None or pyg_data.x is None or pyg_data.edge_index is None:
                    print(f"Warning: dag_to_pyg_data returned invalid data for a sample. Skipping.")
                    skipped += 1
                    continue # Skip if conversion failed or data is incomplete

                # Ensure features exist and have expected dims before adding targets
                if pyg_data.x.shape[1] != config.FEATURE_DIM + 1:
                     print(f"Warning: Feature dimension mismatch in expert data. Expected {config.FEATURE_DIM + 1}, got {pyg_data.x.shape[1]}. Skipping.")
                     skipped += 1
                     continue

                # Store targets directly with the data object (on CPU)
                pyg_data.y_offload = torch.tensor([decision['offload_target']], dtype=torch.long)
                cpu_req = float(decision.get('cpu_alloc_request', 0.0))
                bw_req = float(decision.get('bw_alloc_request', 0.0))
                pyg_data.y_cpu = torch.tensor([cpu_req], dtype=torch.float)
                pyg_data.y_bw = torch.tensor([bw_req], dtype=torch.float)

                self.data.append(pyg_data)
                count += 1
            except Exception as e:
                print(f"Skipping expert sample due to error during processing: {e}")
                traceback.print_exc()
                skipped += 1
        print(f"Processed {count} valid expert samples into Dataset. Skipped {skipped} invalid samples.")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return the PyG Data object directly
        return self.data[idx]

# --- GAT_REUSE_IL Implementation ---
class GAT_REUSE_IL(BaseAlgorithm):
    """
    基于图注意力网络（GAT）的模仿学习（IL）卸载算法，集成了子任务结果复用机制。

    该算法通过学习专家演示数据（由启发式策略生成）来进行任务卸载决策和资源分配。
    其核心组成部分包括：
    - `model (ILDecisionModel)`: 一个基于GAT的神经网络，用于从任务DAG的图表示中
      预测卸载目标（本地或特定RSU）、CPU分配比例和带宽分配比例。
    - `optimizer`: Adam优化器，用于训练 `model`。
    - `train(expert_data)`: 训练方法，使用提供的专家数据集通过监督学习方式
      更新模型参数，目标是最小化模型预测与专家决策之间的差异。
    - `decide_and_execute(task)`: 决策与执行方法。对于给定的任务DAG，
      它首先使用训练好的 `model` 获取卸载决策和资源分配建议。
      然后，它模拟任务的执行：
        - 如果决策为本地执行，则计算本地执行的延迟和能耗。
        - 如果决策为卸载到RSU，则首先计算传输延迟和能耗。接着，对于任务中的每个子任务，
          它会利用A-LSH（Adaptive Locality Sensitive Hashing）机制在目标RSU的缓存中
          查找相似的已计算子任务。如果找到足够相似的匹配项（相似度低于阈值 `δ`），
          则复用缓存结果，子任务计算延迟视为0。否则，在RSU上计算该子任务，
          并将结果（特征向量和模拟结果）存入RSU缓存。
      最后，该方法返回决策详情和执行后的性能统计（延迟、能耗、目标值、缓存命中次数）。
    - `execute_locally(task, vehicle)`: 一个辅助方法，专门处理任务在车辆本地执行的逻辑。

    该算法旨在通过模仿专家行为并结合高效的子任务复用，来实现较低的系统成本
    （延迟和能耗的加权和）。LSH参数（如桶宽 `d0`、敏感度 `μ`）和复用相似性阈值 `δ`
    是影响其性能的关键配置。
    """
    def __init__(self, env: VECEnvironment):
        super().__init__(env, name="GAT-REUSE-IL")
        self.model = ILDecisionModel()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE_IL,
            weight_decay=1e-5 # Small L2 regularization
        )
        utils.initialize_lsh_params()
        self.train_loss_history = [] # Store training loss history here

    # --- train method using Mini-Batches ---
    def train(self, expert_data):
        """
        使用提供的专家数据集通过监督学习方式训练模仿学习（IL）模型。

        该方法将模型 (`self.model`) 设置为训练模式，并使用 Adam 优化器
        (`self.optimizer`) 来最小化模型预测与专家决策之间的损失。
        训练过程按指定的周期数 (`config.EPOCHS_IL`) 进行，每个周期内数据会分批处理。

        主要步骤:
        1.  检查专家数据集 `expert_data` 是否为空，如果为空则跳过训练。
        2.  确定训练设备 (CPU 或 CUDA)。
        3.  将模型转移到选定的设备。
        4.  使用 `ExpertDataset` 类将原始专家数据转换为 PyTorch `Dataset`。
        5.  创建一个 `torch_geometric.loader.PyGDataLoader` 来按批次加载数据。
        6.  对于每个训练周期 (epoch):
            a.  遍历数据加载器提供的每个批次 (batch)。
            b.  将当前批次数据转移到训练设备。
            c.  清零优化器梯度 (`self.optimizer.zero_grad()`)。
            d.  通过模型前向传播获取预测 (`offload_logits`, `cpu_pred`, `bw_pred`)。
            e.  从批次数据中提取专家目标 (`expert_offload`, `expert_cpu`, `expert_bw`)。
            f.  计算损失：
                -   `loss_offload`: 卸载决策的交叉熵损失。
                -   `loss_cpu`: CPU分配比例的均方误差损失 (MSE)。
                -   `loss_bw`: 带宽分配比例的均方误差损失 (MSE)。
                -   `loss`: 上述三个损失的加权和（权重由 `config.LOSS_WEIGHT_LAMBDA1` 等定义）。
            g.  执行反向传播 (`loss.backward()`)。
            h.  （可选）进行梯度裁剪。
            i.  更新模型参数 (`self.optimizer.step()`)。
            j.  累积当前周期的总损失。
        7.  每个周期结束后，计算并打印平均损失，并将平均损失记录到 `self.train_loss_history`。
        8.  训练完成后，将模型设置为评估模式 (`self.model.eval()`)。

        参数:
            expert_data (list): 一个包含 `(task_graph, expert_decision)` 元组的列表，
                                与 `ExpertDataset` 的输入格式相同。
        """
        print(f"[{self.name}] Starting training...")
        if not expert_data:
            print("Warning: Expert dataset empty. Skipping training.")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        print(f" Training on device: {device}")

        # Create Dataset and DataLoader
        try:
             expert_dataset_torch = ExpertDataset(expert_data)
             if len(expert_dataset_torch) == 0:
                 print("Error: No valid data loaded into ExpertDataset. Skipping training.")
                 self.train_loss_history = [0.0] * config.EPOCHS_IL # Fill history with 0
                 return
             batch_size = 32
             expert_loader = PyGDataLoader(expert_dataset_torch, batch_size=batch_size, shuffle=True, num_workers=0)
        except Exception as e:
             print(f"Error creating Dataset/DataLoader: {e}")
             traceback.print_exc()
             self.train_loss_history = [0.0] * config.EPOCHS_IL
             return

        self.model.train()
        self.train_loss_history = [] # Reset history before training

        print(f" Starting {config.EPOCHS_IL} training epochs with {len(expert_loader)} batches per epoch...")
        for epoch in range(config.EPOCHS_IL):
            total_loss = 0
            num_samples_processed = 0
            processed_batches = 0
            for batch in expert_loader:
                if batch is None:
                    print(f"Warning: DataLoader returned None batch in epoch {epoch+1}. Skipping.")
                    continue

                if not hasattr(batch, 'x') or not hasattr(batch, 'edge_index') or not hasattr(batch, 'batch') or \
                   not hasattr(batch, 'y_offload') or not hasattr(batch, 'y_cpu') or not hasattr(batch, 'y_bw'):
                    print(f"Warning: Skipping incomplete batch in epoch {epoch+1}.")
                    continue

                current_batch_size = batch.num_graphs
                if current_batch_size == 0:
                    print(f"Warning: Skipping empty batch in epoch {epoch+1}.")
                    continue

                try:
                    batch = batch.to(device)
                    self.optimizer.zero_grad()
                    offload_logits, cpu_pred, bw_pred = self.model(batch)

                    expert_offload = batch.y_offload.squeeze(-1)
                    expert_cpu = batch.y_cpu
                    expert_bw = batch.y_bw

                    # Shape checks before loss
                    if offload_logits.shape[0] != current_batch_size or \
                       cpu_pred.shape[0] != current_batch_size or \
                       bw_pred.shape[0] != current_batch_size or \
                       expert_offload.shape[0] != current_batch_size or \
                       expert_cpu.shape[0] != current_batch_size or \
                       expert_bw.shape[0] != current_batch_size:
                        print(f"Warning: Shape mismatch in batch {processed_batches}, epoch {epoch+1}. Skipping batch.")
                        continue

                    loss_offload = F.cross_entropy(offload_logits, expert_offload)
                    loss_cpu = F.mse_loss(cpu_pred, expert_cpu)
                    loss_bw = F.mse_loss(bw_pred, expert_bw)

                    loss = (config.LOSS_WEIGHT_LAMBDA1 * loss_offload +
                            config.LOSS_WEIGHT_LAMBDA2 * loss_cpu +
                            config.LOSS_WEIGHT_LAMBDA3 * loss_bw)

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # Optional clipping
                    self.optimizer.step()

                    total_loss += loss.item() * current_batch_size
                    num_samples_processed += current_batch_size
                    processed_batches += 1

                except Exception as e:
                    print(f"Error processing batch {processed_batches} in epoch {epoch+1}: {e}")
                    traceback.print_exc()
                    continue # Skip faulty batch

            if num_samples_processed > 0:
                avg_loss = total_loss / num_samples_processed
                self.train_loss_history.append(avg_loss)
                print(f"[{self.name} Epoch {epoch+1}/{config.EPOCHS_IL}] Avg Loss: {avg_loss:.6f} ({processed_batches} batches)")
            elif processed_batches > 0:
                 print(f"[{self.name} Epoch {epoch+1}/{config.EPOCHS_IL}] Processed batches but num_samples is zero?")
                 self.train_loss_history.append(0.0)
            else:
                print(f"[{self.name} Epoch {epoch+1}/{config.EPOCHS_IL}] No batches processed.")
                self.train_loss_history.append(0.0)

        print(f"[{self.name}] Training complete.")
        self.model.eval() # Set model to evaluation mode

    # --- decide_and_execute ---
    def decide_and_execute(self, task: TaskDAG):
        """
        为给定的任务DAG做出卸载决策，模拟其执行，并返回执行统计信息。

        此方法是算法与环境交互的核心。它首先使用训练好的模仿学习模型
        (`self.model`) 来预测最佳的卸载位置和资源分配。然后，它根据这个决策
        模拟任务的实际执行过程，包括可能的本地计算或到RSU的卸载。
        如果卸载到RSU，它会尝试利用A-LSH机制进行子任务结果的缓存复用。

        主要流程:
        1.  将模型设置为评估模式 (`self.model.eval()`) 并确定设备。
        2.  将输入的 `task.graph` 转换为 PyTorch Geometric `Data` 对象。
        3.  通过模型前向传播获取卸载决策 (`decision_offload`)、CPU分配比例
            (`decision_cpu_alloc`) 和带宽分配比例 (`decision_bw_alloc`)。
        4.  根据 `decision_offload` 判断执行路径：
            a.  **本地执行 (decision_offload == 0)**:
                - 调用 `self.execute_locally(task, vehicle)` 计算本地执行的
                  延迟和能耗。缓存命中数在此路径下为0。
            b.  **卸载到RSU (decision_offload > 0)**:
                - 确定目标RSU。
                - 对模型输出的CPU和带宽分配比例进行裁剪，确保在合理范围内 (0.01 到 1.0)。
                - 计算实际分配给任务的CPU频率和带宽。
                - 计算数据传输到RSU的延迟 (`trans_delay`) 和能耗 (`trans_energy`)。
                - 遍历任务DAG中的每个子任务（按拓扑顺序）：
                    - 获取子任务的特征向量。
                    - 使用A-LSH在目标RSU的缓存中查找：
                        - 对每个LSH表，调用 `utils.compute_lsh_hash` 计算哈希值
                          (此过程内部会调用 `rsu.get_adaptive_bucket_width()` 来获取动态桶宽 `dj`)。
                        - 使用哈希值在RSU缓存 (`rsu.cache_lookup`) 中查找匹配项。
                        - 如果找到缓存项，计算其特征向量与当前子任务特征向量的欧氏距离。
                        - 如果距离小于 `config.REUSE_SIMILARITY_THRESHOLD` (δ)，
                          则认为可以复用。选择距离最小的最佳匹配。
                    - 如果找到可复用的结果 (`cached_result is not None`):
                        - 子任务计算延迟为0，增加 `cache_hit_count`。
                    - 否则 (未找到或不满足复用条件):
                        - 调用 `utils.calculate_rsu_compute_delay` 计算子任务在RSU上的计算延迟。
                        - 如果计算成功，将子任务的模拟结果和特征向量添加到RSU缓存
                          (`rsu.cache_add`)，使用第一个LSH表的哈希值作为键。
                    - 累加子任务的计算延迟到 `rsu_compute_delay_total`。
                - 总延迟 = `trans_delay` + `rsu_compute_delay_total`。
                - 总能耗主要考虑车辆的传输能耗。
                - 更新RSU的当前CPU和带宽负载。
        5.  使用总延迟和总能耗计算最终的系统目标值 (`utils.calculate_objective`)。
        6.  如果过程中发生任何错误（例如，数据转换失败、无效的RSU索引、计算结果为Inf），
            则会捕获异常，并返回表示执行失败的默认统计信息（例如，延迟/能耗为Inf）。

        参数:
            task (TaskDAG): 需要处理和执行的任务DAG对象。

        返回:
            tuple:
                - decision_details (dict): 一个包含模型原始决策的字典
                                           (例如, {'offload': int, 'cpu': float, 'bw': float})。
                                           如果执行失败，可能为空字典。
                - stats (dict): 一个包含任务执行后各项性能指标的字典：
                                {'latency': float, 'energy': float, 'objective': float,
                                 'cache_hit_count': int}。
                                如果执行失败，这些值可能为 `float('inf')` 或0。
        """
        self.model.eval(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); self.model.to(device)
        total_latency = 0; total_energy = 0; cache_hit_count = 0;
        decision_details = {}; stats = {'latency': float('inf'), 'energy': float('inf'), 'objective': float('inf'), 'cache_hit_count': 0}
        try:
            data = dag_to_pyg_data(task.graph, task.subtasks) # Keep on CPU initially
            if data is None or data.x is None or data.edge_index is None:
                raise ValueError("Failed to convert DAG to valid PyG data.")

            with torch.no_grad():
                 data_batch = Batch.from_data_list([data]).to(device)
                 offload_logits, cpu_pred, bw_pred = self.model(data_batch)
                 offload_prob = F.softmax(offload_logits[0], dim=-1)
                 decision_offload = torch.argmax(offload_prob).item()
                 decision_cpu_alloc = cpu_pred[0].item()
                 decision_bw_alloc = bw_pred[0].item()

                 # --- 添加可选的打印进行调试 ---
                 # print(f"DEBUG: Task {task.id} offload decision: {decision_offload}")
                 # ---------------------------

            decision_details = {'offload': decision_offload, 'cpu': decision_cpu_alloc, 'bw': decision_bw_alloc};
            vehicle = self.env.get_vehicle_by_id(task.vehicle_id);
            veh_idx = utils.get_vehicle_index(task.vehicle_id)
            if not vehicle or veh_idx == -1: raise ValueError(f"Vehicle {task.vehicle_id} / Index invalid")

            if decision_offload == 0: # Execute Locally
                _, stats = self.execute_locally(task, vehicle)

            else: # Execute on RSU
                rsu_idx = decision_offload - 1;
                rsu = self.env.get_rsu(rsu_idx);
                if not rsu: raise ValueError(f"Invalid RSU index {rsu_idx} from model")

                # Clamp allocations to prevent extreme values, ensure minimum > 0
                decision_cpu_alloc = np.clip(decision_cpu_alloc, 0.01, 1.0)
                decision_bw_alloc = np.clip(decision_bw_alloc, 0.01, 1.0)

                allocated_cpu = decision_cpu_alloc * rsu.total_cpu_frequency
                allocated_bw = decision_bw_alloc * config.CHANNEL_BANDWIDTH_PER_VEHICLE # Assume fixed channel BW per vehicle

                # Ensure minimum resource allocation
                allocated_cpu = max(1e3, allocated_cpu)
                allocated_bw = max(1e3, allocated_bw)

                # Update RSU load estimates *before* calculation might be more realistic?
                # Or update after? Let's update after calculation for now.

                # Calculate transmission
                noise_power = utils.get_noise_power(allocated_bw);
                gain = self.env.get_channel_gain(veh_idx, rsu_idx)
                rate = utils.calculate_transmission_rate(vehicle.power_transmit, gain, allocated_bw, noise_power);
                trans_delay = utils.calculate_transmission_delay(task.data_size, rate);
                trans_energy = utils.calculate_transmission_energy(vehicle.power_transmit, trans_delay)

                if np.isinf(trans_delay) or np.isinf(trans_energy):
                     raise RuntimeError(f"Transmission failed for task {task.id} to RSU {rsu.id} (Rate: {rate:.2e}, Delay: {trans_delay:.2e})")

                total_latency += trans_delay; total_energy += trans_energy

                # --- RSU Computation (with LSH Reuse) ---
                rsu_compute_delay_total = 0
                # Ensure topological sort is valid before iterating
                subtask_node_list = task.get_topological_nodes()
                if not subtask_node_list:
                    print(f"Warning: Task {task.id} has no valid subtasks in topological order. Skipping RSU computation.")
                else:
                    for subtask_id in subtask_node_list:
                        subtask = task.get_subtask(subtask_id);
                        if not subtask: continue

                        # --- LSH Logic Block ---
                        subtask.is_reused = False; best_match_dist = float('inf'); cached_result = None; feature_vec = subtask.feature_vector; hash_val_for_caching = None
                        # Initialize LSH params if not done already (should be done once per script run)
                        if not utils.LSH_PARAMS: utils.initialize_lsh_params()

                        # --- LSH Cache Lookup ---
                        # 遍历所有LSH哈希表，尝试找到匹配的缓存项
                        for l in range(config.NUM_HASH_TABLES): # config.NUM_HASH_TABLES (L) 是哈希表的数量
                            try:
                                # 为当前子任务的特征向量计算在第l个哈希表中的哈希值
                                # utils.compute_lsh_hash 内部会调用 rsu.get_adaptive_bucket_width() 获取动态桶宽 dj
                                hash_val = utils.compute_lsh_hash(feature_vec, rsu, l)
                                if l == 0: hash_val_for_caching = hash_val # 保存第一个哈希表的哈希值，用于后续可能的缓存添加

                                cached_item = rsu.cache_lookup(hash_val) # 在RSU缓存中查找此哈希值
                                if cached_item is not None: # 如果找到了具有相同哈希值的缓存项
                                    cached_feature = cached_item.get('feature_vector') # 获取缓存项的特征向量
                                    # 确保特征向量类型正确，以计算距离
                                    if isinstance(cached_feature, np.ndarray) and isinstance(feature_vec, np.ndarray):
                                         dist = np.linalg.norm(cached_feature - feature_vec) # 计算当前子任务与缓存项特征向量之间的欧氏距离
                                         # 检查距离是否小于相似性阈值 config.REUSE_SIMILARITY_THRESHOLD (δ)
                                         if dist < config.REUSE_SIMILARITY_THRESHOLD:
                                             # 如果满足阈值，并且这个匹配比之前找到的更好（距离更小），则更新最佳匹配
                                             if dist < best_match_dist:
                                                 best_match_dist = dist
                                                 cached_result = cached_item.get('result') # 获取缓存的结果
                                    else: print(f"Warn: Invalid feature vector types for dist calc in LSH lookup {subtask.id}. Cache:{type(cached_feature)}, Current:{type(feature_vec)}")
                            except Exception as e: print(f"Error during LSH lookup for {subtask.id} (table {l}) on RSU {rsu.id}: {e}")
                        # --- End LSH Lookup ---

                        if cached_result is not None: # 如果找到了可复用的缓存结果
                             subtask.is_reused = True # 标记此子任务为已复用
                             subtask.result = cached_result # 使用缓存的结果
                             cache_hit_count += 1 # 增加缓存命中计数
                             compute_delay = 0 # 复用的子任务计算延迟视为0
                             # print(f"DEBUG: Subtask {subtask.id} reused on RSU {rsu.id}") # Optional debug
                        else: # 如果没有找到可复用的结果，则需要在RSU上计算
                             compute_delay = utils.calculate_rsu_compute_delay(subtask, allocated_cpu, is_reused=False)
                             if np.isinf(compute_delay): # 如果计算延迟为无穷大（例如资源不足）
                                 raise RuntimeError(f"RSU compute failed for subtask {subtask.id} (Alloc CPU: {allocated_cpu:.2e}, Cycles: {subtask.cpu_cycles:.2e})")

                             # 只有当计算成功（延迟非无穷）且有有效的哈希值时，才将结果添加到缓存
                             if not np.isinf(compute_delay) and hash_val_for_caching is not None:
                                 subtask.result = f"Computed_{subtask.id}" # 模拟计算结果
                                 # 将新计算的子任务结果及其特征向量添加到RSU缓存中
                                 rsu.cache_add(hash_val_for_caching, subtask.result, subtask.feature_vector)
                        # --- End LSH Logic Block (Reuse or Compute & Cache) ---

                        rsu_compute_delay_total += compute_delay
                # --- End RSU Subtask Loop ---

                total_latency += rsu_compute_delay_total

                # Update RSU loads *after* processing the task
                # Be careful about over-estimation if tasks run in parallel within a slot
                rsu.current_cpu_load += allocated_cpu # Add the allocated amount
                rsu.current_bw_load += allocated_bw # Add allocated bandwidth

                if np.isinf(total_latency) or np.isinf(total_energy):
                     raise RuntimeError("Execution resulted in Inf Latency/Energy")

                objective = utils.calculate_objective(total_latency, total_energy)
                stats = {'latency': total_latency, 'energy': total_energy, 'objective': objective, 'cache_hit_count': cache_hit_count}

        except Exception as e:
            print(f"Error during decide_and_execute for task {task.id} [{self.name}]: {e}")
            traceback.print_exc()
            # Return default stats indicating failure
            stats = {'latency': float('inf'), 'energy': float('inf'), 'objective': float('inf'), 'cache_hit_count': 0}
            decision_details = {} # No valid decision made

        return decision_details, stats

    # --- execute_locally ---
    def execute_locally(self, task: TaskDAG, vehicle: Vehicle):
        """
        模拟任务DAG在车辆本地执行的过程，并计算相应的延迟和能耗。

        该方法遍历任务DAG中的所有子任务（按拓扑顺序），并为每个子任务
        计算其在车辆本地CPU上执行所需的延迟和消耗的能量。
        这里不涉及任何缓存复用。

        参数:
            task (TaskDAG): 需要在本地执行的任务DAG对象。
            vehicle (Vehicle): 执行该任务的车辆对象，用于获取车辆的CPU频率
                               和计算功率。

        返回:
            tuple:
                - decision_details (dict): 一个表示本地执行决策的字典。
                                           格式为 `{'offload': 0, 'cpu': 1.0, 'bw': 0.0}`，
                                           其中 `offload: 0` 表示本地执行，CPU和BW分配
                                           在此上下文中不直接相关，但为保持一致性而提供。
                - stats (dict): 一个包含本地执行后各项性能指标的字典：
                                {'latency': float, 'energy': float, 'objective': float,
                                 'cache_hit_count': 0}。
                                如果任何子任务的计算导致无限延迟或能耗（例如，车辆CPU频率为0），
                                则总延迟和总能耗将为 `float('inf')`。
                                `cache_hit_count` 对于本地执行始终为0。
        """
        total_latency = 0; total_energy = 0; possible = True
        subtask_node_list = task.get_topological_nodes()
        if not subtask_node_list:
            print(f"Warning: Task {task.id} has no valid subtasks in topological order for local execution.")
            possible = False
        else:
            for subtask_id in subtask_node_list:
                subtask = task.get_subtask(subtask_id);
                if not subtask: continue
                subtask.is_reused = False; # No reuse locally
                delay = utils.calculate_local_compute_delay(subtask, vehicle.cpu_frequency);
                energy = utils.calculate_local_compute_energy(vehicle.power_compute, delay)
                if np.isinf(delay) or np.isinf(energy):
                     print(f"Warning: Local execution impossible for subtask {subtask.id} (Delay: {delay}, Energy: {energy})")
                     possible = False; break
                total_latency += delay; total_energy += energy

        if not possible:
            total_latency = float('inf'); total_energy = float('inf')

        objective = utils.calculate_objective(total_latency, total_energy);
        stats = {'latency': total_latency, 'energy': total_energy, 'objective': objective, 'cache_hit_count': 0};
        decision_details = {'offload': 0, 'cpu': 1.0, 'bw': 0.0} # CPU/BW not relevant for local
        return decision_details, stats


# --- GAT-IL (no reuse) Implementation ---
class GAT_IL_NoReuse(GAT_REUSE_IL): # Inherits from corrected GAT_REUSE_IL
    """
    基于图注意力网络（GAT）的模仿学习（IL）卸载算法，但不使用子任务结果复用机制。

    此类继承自 `GAT_REUSE_IL`，并覆盖了 `decide_and_execute` 方法以移除
    RSU缓存查找和子任务复用的逻辑。
    当任务被卸载到RSU时，所有的子任务都会在RSU上重新计算，即使之前
    计算过相似的子任务。

    主要特点和目的：
    - **行为与 `GAT_REUSE_IL` 相似**：它同样使用 `ILDecisionModel` 进行卸载决策和
      资源分配预测，并且训练过程也依赖于专家数据。
    - **无复用机制**：核心区别在于其 `decide_and_execute` 方法在处理RSU执行时，
      不进行任何缓存查找或结果复用。因此，其 `cache_hit_count` 始终为0。
    - **作为基准**：该算法主要用作一个对比基准，以评估 `GAT_REUSE_IL` 中
      子任务复用机制所带来的性能增益（例如，在延迟降低或系统成本优化方面）。

    通过比较 `GAT_IL_NoReuse` 和 `GAT_REUSE_IL` 的仿真结果，可以量化
    复用策略对整体系统性能的影响。
    """
    def __init__(self, env: VECEnvironment):
        super().__init__(env) # Calls corrected parent __init__
        self.name = "GAT-IL (no reuse)"
        # Training method is inherited

    # --- Override decide_and_execute to disable reuse ---
    def decide_and_execute(self, task: TaskDAG):
        """
        为给定的任务DAG做出卸载决策并模拟执行，但不使用任何子任务结果复用机制。

        此方法覆盖了父类 `GAT_REUSE_IL` 中的同名方法。其核心逻辑与父类相似：
        使用模仿学习模型 (`self.model`) 进行决策，然后模拟本地执行或RSU卸载。
        **主要区别**在于，当任务被卸载到RSU时，此方法 **不进行任何缓存查找或复用**。
        所有子任务都会在RSU上被计算，即使之前处理过相似的子任务。

        因此，`cache_hit_count` 在此方法的返回结果中始终为0。

        参数:
            task (TaskDAG): 需要处理和执行的任务DAG对象。

        返回:
            tuple:
                - decision_details (dict): 包含模型原始决策的字典
                                           (例如, {'offload': int, 'cpu': float, 'bw': float})。
                - stats (dict): 包含任务执行后各项性能指标的字典：
                                {'latency': float, 'energy': float, 'objective': float,
                                 'cache_hit_count': 0}。
                                `cache_hit_count` 固定为0。
        """
        self.model.eval(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); self.model.to(device);
        total_latency = 0; total_energy = 0
        decision_details = {}; stats = {'latency': float('inf'), 'energy': float('inf'), 'objective': float('inf'), 'cache_hit_count': 0}
        try:
            data = dag_to_pyg_data(task.graph, task.subtasks)
            if data is None or data.x is None or data.edge_index is None:
                 raise ValueError("Failed to convert DAG to valid PyG data.")

            with torch.no_grad():
                 data_batch = Batch.from_data_list([data]).to(device)
                 offload_logits, cpu_pred, bw_pred = self.model(data_batch)
                 offload_prob = F.softmax(offload_logits[0], dim=-1)
                 decision_offload = torch.argmax(offload_prob).item()
                 decision_cpu_alloc = cpu_pred[0].item()
                 decision_bw_alloc = bw_pred[0].item()

            decision_details = {'offload': decision_offload, 'cpu': decision_cpu_alloc, 'bw': decision_bw_alloc};
            vehicle = self.env.get_vehicle_by_id(task.vehicle_id);
            veh_idx = utils.get_vehicle_index(task.vehicle_id)
            if not vehicle or veh_idx == -1: raise ValueError(f"Vehicle {task.vehicle_id} / Index invalid")

            if decision_offload == 0: # Execute Locally
                _, stats = self.execute_locally(task, vehicle) # Uses inherited execute_locally

            else: # Execute on RSU (No Reuse Logic)
                rsu_idx = decision_offload - 1;
                rsu = self.env.get_rsu(rsu_idx);
                if not rsu: raise ValueError(f"Invalid RSU index {rsu_idx} from model")

                # Clamp allocations
                decision_cpu_alloc = np.clip(decision_cpu_alloc, 0.01, 1.0)
                decision_bw_alloc = np.clip(decision_bw_alloc, 0.01, 1.0)
                allocated_cpu = max(1e3, decision_cpu_alloc * rsu.total_cpu_frequency)
                allocated_bw = max(1e3, decision_bw_alloc * config.CHANNEL_BANDWIDTH_PER_VEHICLE)

                # Transmission
                noise_power = utils.get_noise_power(allocated_bw);
                gain = self.env.get_channel_gain(veh_idx, rsu_idx)
                rate = utils.calculate_transmission_rate(vehicle.power_transmit, gain, allocated_bw, noise_power);
                trans_delay = utils.calculate_transmission_delay(task.data_size, rate);
                trans_energy = utils.calculate_transmission_energy(vehicle.power_transmit, trans_delay)

                if np.isinf(trans_delay) or np.isinf(trans_energy):
                     raise RuntimeError(f"Transmission failed for task {task.id} to RSU {rsu.id}")

                total_latency += trans_delay; total_energy += trans_energy

                # RSU Computation (Always is_reused=False)
                rsu_compute_delay_total = 0
                subtask_node_list = task.get_topological_nodes()
                if not subtask_node_list:
                     print(f"Warning: Task {task.id} has no valid subtasks (NoReuse). Skipping RSU computation.")
                else:
                     for subtask_id in subtask_node_list:
                         subtask = task.get_subtask(subtask_id);
                         if not subtask: continue
                         # --- Always compute, no reuse check ---
                         compute_delay = utils.calculate_rsu_compute_delay(subtask, allocated_cpu, is_reused=False)
                         # -------------------------------------
                         if np.isinf(compute_delay):
                             raise RuntimeError(f"RSU compute failed {subtask.id} (NoReuse)")
                         rsu_compute_delay_total += compute_delay

                total_latency += rsu_compute_delay_total

                # Update RSU loads
                rsu.current_cpu_load += allocated_cpu
                rsu.current_bw_load += allocated_bw

                objective = utils.calculate_objective(total_latency, total_energy);
                stats = {'latency': total_latency, 'energy': total_energy, 'objective': objective, 'cache_hit_count': 0} # Cache hits always 0

        except Exception as e:
            print(f"Error during decide_and_execute for task {task.id} [{self.name}]: {e}")
            traceback.print_exc()
            stats = {'latency': float('inf'), 'energy': float('inf'), 'objective': float('inf'), 'cache_hit_count': 0}
            decision_details = {}

        return decision_details, stats


# --- GAT-REUSE-DRL Implementation ---
class GAT_REUSE_DRL(BaseAlgorithm):
    """
    基于图注意力网络（GAT）的深度强化学习（DRL）卸载算法，集成了子任务结果复用机制。

    该算法采用Actor-Critic架构，通过与VEC环境的交互来学习最优的卸载策略，
    目标是最小化系统成本（通常是延迟和能耗的加权和，表现为最大化奖励）。

    核心组件：
    - `actor (ActorNetwork)`: 策略网络，基于当前状态（任务DAG的GAT嵌入）输出动作概率
      （针对卸载目标）和连续动作的参数（针对CPU和带宽分配）。
    - `critic (CriticNetwork)`:价值网络，评估当前状态的价值，用于指导Actor网络的更新。
    - `actor_optimizer`, `critic_optimizer`: 分别用于更新Actor和Critic网络的优化器。
    - `replay_buffer (ReplayBuffer)`: 存储经验元组 `(state, action, reward, next_state, done)`，
      用于稳定和提高学习效率。
    - `select_action(state_data)`: 根据Actor网络的输出，为给定的状态（任务DAG）选择一个动作
      （包括离散的卸载目标和连续的资源分配比例）。动作选择包含探索性。
    - `decide_and_execute(task)`:
        1. 使用 `select_action` 为当前任务获取决策。
        2. 模拟任务执行，与 `GAT_REUSE_IL` 类似，如果卸载到RSU，则会尝试使用A-LSH机制
           进行子任务结果的缓存查找和复用。
        3. 计算执行后的性能（延迟、能耗、目标值）并据此计算奖励信号（通常是负的目标值）。
        4. 将 `(state, action_details, reward, next_state=None, done=True)` 经验存入回放缓冲区。
           这里假设每个任务的完成是一个独立的episode结束点。
    - `update_network()`: 从回放缓冲区中采样一批经验，计算损失并更新Actor和Critic网络的参数。
      Critic通过最小化TD误差来更新，Actor通过策略梯度方法（利用Critic计算的优势值）来更新。
    - `run_slot(tasks)`: 在每个仿真时隙处理任务后，会调用 `update_network` 多次
      （由 `config.EPOCHS_DRL_PER_SLOT_UPDATE` 控制）来训练网络。
    - `execute_locally`: 继承自 `GAT_REUSE_IL`，处理本地执行逻辑。

    与模仿学习算法不同，DRL算法不需要专家数据，而是通过自身的探索和从环境反馈中学习。
    它也集成了与 `GAT_REUSE_IL` 相同的子任务复用逻辑，旨在通过智能决策和高效复用
    来优化长期累积奖励。
    """
    def __init__(self, env: VECEnvironment):
        super().__init__(env, name="GAT-REUSE-DRL")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{self.name}] Using device: {self.device}")
        self.actor = ActorNetwork().to(self.device)
        self.critic = CriticNetwork().to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.LEARNING_RATE_DRL_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.LEARNING_RATE_DRL_CRITIC)
        self.replay_buffer = ReplayBuffer()
        utils.initialize_lsh_params() # Ensure LSH params are ready

    def select_action(self, state_data):
        """
        使用 Actor 网络为给定的状态（任务DAG）选择一个动作。

        此方法将 Actor 模型 (`self.actor`) 设置为评估模式，并根据其输出的
        概率分布和参数来采样或确定动作。

        流程:
        1.  将 Actor 网络设置为评估模式 (`self.actor.eval()`)。
        2.  在不计算梯度的上下文 (`torch.no_grad()`) 中执行：
            a.  将输入的 `state_data` (单个PyG `Data` 对象或已批处理的 `Batch` 对象)
                转换为批处理形式并移至指定设备。
            b.  通过 Actor 网络前向传播获取 `offload_prob` (卸载决策的概率分布),
                `cpu_mean` (CPU分配比例的均值), 和 `bw_mean` (带宽分配比例的均值)。
            c.  对于离散的卸载动作：
                -   创建一个 `Categorical` 分布对象。
                -   从该分布中采样一个卸载动作 (`offload_action`)。
            d.  对于连续的CPU和带宽分配动作：
                -   直接使用 Actor 输出的均值 `cpu_mean` 和 `bw_mean` 作为动作值。
                -   （可选地，可以从以这些均值为中心的分布中采样，但当前实现是确定性的）。
                -   将这些值裁剪到有效范围内 (例如, 0.01 到 1.0)。
        3.  将选择的动作（卸载目标、CPU分配、带宽分配）组合成一个字典 `action`。
        4.  创建一个 `action_details_for_training` 字典，存储用于后续训练的动作张量
            （例如，采样到的离散动作的张量、连续动作值的张量，以及可能的log_prob）。
            这些张量会被 `detach()` 并移至CPU，以便存入回放缓冲区。

        参数:
            state_data (torch_geometric.data.Data or torch_geometric.data.Batch):
                表示当前状态的PyG图数据对象或已批处理的图数据。

        返回:
            tuple:
                - action (dict): 一个包含最终选定动作的字典，格式为
                                 `{'offload': int, 'cpu': float, 'bw': float}`。
                                 这些是实际执行的动作值。
                - action_details_for_training (dict): 一个包含用于训练的动作相关张量的字典。
                                                      例如 `{'offload_action': tensor,
                                                      'cpu_action': tensor, 'bw_action': tensor}`。
        """
        self.actor.eval()
        with torch.no_grad():
            # Ensure input is a batch for the model
            if isinstance(state_data, Data):
                 state_batch = Batch.from_data_list([state_data]).to(self.device)
            else: # Assume already a batch if not a single Data object
                 state_batch = state_data.to(self.device)

            offload_prob, cpu_mean, bw_mean = self.actor(state_batch)

            # Process output for the single graph
            offload_prob_single = offload_prob[0]
            cpu_mean_single = cpu_mean[0]
            bw_mean_single = bw_mean[0]

            # Sample discrete action
            offload_dist = Categorical(offload_prob_single)
            offload_action = offload_dist.sample()

            # Use mean directly for continuous actions (after sigmoid in actor)
            # Clamp to ensure actions are within a valid range (e.g., 0.01 to 1.0)
            cpu_action = torch.clamp(cpu_mean_single, 0.01, 1.0)
            bw_action = torch.clamp(bw_mean_single, 0.01, 1.0)

        # Return actions as scalars/items
        action = {'offload': offload_action.item(), 'cpu': cpu_action.item(), 'bw': bw_action.item()}

        # Store tensors for training (on CPU, detached)
        action_details_for_training = {
            'offload_action': offload_action.cpu().detach(),
            'cpu_action': cpu_action.cpu().detach(),
            'bw_action': bw_action.cpu().detach()
            # Optional: Store log_prob if needed for specific PG updates
            # 'log_prob_offload': offload_dist.log_prob(offload_action).cpu().detach()
        }
        return action, action_details_for_training

    def decide_and_execute(self, task: TaskDAG):
        """
        为给定的任务DAG使用DRL策略选择动作，模拟其执行，计算奖励，并存储经验。

        此方法是DRL算法与环境交互的核心循环的一部分。它与 `GAT_REUSE_IL` 的
        `decide_and_execute` 在执行模拟（包括A-LSH缓存复用）方面相似，但其决策来源
        是 `self.select_action`（即Actor网络），并且它额外计算奖励并将经验
        `(state, action, reward, next_state, done)` 存入 `self.replay_buffer`。

        主要流程:
        1.  将Actor和Critic网络设置为评估模式。
        2.  将输入的 `task.graph` 转换为 PyTorch Geometric `Data` 对象 (`state_data_pyg`)。
        3.  调用 `self.select_action(state_data_pyg)` 从Actor网络获取卸载决策
            (`action_dict`) 和用于训练的动作细节 (`action_details_train`)。
        4.  根据 `action_dict` 中的决策模拟任务执行（本地或RSU卸载）：
            -   本地执行逻辑与 `GAT_REUSE_IL.execute_locally` 相同。
            -   RSU卸载逻辑与 `GAT_REUSE_IL.decide_and_execute` 中的RSU部分相似，
                包括数据传输计算、子任务的A-LSH缓存查找和复用（如果命中）、
                或在RSU上计算（如果未命中或不复用），以及更新RSU负载。
        5.  计算执行后的总延迟 (`total_latency`) 和总能耗 (`total_energy`)。
        6.  计算系统目标值 (`objective = utils.calculate_objective(...)`)。
        7.  计算奖励 (`reward`)。通常奖励是负的目标值（因为目标是最小化成本），
            可能会乘以一个缩放因子 (`config.REWARD_SCALE`)。可以添加对违反约束的惩罚。
        8.  将最终的性能统计（包括奖励）存储在 `stats` 字典中。
        9.  如果状态、动作细节有效，则将经验元组 `(state_data_cpu, action_details_train,
            reward, next_state_data=None, done=True)` 添加到 `self.replay_buffer`。
            这里假设每个任务的完成代表一个episode的结束 (`done=True`)，且没有显式的下一个状态。
        10. 如果过程中发生错误，则捕获异常，返回表示失败的默认统计信息，并可能赋一个大的负奖励。

        参数:
            task (TaskDAG): 需要处理和执行的任务DAG对象。

        返回:
            tuple:
                - decision_details (dict): 一个包含Actor网络所选动作的字典
                                           (例如, {'offload': int, 'cpu': float, 'bw': float})。
                - stats (dict): 一个包含任务执行后各项性能指标（包括奖励）的字典：
                                {'latency': float, 'energy': float, 'objective': float,
                                 'cache_hit_count': int, 'reward': float}。
        """
        # Use the same logic as GAT_REUSE_IL for execution, but get decisions from self.select_action
        self.actor.eval(); self.critic.eval(); # Ensure models are in eval mode
        decision_details = {}; stats = {'latency': float('inf'), 'energy': float('inf'), 'objective': float('inf'), 'cache_hit_count': 0, 'reward': -np.inf}
        state_data_cpu = None # Keep state on CPU for buffer
        action_details_train = None # To store action tensors

        try:
            state_data_pyg = dag_to_pyg_data(task.graph, task.subtasks) # Keep on CPU initially
            if state_data_pyg is None or state_data_pyg.x is None or state_data_pyg.edge_index is None:
                raise ValueError("Failed to convert DAG to valid PyG data for DRL state.")
            state_data_cpu = state_data_pyg # Reference for buffer

            # --- Get action from DRL policy ---
            action_dict, action_details_train = self.select_action(state_data_cpu)
            # --------------------------------

            decision_details = action_dict;
            decision_offload = action_dict['offload'];
            decision_cpu_alloc = action_dict['cpu']; # Already clamped in select_action
            decision_bw_alloc = action_dict['bw'];  # Already clamped in select_action

            total_latency = 0; total_energy = 0; cache_hit_count = 0;
            vehicle = self.env.get_vehicle_by_id(task.vehicle_id);
            veh_idx = utils.get_vehicle_index(task.vehicle_id)
            if not vehicle or veh_idx == -1: raise ValueError(f"Vehicle {task.vehicle_id} / Index invalid")

            if decision_offload == 0: # Execute Locally
                _, exec_stats = self.execute_locally(task, vehicle) # Use inherited execute_locally
                total_latency=exec_stats['latency']
                total_energy=exec_stats['energy']
                cache_hit_count = 0 # Ensure cache hits are 0 for local

            else: # Execute on RSU
                rsu_idx = decision_offload - 1
                if not (0 <= rsu_idx < len(self.env.rsus)):
                    raise ValueError(f"Invalid RSU index {rsu_idx} chosen by DRL actor.")
                rsu = self.env.get_rsu(rsu_idx);
                if not rsu: raise ValueError(f"RSU object not found for index {rsu_idx}")

                # Use clamped allocations directly
                allocated_cpu = max(1e3, decision_cpu_alloc * rsu.total_cpu_frequency)
                allocated_bw = max(1e3, decision_bw_alloc * config.CHANNEL_BANDWIDTH_PER_VEHICLE)

                # Calculate transmission
                noise_power = utils.get_noise_power(allocated_bw);
                gain = self.env.get_channel_gain(veh_idx, rsu_idx)
                rate = utils.calculate_transmission_rate(vehicle.power_transmit, gain, allocated_bw, noise_power);
                trans_delay = utils.calculate_transmission_delay(task.data_size, rate);
                trans_energy = utils.calculate_transmission_energy(vehicle.power_transmit, trans_delay)

                if np.isinf(trans_delay) or np.isinf(trans_energy):
                     raise RuntimeError(f"Transmission failed for task {task.id} to RSU {rsu.id} (DRL)")

                total_latency += trans_delay; total_energy += trans_energy

                # --- RSU Computation (with LSH Reuse) - Same logic as GAT_REUSE_IL ---
                rsu_compute_delay_total = 0
                subtask_node_list = task.get_topological_nodes()
                if not subtask_node_list:
                    print(f"Warning: Task {task.id} has no valid subtasks (DRL). Skipping RSU computation.")
                else:
                     # Initialize LSH params if not done already
                    if not utils.LSH_PARAMS: utils.initialize_lsh_params()
                    for subtask_id in subtask_node_list:
                        subtask = task.get_subtask(subtask_id);
                        if not subtask: continue
                        subtask.is_reused = False; best_match_dist = float('inf'); cached_result = None; feature_vec = subtask.feature_vector; hash_val_for_caching = None
                        for l in range(config.NUM_HASH_TABLES):
                            try:
                                hash_val = utils.compute_lsh_hash(feature_vec, rsu, l);
                                if l == 0: hash_val_for_caching = hash_val
                                cached_item = rsu.cache_lookup(hash_val)
                                if cached_item is not None:
                                    cached_feature = cached_item.get('feature_vector')
                                    if isinstance(cached_feature, np.ndarray) and isinstance(feature_vec, np.ndarray):
                                         dist = np.linalg.norm(cached_feature - feature_vec)
                                         if dist < config.REUSE_SIMILARITY_THRESHOLD:
                                             if dist < best_match_dist: best_match_dist = dist; cached_result = cached_item.get('result')
                                    else: print(f"Warn: Invalid types {subtask.id}. Cache:{type(cached_feature)}, Cur:{type(feature_vec)}")
                            except Exception as e: print(f"Error LSH {subtask.id} (table {l}): {e}")
                        if cached_result is not None:
                             subtask.is_reused = True; subtask.result = cached_result; cache_hit_count += 1; compute_delay = 0
                        else:
                             compute_delay = utils.calculate_rsu_compute_delay(subtask, allocated_cpu, is_reused=False);
                             if np.isinf(compute_delay): raise RuntimeError(f"RSU compute failed {subtask.id} (DRL)")
                             if not np.isinf(compute_delay) and hash_val_for_caching is not None:
                                  subtask.result = f"Computed_{subtask.id}"; rsu.cache_add(hash_val_for_caching, subtask.result, subtask.feature_vector)

                        rsu_compute_delay_total += compute_delay
                # --- End reuse logic block ---
                total_latency += rsu_compute_delay_total

                # Update RSU loads
                rsu.current_cpu_load += allocated_cpu
                rsu.current_bw_load += allocated_bw

            # --- Calculate final stats and reward ---
            if np.isinf(total_latency) or np.isinf(total_energy):
                # This case should ideally be caught earlier, but double-check
                print(f"Warning: Task {task.id} resulted in infinite latency/energy after execution.")
                objective = float('inf')
                reward = -1e9 # Assign large negative reward
            else:
                objective = utils.calculate_objective(total_latency, total_energy)
                # Reward is negative objective (minimize objective = maximize reward)
                # Add penalty if constraints are violated (optional)
                reward = -objective * config.REWARD_SCALE
                # Example constraint check:
                # if total_latency > config.MAX_TOLERABLE_LATENCY: reward -= 100 # Add penalty
                # if total_energy > config.MAX_VEHICLE_ENERGY: reward -= 100 # Add penalty

            # Ensure reward is finite before adding to buffer
            if np.isinf(reward): reward = -1e9 # Fallback large negative reward

            stats = {'latency': total_latency, 'energy': total_energy, 'objective': objective, 'cache_hit_count': cache_hit_count, 'reward': reward}

            # --- Store experience in buffer ---
            if state_data_cpu is not None and action_details_train is not None:
                 # Assuming each task completion is a terminal step for this state
                 done = True
                 self.replay_buffer.add(state_data_cpu, action_details_train, reward, None, done)
            # --------------------------------

        except Exception as e:
            print(f"Error during decide_and_execute for task {task.id} [{self.name}]: {e}")
            traceback.print_exc()
            # Return default stats indicating failure, assign large negative reward
            stats = {'latency': float('inf'), 'energy': float('inf'), 'objective': float('inf'), 'cache_hit_count': 0, 'reward': -1e9}
            decision_details = {}
            # Do not add this experience to the buffer if execution failed critically
        return decision_details, stats

    def update_network(self):
        """
        使用从经验回放缓冲区中采样的经验来更新Actor和Critic网络。

        此方法实现了DRL算法（特别是Actor-Critic类型）的核心学习步骤。
        如果缓冲区中的经验数量不足以构成一个批次，则不执行更新。

        主要流程:
        1.  检查回放缓冲区大小，如果小于 `config.DRL_BATCH_SIZE`，则返回0损失。
        2.  将Actor和Critic网络设置为训练模式 (`self.actor.train()`, `self.critic.train()`)。
        3.  从 `self.replay_buffer` 中采样一个批次的经验
            (`state_batch`, `actions_batch`, `rewards_tensor`, `dones_tensor`)。
            如果采样失败，则跳过更新。

        4.  **Critic 网络更新**:
            a.  清零Critic优化器的梯度 (`self.critic_optimizer.zero_grad()`)。
            b.  通过Critic网络前向传播 `state_batch`，得到当前状态的价值估计 `current_values`。
            c.  计算目标价值 `target_values`。在此实现中，由于假设每个任务完成即episode结束
                (`done=True`)且没有明确的下一个状态价值，目标价值直接设为采样到的奖励 `rewards_tensor`。
                （在更一般的AC算法中，`target_values = rewards + gamma * V(next_state) * (1-done)`）。
            d.  计算Critic损失（均方误差损失）：`critic_loss = F.mse_loss(current_values, target_values)`。
            e.  反向传播Critic损失 (`critic_loss.backward()`)。
            f.  （可选）裁剪Critic网络的梯度范数。
            g.  执行Critic优化器步骤 (`self.critic_optimizer.step()`)。

        5.  **Actor 网络更新**:
            a.  清零Actor优化器的梯度 (`self.actor_optimizer.zero_grad()`)。
            b.  计算优势估计 `advantages`。在这里，它被计算为 `(target_values - current_values).detach()`。
                `.detach()` 用于防止梯度从Actor的损失流向Critic。
            c.  让Actor网络重新评估采样批次中的状态 `state_batch`，得到当前策略下的
                `offload_prob_current`, `cpu_mean_current`, `bw_mean_current`。
            d.  计算在 `actions_batch` 中实际采取的离散卸载动作的对数概率 `log_probs_offload`。
            e.  计算Actor损失（策略梯度损失）：
                `actor_loss_offload = -(log_probs_offload * advantages.squeeze()).mean()`。
                目标是最大化 `log_prob * advantage`，因此损失是其负值。
            f.  （可选）可以为连续动作（CPU/BW分配）添加额外的损失项，例如MSE损失，
                但当前实现主要集中在离散卸载动作的策略梯度上。
            g.  反向传播Actor损失 (`actor_loss.backward()`)。
            h.  （可选）裁剪Actor网络的梯度范数。
            i.  执行Actor优化器步骤 (`self.actor_optimizer.step()`)。

        返回:
            tuple:
                - avg_actor_loss (float): 计算得到的Actor网络的平均损失。
                - avg_critic_loss (float): 计算得到的Critic网络的平均损失。
                                          如果更新未执行（例如，缓冲区数据不足），则返回 (0.0, 0.0)。
        """
        if len(self.replay_buffer) < config.DRL_BATCH_SIZE: return 0.0, 0.0 # Return separate losses
        self.actor.train(); self.critic.train()
        sampled_data = self.replay_buffer.sample(config.DRL_BATCH_SIZE, self.device)
        if sampled_data[0] is None: print("Skipping DRL update: Sampling failed."); return 0.0, 0.0
        state_batch, actions_batch, rewards_tensor, _, dones_tensor = sampled_data # actions_batch is dict of tensors

        # --- Critic Update ---
        self.critic_optimizer.zero_grad() # 清零Critic优化器的梯度
        current_values = self.critic(state_batch) # Critic网络评估当前状态批次的价值 V(s_t)
        # 计算目标价值。由于假设每个任务完成即episode结束 (done=True)，
        # 且next_state为None，所以target_values直接是rewards_tensor (即 R_t)。
        # 对于非终止状态，通常是 R_t + gamma * V(s_{t+1})。
        target_values = rewards_tensor # 形状 [batch, 1]

        if current_values.shape != target_values.shape:
             print(f"Error: Critic shape mismatch. Current: {current_values.shape}, Target: {target_values.shape}")
             return 0.0, 0.0 # 如果形状不匹配，则跳过更新
        # Critic损失：当前价值估计与目标价值之间的均方误差
        critic_loss = F.mse_loss(current_values, target_values)
        critic_loss.backward() # 反向传播Critic损失
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0) # 裁剪Critic网络的梯度，防止梯度爆炸
        self.critic_optimizer.step() # 执行Critic优化器步骤，更新Critic网络参数
        avg_critic_loss = critic_loss.item() # 记录Critic损失值

        # --- Actor Update (Policy Gradient based on Advantage) ---
        self.actor_optimizer.zero_grad() # 清零Actor优化器的梯度
        # 计算优势值 A(s_t, a_t) = Q(s_t, a_t) - V(s_t)
        # 在这里，Q(s_t, a_t) 的估计用 target_values (即 R_t，因为是终止状态) 代替。
        # V(s_t) 是 current_values。
        # .detach()确保在计算优势时，梯度不会从actor流向critic。
        advantages = (target_values - current_values).detach() # 形状 [batch, 1]

        # Actor网络重新评估采样批次中的状态，以获取当前策略下的动作概率和参数
        offload_prob_current, cpu_mean_current, bw_mean_current = self.actor(state_batch)

        # --- Offload Action Log Prob (离散卸载动作的对数概率) ---
        # 基于Actor输出的概率分布，为实际采取的离散卸载动作计算对数概率 log(π(a_offload|s))
        offload_dist_current = Categorical(offload_prob_current) # 创建离散动作的分类分布
        action_indices_offload = actions_batch['offload'] # 从经验中获取实际采取的卸载动作索引
        if action_indices_offload.ndim > 1: action_indices_offload = action_indices_offload.squeeze(-1) # 确保形状为 [batch]
        try:
            log_probs_offload = offload_dist_current.log_prob(action_indices_offload) # 计算对数概率，形状 [batch]
        except ValueError as e: # 处理潜在的由于概率或索引问题导致的错误
             print(f"Error calculating log_prob for offload actions: {e}")
             print(f"  Probabilities shape: {offload_prob_current.shape}")
             print(f"  Action indices shape: {action_indices_offload.shape}")
             print(f"  Action indices values: {action_indices_offload}")
             return avg_critic_loss, 0.0 # 跳过Actor更新

        # Actor损失：最大化 E[Adv * log(π(a|s))]，等价于最小化 -E[Adv * log(π(a|s))]
        # advantages.squeeze() 将 [batch, 1] 的优势值张量变为 [batch] 以匹配 log_probs_offload 的形状
        actor_loss_offload = -(log_probs_offload * advantages.squeeze()).mean() # 计算批次平均损失

        # --- 可选: 为连续动作添加损失 (当前简化为仅使用离散动作损失) ---
        # actor_loss_cpu = F.mse_loss(cpu_mean_current, actions_batch['cpu']) # 例如，与实际采取的连续动作的MSE
        # actor_loss_bw = F.mse_loss(bw_mean_current, actions_batch['bw'])
        # actor_loss = actor_loss_offload + 0.1 * actor_loss_cpu + 0.1 * actor_loss_bw # 组合损失
        actor_loss = actor_loss_offload # 当前实现：仅使用离散卸载动作的策略梯度损失

        actor_loss.backward() # 反向传播Actor损失
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0) # 裁剪Actor网络的梯度
        self.actor_optimizer.step() # 执行Actor优化器步骤，更新Actor网络参数
        avg_actor_loss = actor_loss.item()

        return avg_actor_loss, avg_critic_loss


    def run_slot(self, tasks):
        self.actor.eval(); self.critic.eval() # Set to eval for decision making
        super().run_slot(tasks) # Call base class run_slot which does the decision making loop & dj collection
        self.actor.train(); self.critic.train() # Set back to train for updates

        total_actor_loss = 0; total_critic_loss = 0; update_count = 0
        updates_this_slot = 0
        # Perform multiple updates per slot if desired and buffer has enough samples
        while updates_this_slot < config.EPOCHS_DRL_PER_SLOT_UPDATE and len(self.replay_buffer) >= config.DRL_BATCH_SIZE:
             actor_loss, critic_loss = self.update_network();
             if actor_loss is not None and critic_loss is not None: # Check if update was successful
                 total_actor_loss += actor_loss
                 total_critic_loss += critic_loss
                 update_count += 1
             updates_this_slot += 1

        avg_actor_loss = total_actor_loss / update_count if update_count > 0 else 0
        avg_critic_loss = total_critic_loss / update_count if update_count > 0 else 0
        # Optionally print update losses periodically
        if update_count > 0 and (self.env.current_slot % 20 == 1): # Print less frequently
            print(f"  [{self.name} Slot {self.env.current_slot} Updates] Avg Actor Loss: {avg_actor_loss:.4f}, Avg Critic Loss: {avg_critic_loss:.4f} ({update_count} updates)")


    # Inherit execute_locally from GAT_REUSE_IL (which inherits from Base or defines its own)
    # Make sure GAT_REUSE_IL has the correct execute_locally defined
    execute_locally = GAT_REUSE_IL.execute_locally