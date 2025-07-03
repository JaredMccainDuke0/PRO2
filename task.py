"""
此模块定义了计算任务及其子任务的数据结构。

在车载边缘计算（VEC）的上下文中，一个复杂的计算任务通常被建模为
一个有向无环图（DAG），其中节点代表子任务，边代表子任务之间的依赖关系和数据传输。

主要包含以下类：

1.  `SubTask`:
    表示DAG中的一个单独的子任务。每个子任务具有以下属性：
    - `id`: 子任务的唯一标识符。
    - `task_id`: 其所属父任务DAG的标识符。
    - `cpu_cycles`: 执行此子任务所需的CPU周期数。
    - `feature_vector`: 一个数值向量，描述子任务的特性（例如，标准化的CPU周期、
      输入/输出数据大小、任务类型等）。这个特征向量用于机器学习模型的输入，
      以及A-LSH（Adaptive Locality Sensitive Hashing）中的相似性比较。
    - `result`: 子任务的计算结果（在仿真中通常是占位符）。
    - `hash_value`: 子任务通过A-LSH计算得到的哈希值，用于缓存查找。
    - `is_reused`:一个布尔标志，指示此子任务的结果是否从缓存中复用。

2.  `TaskDAG`:
    表示一个完整的计算任务，由多个相互依赖的子任务组成。
    其核心属性和功能包括：
    - `id`: 任务DAG的唯一标识符。
    - `vehicle_id`: 发起此任务的车辆的ID。
    - `graph`: 一个 `networkx.DiGraph` 对象，存储了子任务（节点）及其依赖关系（边）。
      图中包含虚拟的入口和出口节点。
    - `subtasks`:一个字典，存储了所有 `SubTask` 对象，通过子任务ID索引。
    - `data_size`: 整个任务的总数据大小，主要用于估算卸载时的初始传输成本。
    - `_create_dag()`:一个内部方法，在任务实例化时随机生成DAG的结构，包括创建子任务节点
      和它们之间的边。边的权重可以表示数据传输量。
    - `get_topological_nodes()`: 返回DAG中子任务节点的拓扑排序列表（不包括虚拟入口/出口节点），
      这对于按顺序处理子任务非常重要。
    - `get_subtask(subtask_id)`: 根据ID检索特定的 `SubTask` 对象。

这些数据结构是整个仿真系统的基础，因为它们定义了需要处理和决策的工作负载。
它们与 `config.py` 交互以获取任务生成的参数（如子任务数量范围、CPU周期范围等），
并被 `algorithms.py` 中的卸载算法和 `gat_model.py` 中的GAT模型使用。
"""
# task.py
import numpy as np
import networkx as nx
import config
import traceback # For detailed error printing

class SubTask:
    """
    表示计算任务有向无环图（DAG）中的一个单独子任务。

    每个子任务是构成一个完整 `TaskDAG` 的基本计算单元。它具有特定的计算需求
    和一组特征，这些特征用于机器学习模型的输入以及在RSU缓存中进行相似性比较。

    属性:
        id (str): 子任务的唯一标识符，通常格式为 "task_k_sub_s"，
                  其中 k 是父任务的索引，s 是子任务的索引。
        task_id (str): 该子任务所属的父 `TaskDAG` 的ID。
        cpu_cycles (float): 执行此子任务所需的CPU周期数 (Ck,s)。
                            这是衡量子任务计算量的主要指标。
        feature_vector (np.ndarray): 一个固定长度 (由 `config.FEATURE_DIM` 定义)
                                     的NumPy数组，表示子任务的数值特征。
                                     这些特征可能包括标准化的CPU周期、标准化的输入/输出数据大小、
                                     以及表示任务类型的嵌入（例如one-hot编码）等。
                                     此向量用于GAT模型的输入和A-LSH哈希计算。
        result (any): 子任务的计算结果。在当前仿真中，这通常是一个占位符字符串，
                      例如 "Computed_subtask_id"，用于指示任务已计算。
        hash_value (tuple): 通过A-LSH（Adaptive Locality Sensitive Hashing）算法
                            为该子任务的 `feature_vector` 计算得到的哈希元组。
                            此哈希值用于在RSU的缓存中快速查找可能匹配的已计算子任务。
        is_reused (bool): 一个布尔标志，指示此子任务的结果是否是从RSU缓存中复用的。
                          如果为 `True`，则表示找到了一个足够相似的已缓存结果，
                          避免了重新计算。默认为 `False`。

    在初始化 `SubTask` 对象时，会根据其原始属性（如 `cpu_cycles`, `data_in`, `data_out`,
    `task_type_embedding`）构建 `feature_vector`。这个构建过程包括对原始值进行归一化
    （例如，除以配置中的最大值），然后将它们与任务类型嵌入连接起来，并确保最终的
    特征向量长度符合 `config.FEATURE_DIM` 的要求（通过填充或截断）。
    """
    # --- CORRECTED: __init__ accepts richer features ---
    def __init__(self, id, task_id, cpu_cycles, data_in, data_out, task_type_embedding):
        self.id = id # Unique identifier (e.g., "task_k_sub_s")
        self.task_id = task_id # Identifier of the parent DAG task (e.g., "task_k")
        self.cpu_cycles = cpu_cycles # Ck,s

        # --- CORRECTED: Construct richer feature vector ---
        try:
            # Normalize features before combining
            norm_cpu = np.clip(cpu_cycles / config.TASK_CPU_CYCLES_MAX, 0.0, 1.0)
            # Use TASK_DATA_SIZE_MAX for normalization as it represents the scale of task data transfer
            norm_data_in = np.clip(data_in / config.TASK_DATA_SIZE_MAX, 0.0, 1.0)
            norm_data_out = np.clip(data_out / config.TASK_DATA_SIZE_MAX, 0.0, 1.0)

            # Combine normalized features and task type embedding
            # Ensure task_type_embedding is a flat numpy array
            if isinstance(task_type_embedding, list):
                task_type_embedding = np.array(task_type_embedding, dtype=np.float32)
            elif not isinstance(task_type_embedding, np.ndarray):
                 print(f"Warning: task_type_embedding for {id} is not list or ndarray ({type(task_type_embedding)}). Using zeros.")
                 # Fallback: create zeros if type is wrong
                 task_type_embedding = np.zeros(config.FEATURE_DIM - 3) # Adjust size based on expected embedding length

            raw_features = np.concatenate(
                ([norm_cpu, norm_data_in, norm_data_out], task_type_embedding.flatten())
            ).astype(np.float32)

            # Pad or truncate to match config.FEATURE_DIM exactly
            current_len = len(raw_features)
            if current_len < config.FEATURE_DIM:
                padding = np.zeros(config.FEATURE_DIM - current_len, dtype=np.float32)
                self.feature_vector = np.concatenate((raw_features, padding))
            elif current_len > config.FEATURE_DIM:
                self.feature_vector = raw_features[:config.FEATURE_DIM]
            else:
                 self.feature_vector = raw_features

            # Basic check for NaN/Inf
            if not np.all(np.isfinite(self.feature_vector)):
                print(f"Warning: NaN or Inf found in feature vector for {self.id}. Clamping.")
                self.feature_vector = np.nan_to_num(self.feature_vector, nan=0.0, posinf=1.0, neginf=0.0)


        except Exception as e:
            print(f"Error creating feature vector for subtask {self.id}: {e}")
            traceback.print_exc()
            # Fallback to zeros if error occurs
            self.feature_vector = np.zeros(config.FEATURE_DIM, dtype=np.float32)
        # --- END CORRECTION ---

        self.result = None # Placeholder for computation result
        self.hash_value = None # Placeholder for A-LSH hash
        self.is_reused = False # Flag indicating if result was reused

class TaskDAG:
    """
    表示一个计算任务，该任务被建模为一个有向无环图 (DAG)。

    在这个模型中，一个复杂的计算任务由多个相互依赖的子任务 (`SubTask` 对象) 构成。
    DAG的节点代表子任务，有向边代表子任务之间的执行顺序和数据依赖关系。

    属性:
        id (str): 任务DAG的唯一标识符 (例如, "task_k")。
        vehicle_id (str): 发起此任务DAG的车辆的ID。
        graph (nx.DiGraph): 一个 `networkx.DiGraph` 对象，用于存储DAG的结构。
                            图中的节点是子任务ID（以及虚拟的入口/出口节点），
                            节点数据中包含对相应 `SubTask` 对象的引用（如果适用）。
                            边可以带有权重，表示子任务间的数据传输量。
        subtasks (dict[str, SubTask]): 一个字典，将子任务ID映射到其对应的 `SubTask` 对象。
                                       这提供了一种快速访问特定子任务的方式。
        entry_node (str): DAG中的虚拟入口节点的ID。所有没有前驱的实际子任务都连接自此节点。
        exit_node (str): DAG中的虚拟出口节点的ID。所有没有后继的实际子任务都连接至此节点。
        data_size (float): 整个任务DAG的总数据大小 (Dk)。这个值主要用于估算当整个任务
                           被卸载到RSU时，初始数据传输的成本。子任务之间的具体数据传输量
                           可能在图的边上定义。
        num_task_types (int): 用于生成子任务特征时，任务类型的数量。这影响子任务
                              特征向量中任务类型嵌入（如one-hot编码）的维度。

    核心方法:
        _create_dag(): 在 `TaskDAG` 对象初始化时被调用。此方法负责：
                       1. 根据配置参数 (`config.MIN_SUBTASKS`, `config.MAX_SUBTASKS`)
                          随机确定子任务的数量。
                       2. 创建虚拟的入口和出口节点。
                       3. 为每个子任务实例化一个 `SubTask` 对象（具有随机生成的CPU周期、
                          输入/输出数据大小和任务类型嵌入），并将其添加到 `graph` 和 `subtasks` 字典中。
                       4. 在图中添加边来定义子任务之间的依赖关系。当前实现中，这通常是
                          一个简化的线性链结构，但可以扩展为更复杂的DAG拓扑。
                          边也可能包含数据传输大小的信息。
        get_topological_nodes(): 返回DAG中所有实际子任务节点ID的拓扑排序列表。
                                 这个顺序对于按正确的依赖关系处理子任务至关重要。
                                 它会排除虚拟的入口和出口节点。
        get_subtask(subtask_id): 根据给定的子任务ID从 `subtasks` 字典中检索 `SubTask` 对象。

    `TaskDAG` 是VEC仿真中工作负载的基本单位。卸载算法需要为每个 `TaskDAG` 做出决策，
    而GAT模型则直接处理其图结构 (`graph`) 以提取特征并进行预测。
    """
    def __init__(self, id, vehicle_id):
        self.id = id # Unique task ID (e.g., "task_k")
        self.vehicle_id = vehicle_id
        self.graph = nx.DiGraph()
        self.subtasks = {} # Store SubTask objects, keyed by subtask_id
        self.entry_node = None
        self.exit_node = None
        # Overall task data size used for transmission cost estimate if offloaded
        self.data_size = np.random.uniform(config.TASK_DATA_SIZE_MIN, config.TASK_DATA_SIZE_MAX) # Dk
        # --- CORRECTED: Define number of task types based on FEATURE_DIM ---
        # Example: Assume 3 base features (cpu, data_in, data_out) + one-hot encoding for type
        self.num_task_types = max(1, config.FEATURE_DIM - 3) # Ensure at least 1 type if FEATURE_DIM <= 3
        # --- END CORRECTION ---
        self._create_dag()

    def _create_dag(self):
        """
        在TaskDAG对象初始化时，创建内部的子任务有向无环图结构。

        此方法是一个内部辅助函数，负责：
        1.  随机确定DAG中实际子任务的数量，范围在 `config.MIN_SUBTASKS` 和
            `config.MAX_SUBTASKS` 之间。
        2.  创建虚拟的入口 (`self.entry_node`) 和出口 (`self.exit_node`) 节点，
            并将它们添加到 `self.graph` 中。
        3.  对于每个要创建的实际子任务：
            a.  生成唯一的子任务ID。
            b.  随机生成子任务的属性，如CPU周期数 (`cpu_cycles`)、概念上的输入/输出数据大小
                (`data_in`, `data_out`)，以及任务类型嵌入 (`task_type_embedding`)。
                这些原始属性将用于构造子任务的特征向量。
            c.  实例化一个 `SubTask` 对象，并将此对象存储在 `self.subtasks` 字典中
                （以子任务ID为键），同时将其作为节点添加到 `self.graph` 中（节点数据
                包含对 `SubTask` 对象的引用）。
            d.  处理在 `SubTask` 实例化过程中可能发生的任何错误。
        4.  如果未能成功创建任何有效的子任务节点，则直接连接入口和出口节点。
        5.  在 `self.graph` 中添加边来定义子任务之间的依赖关系：
            a.  连接虚拟入口节点到第一个实际子任务节点。
            b.  （当前简化实现）以线性链的方式连接实际子任务节点，即第 `i` 个子任务
                连接到第 `i+1` 个子任务。边上可以附带随机生成的数据传输大小。
                未来可以扩展为更复杂的DAG拓扑结构（如分支、合并）。
            c.  连接最后一个实际子任务节点到虚拟出口节点。

        副作用:
            - 修改 `self.graph` (NetworkX DiGraph) 以包含节点和边。
            - 修改 `self.subtasks` (字典) 以存储创建的 `SubTask` 实例。
            - 设置 `self.entry_node` 和 `self.exit_node` 属性。
        """
        num_subtasks = np.random.randint(config.MIN_SUBTASKS, config.MAX_SUBTASKS + 1)
        if num_subtasks <= 0:
             print(f"Warning: Task {self.id} created with 0 subtasks.")
             # Add edge from entry to exit directly if no subtasks
             self.entry_node = f"{self.id}_entry"
             self.exit_node = f"{self.id}_exit"
             self.graph.add_node(self.entry_node, data={'subtask': None})
             self.graph.add_node(self.exit_node, data={'subtask': None})
             self.graph.add_edge(self.entry_node, self.exit_node, data_transfer=0)
             return

        self.entry_node = f"{self.id}_entry"
        self.exit_node = f"{self.id}_exit"
        self.graph.add_node(self.entry_node, data={'subtask': None}) # Virtual entry
        self.graph.add_node(self.exit_node, data={'subtask': None})  # Virtual exit

        # Create subtask nodes
        subtask_ids = []
        for s in range(num_subtasks):
            subtask_id = f"{self.id}_sub_{s}"
            subtask_ids.append(subtask_id)
            cpu_cycles = np.random.uniform(config.TASK_CPU_CYCLES_MIN, config.TASK_CPU_CYCLES_MAX)

            # --- CORRECTED: Generate richer features ---
            # Example: input/output data per subtask, task type
            # These are conceptual sizes for feature representation, not necessarily matching edge transfers exactly
            data_in = np.random.uniform(config.TASK_DATA_SIZE_MIN * 0.05, config.TASK_DATA_SIZE_MAX * 0.2)
            data_out = np.random.uniform(config.TASK_DATA_SIZE_MIN * 0.05, config.TASK_DATA_SIZE_MAX * 0.2)

            # Generate one-hot encoding for task type
            task_type_index = np.random.randint(0, self.num_task_types)
            task_type_embedding = np.zeros(self.num_task_types, dtype=np.float32)
            task_type_embedding[task_type_index] = 1.0
            # --- END CORRECTION ---

            try:
                 # Pass new features to SubTask constructor
                 subtask_obj = SubTask(id=subtask_id, task_id=self.id, cpu_cycles=cpu_cycles,
                                       data_in=data_in, data_out=data_out, task_type_embedding=task_type_embedding)
                 self.subtasks[subtask_id] = subtask_obj
                 self.graph.add_node(subtask_id, data={'subtask': subtask_obj})
            except Exception as e:
                 print(f"Error instantiating SubTask {subtask_id}: {e}")
                 traceback.print_exc()
                 # Skip adding this node if subtask creation failed
                 continue

        # Filter out any potential None entries in subtask_ids if creation failed
        valid_subtask_ids = [sid for sid in subtask_ids if sid in self.graph]

        if not valid_subtask_ids:
            print(f"Warning: Task {self.id} has no valid subtask nodes after creation attempts.")
            # Connect entry to exit if all subtask creations failed
            self.graph.add_edge(self.entry_node, self.exit_node, data_transfer=0)
            return

        # --- Add edges (Connect entry, connect subtasks, connect exit) ---
        # Connect entry to the first valid subtask
        self.graph.add_edge(self.entry_node, valid_subtask_ids[0], data_transfer=0) # No data transfer from virtual entry

        # Connect subtasks (Simplified: linear chain for now)
        # A more complex DAG structure could be implemented here (e.g., random branching/merging)
        for i in range(len(valid_subtask_ids) - 1):
             # Data transfer size between actual subtasks
             # Could be related to the data_out of the source or data_in of the destination
             transfer_size = np.random.uniform(0.1, 0.5) * self.data_size # Example fraction of total task data size
             self.graph.add_edge(valid_subtask_ids[i], valid_subtask_ids[i+1], data_transfer=transfer_size)

        # Connect the last valid subtask to exit
        self.graph.add_edge(valid_subtask_ids[-1], self.exit_node, data_transfer=0) # No data transfer to virtual exit


    def get_topological_nodes(self):
        """
        返回TaskDAG中所有实际子任务节点的ID列表，按拓扑顺序排列。

        拓扑排序确保了对于图中的任意有向边 (u, v)，节点 u 都在节点 v 之前出现。
        这对于按正确的依赖顺序处理子任务至关重要（例如，在模拟执行或计算成本时）。
        此方法会从排序结果中排除虚拟的入口 (`self.entry_node`) 和出口
        (`self.exit_node`) 节点。

        该方法首先检查图是否确实是一个有向无环图（DAG）。如果图中存在环路，
        则无法进行拓扑排序，此时会打印错误信息并返回空列表。

        返回:
            list[str]: 一个包含实际子任务节点ID的字符串列表，按拓扑顺序排列。
                       如果图不是DAG，或者在排序过程中发生其他错误，
                       或者图中没有实际的子任务节点，则返回空列表。
        """
        try:
            # Ensure graph has more than just entry/exit before sorting
            if len(self.graph) > 2 and nx.is_directed_acyclic_graph(self.graph):
                 nodes = list(nx.topological_sort(self.graph))
                 # Filter out virtual nodes
                 return [n for n in nodes if n != self.entry_node and n != self.exit_node]
            elif not nx.is_directed_acyclic_graph(self.graph):
                 print(f"Error: Graph for task {self.id} contains a cycle.")
                 # Attempt to find cycles for debugging
                 try:
                     cycles = list(nx.simple_cycles(self.graph))
                     print(f"  Cycles found: {cycles}")
                 except Exception as cycle_err:
                     print(f"  Could not find cycles due to error: {cycle_err}")
                 return []
            else:
                 # Graph has only entry/exit or is empty after filtering
                 return []
        except nx.NetworkXUnfeasible:
            # This exception might be raised if the graph has cycles and topological_sort is called
            print(f"Error: Graph for task {self.id} is not a DAG (NetworkXUnfeasible).")
            return []
        except Exception as e:
             print(f"Error during topological sort for task {self.id}: {e}")
             traceback.print_exc()
             return []


    def get_subtask(self, subtask_id):
        """
        根据子任务ID从TaskDAG的 `subtasks` 字典中检索 `SubTask` 对象。

        参数:
            subtask_id (str): 要检索的子任务的唯一ID。

        返回:
            SubTask or None: 如果找到了具有指定ID的子任务，则返回对应的 `SubTask` 对象。
                             否则（即子任务ID不存在于 `self.subtasks` 字典中），返回 `None`。
        """
        return self.subtasks.get(subtask_id)