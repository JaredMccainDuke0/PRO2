"""
此模块定义了基于图注意力网络（GAT）的神经网络模型结构。

这些模型用于处理以有向无环图（DAG）形式表示的计算任务，并为任务卸载和资源分配决策提供支持。
模块中的核心组件包括：

1.  `dag_to_pyg_data(task_dag, subtask_map)`:
    一个辅助函数，用于将 `networkx` 图表示的任务DAG转换为 `torch_geometric.data.Data` 对象。
    这个对象可以被PyTorch Geometric库中的GAT模型直接使用。它处理节点特征的提取和边信息的转换。

2.  `GATNetwork(input_dim, hidden_dim, output_dim, heads, dropout, num_layers)`:
    一个通用的GAT网络实现。它由多个GAT卷积层组成，能够学习图中节点的表示。
    网络的输出是整个图的嵌入（通过全局平均池化得到）以及各个节点的最终嵌入。
    此网络是后续特定任务模型的基础。

3.  `ILDecisionModel()`:
    一个用于模仿学习（Imitation Learning, IL）的决策模型。
    它内部使用 `GATNetwork` 来处理输入的任务DAG，并输出三个决策头：
    - `offload_head`: 预测任务应该在本地执行还是卸载到某个RSU（分类任务）。
    - `cpu_alloc_head`: 预测卸载到RSU时应分配的CPU资源比例（回归任务，输出经过sigmoid激活）。
    - `bw_alloc_head`: 预测卸载到RSU时应分配的带宽资源比例（回归任务，输出经过sigmoid激活）。

4.  `ActorNetwork()`:
    用于深度强化学习（Deep Reinforcement Learning, DRL）中的Actor网络。
    结构与 `ILDecisionModel` 类似，使用 `GATNetwork` 获取图嵌入。
    输出包括：
    - `offload_prob`: 卸载决策的概率分布（经过softmax）。
    - `cpu_mean`: 建议的CPU分配比例的均值（经过sigmoid）。
    - `bw_mean`: 建议的带宽分配比例的均值（经过sigmoid）。
    这些输出用于在DRL算法中选择动作。

5.  `CriticNetwork()`:
    用于DRL中的Critic网络。
    同样使用 `GATNetwork` 获取图嵌入，并输出一个单一的值，表示输入状态（任务DAG）的价值估计。
    这个价值用于评估Actor网络所选动作的好坏，并指导Actor网络的更新。

所有模型都依赖于 `config.py` 中定义的参数，例如GAT的隐藏层维度、注意力头数、层数等。
它们设计为可以处理批量的图数据，这对于高效训练至关重要。
"""
# gat_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool # Import global_mean_pool
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import config

def dag_to_pyg_data(task_dag: nx.DiGraph, subtask_map):
    """
    将NetworkX格式的任务DAG转换为PyTorch Geometric (PyG) `Data` 对象。

    PyG `Data` 对象是PyTorch Geometric库中用于表示图数据的标准格式，
    可以直接被GAT等图神经网络模型使用。此函数负责从输入的 `task_dag`
    (一个NetworkX `DiGraph`) 中提取节点特征、边连接信息和边属性。

    转换过程：
    1.  为图中的每个节点创建一个从节点ID到整数索引的映射 (`node_map`)。
    2.  遍历图中的每个节点：
        -   从节点数据中获取关联的 `SubTask` 对象。
        -   提取子任务的特征向量 (`subtask.feature_vector`) 和标准化的CPU周期数
            (`subtask.cpu_cycles / config.TASK_CPU_CYCLES_MAX`)。
        -   将这两部分连接起来，形成该节点的特征张量。
        -   如果节点没有关联的子任务（例如虚拟入口/出口节点），则使用零向量作为特征。
    3.  遍历图中的每条边：
        -   将边的源节点和目标节点ID转换为它们对应的整数索引。
        -   提取边的属性，例如标准化的数据传输量
            (`data.get('data_transfer', 0) / config.TASK_DATA_SIZE_MAX`)。
    4.  将收集到的节点特征列表、边索引列表和边属性列表分别转换为PyTorch张量。
    5.  使用这些张量创建一个PyG `Data` 对象，其中：
        -   `data.x`: 节点特征张量。
        -   `data.edge_index`: 边索引张量 (COO格式, 形状为 `[2, num_edges]`)。
        -   `data.edge_attr`: 边属性张量。

    参数:
        task_dag (nx.DiGraph): 要转换的任务DAG，表示为一个NetworkX有向图。
                               节点数据应包含一个 'data' 字典，其中可能有 'subtask'键，
                               对应一个 `SubTask` 对象。边数据可以包含 'data_transfer'。
        subtask_map (dict or None): 一个将子任务ID映射到 `SubTask` 对象的字典。
                                    （当前函数实现似乎直接从 `task_dag.nodes[node_id]['data']['subtask']`
                                    获取子任务，所以此参数可能未被充分利用或用于旧版本逻辑）。

    返回:
        torch_geometric.data.Data: 一个表示输入DAG的PyG `Data` 对象。
    """
    # (此函数保持不变)
    node_features = []; node_map = {node_id: i for i, node_id in enumerate(task_dag.nodes())}; edge_list = []; edge_attr = []
    for node_id in task_dag.nodes():
        node_data = task_dag.nodes[node_id]['data']
        if node_data and 'subtask' in node_data and node_data['subtask']:
            subtask = node_data['subtask']; features = np.concatenate([subtask.feature_vector, [subtask.cpu_cycles / config.TASK_CPU_CYCLES_MAX]]).astype(np.float32); node_features.append(features)
        else: node_features.append(np.zeros(config.FEATURE_DIM + 1, dtype=np.float32))
    for u, v, data in task_dag.edges(data=True):
        edge_list.append([node_map[u], node_map[v]]); transfer = data.get('data_transfer', 0) / config.TASK_DATA_SIZE_MAX; edge_attr.append([transfer])
    x = torch.tensor(np.array(node_features), dtype=torch.float); edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous(); edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


class GATNetwork(nn.Module):
    """
    一个通用的图注意力网络（GAT）模型。

    该网络接收一个（或一批）图作为输入，并学习图中节点的嵌入表示，以及整个图的全局嵌入。
    它由多个GAT卷积层 (`GATConv`) 组成，每层后通常接一个激活函数（如ELU）和Dropout层
    以防止过拟合。

    网络的结构特点：
    - 输入层接收原始节点特征。
    - 中间层使用多头注意力机制来聚合邻居节点的信息，生成更丰富的节点表示。
      不同注意力头的输出会被拼接（concat=True，除了最后一层）。
    - 输出层通常使用单头注意力（concat=False）或将多头输出平均，产生最终的节点嵌入。
    - 为了获得整个图的嵌入，通常在最后一层GATConv之后应用一个全局池化操作
      （例如 `global_mean_pool`），它将所有节点的嵌入聚合为一个固定大小的向量。

    参数:
        input_dim (int): 输入节点特征的维度。
        hidden_dim (int): GAT层隐藏单元的数量。
        output_dim (int): GAT网络输出的图嵌入的维度（也是最后一层GATConv的输出节点嵌入维度，
                          如果最后一层不进行拼接且只有一个头）。
        heads (int): GAT层中注意力头的数量。默认为 `config.GAT_ATTENTION_HEADS`。
        dropout (float): Dropout比率。默认为 `config.DROPOUT_RATE`。
        num_layers (int): GAT网络的层数。默认为 `config.GAT_LAYERS`。

    前向传播 (`forward(data)`):
        输入 `data` 是一个 `torch_geometric.data.Data` 对象（或 `Batch` 对象），
        包含 `x` (节点特征), `edge_index` (边连接), 和 `batch` (批处理索引)。
        输出:
            - `graph_embedding`: 图（或批次中每个图）的全局嵌入向量。
                               形状为 `[batch_size, output_dim]`。
            - `x`: 更新后的节点嵌入。形状为 `[num_nodes_in_batch, output_dim]`。
    """
    def __init__(self, input_dim, hidden_dim, output_dim, heads=config.GAT_ATTENTION_HEADS, dropout=config.DROPOUT_RATE, num_layers=config.GAT_LAYERS):
        super().__init__()
        self.num_layers = num_layers; self.dropout = dropout; self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2): self.gat_layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        self.gat_layers.append(GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=dropout))

    def forward(self, data):
        """
        定义GAT网络的前向传播逻辑。

        参数:
            data (torch_geometric.data.Data or torch_geometric.data.Batch):
                输入的图数据。它可以是单个图的 `Data` 对象，或者是多个图批处理后的
                `Batch` 对象。该对象必须包含以下属性：
                - `x`: 节点特征张量，形状 `[num_nodes, input_dim]`。
                - `edge_index`: 边索引张量 (COO格式)，形状 `[2, num_edges]`。
                - `batch`: 批处理索引张量（如果输入是 `Batch` 对象），形状 `[num_nodes]`，
                           指示每个节点属于批次中的哪个图。如果输入是单个 `Data` 对象，
                           `batch` 会自动生成为全零张量。

        返回:
            tuple:
                - graph_embedding (torch.Tensor): 图（或批次中每个图）的全局嵌入向量。
                                                  形状为 `[batch_size, output_dim]`。
                                                  通过对最后一层GAT的节点嵌入应用
                                                  `global_mean_pool` 获得。
                - x (torch.Tensor): 更新后的节点嵌入。
                                    形状为 `[num_nodes_in_batch, output_dim]`。
                                    这是最后一层GATConv的输出。
        """
        # --- 修正: 接收批处理对象 data (或单个 Data) ---
        x, edge_index, batch = data.x, data.edge_index, data.batch # 获取 batch 索引

        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.gat_layers[i](x, edge_index)
            if i < self.num_layers - 1: x = F.elu(x)

        # --- 修正: 使用 global_mean_pool 处理批处理 ---
        # global_mean_pool 需要节点特征 x 和 batch 索引
        # 输出形状将是 [batch_size, output_dim]
        graph_embedding = global_mean_pool(x, batch)

        return graph_embedding, x # 返回图嵌入 [batch_size, output_dim] 和节点嵌入


class ILDecisionModel(nn.Module):
    """
    用于模仿学习（Imitation Learning, IL）的决策模型。

    该模型基于 `GATNetwork` 来处理输入的任务DAG（表示为图数据），
    并从学习到的图嵌入中预测卸载决策和资源分配比例。
    它的目标是模仿专家策略提供的决策。

    结构:
    1.  **GAT基础网络 (`self.gat`)**: 一个 `GATNetwork` 实例，用于将输入的
        任务DAG（`torch_geometric.data.Data` 对象）编码为一个固定维度的图嵌入向量。
        输入节点特征维度由 `config.FEATURE_DIM + 1` （特征+CPU周期）确定，
        GAT输出的图嵌入维度为 `config.GAT_HIDDEN_DIM`。

    2.  **决策头 (Decision Heads)**: 三个独立的线性层，接收来自GAT的图嵌入作为输入：
        -   `self.offload_head`: 输出卸载决策的logits。这是一个分类任务，
            预测任务是在本地执行还是卸载到 `config.NUM_RSUS` 个RSU中的某一个。
            输出维度为 `config.NUM_RSUS + 1`。
        -   `self.cpu_alloc_head`: 预测卸载到RSU时应分配的CPU资源比例。
            这是一个回归任务，输出一个标量值，然后通过Sigmoid函数映射到 (0, 1) 区间。
        -   `self.bw_alloc_head`: 预测卸载到RSU时应分配的带宽资源比例。
            这也是一个回归任务，输出一个标量值，然后通过Sigmoid函数映射到 (0, 1) 区间。

    前向传播 (`forward(data)`):
        输入 `data` 是一个 `torch_geometric.data.Data` 对象（或 `Batch` 对象）。
        1. `data` 通过 `self.gat` 得到图嵌入 `graph_embedding`。
        2. `graph_embedding` 分别输入到三个决策头。
        输出:
            - `offload_logits`: 卸载决策的原始logit值，形状 `[batch_size, NUM_RSUS + 1]`。
            - `cpu_allocation`: 预测的CPU分配比例，形状 `[batch_size, 1]`，值在 (0,1) 之间。
            - `bw_allocation`: 预测的带宽分配比例，形状 `[batch_size, 1]`，值在 (0,1) 之间。

    该模型通常通过监督学习进行训练，损失函数会比较模型的预测（例如，`offload_logits`
    对应的类别概率，以及 `cpu_allocation`, `bw_allocation` 的值）与专家数据集中的
    实际决策。
    """
    def __init__(self):
        super().__init__()
        gat_input_dim = config.FEATURE_DIM + 1
        # GAT 输出图嵌入的维度为 GAT_HIDDEN_DIM
        self.gat = GATNetwork(gat_input_dim, config.GAT_HIDDEN_DIM, config.GAT_HIDDEN_DIM)
        # 决策头输入维度现在是 GAT_HIDDEN_DIM
        self.offload_head = nn.Linear(config.GAT_HIDDEN_DIM, config.NUM_RSUS + 1)
        self.cpu_alloc_head = nn.Linear(config.GAT_HIDDEN_DIM, 1)
        self.bw_alloc_head = nn.Linear(config.GAT_HIDDEN_DIM, 1)

    def forward(self, data):
        """
        定义模仿学习决策模型的前向传播逻辑。

        该方法接收图数据，通过GAT网络提取图嵌入，然后将图嵌入传递给
        三个独立的决策头，分别输出卸载决策的logits、CPU分配比例和带宽分配比例。

        参数:
            data (torch_geometric.data.Data or torch_geometric.data.Batch):
                输入的图数据，与 `GATNetwork.forward` 的输入格式相同。
                包含 `x` (节点特征), `edge_index` (边连接), 和 `batch` (批处理索引)。

        返回:
            tuple:
                - offload_logits (torch.Tensor): 卸载决策的原始logit值。
                                                 形状为 `[batch_size, NUM_RSUS + 1]`。
                                                 需要进一步处理（如Softmax）以获得概率。
                - cpu_allocation (torch.Tensor): 预测的CPU分配比例。
                                                 形状为 `[batch_size, 1]`。
                                                 值已经过Sigmoid激活，在 (0,1) 区间内。
                - bw_allocation (torch.Tensor): 预测的带宽分配比例。
                                                形状为 `[batch_size, 1]`。
                                                值已经过Sigmoid激活，在 (0,1) 区间内。
        """
        # graph_embedding 的形状现在是 [batch_size, GAT_HIDDEN_DIM]
        graph_embedding, _ = self.gat(data)
        # 决策头的输出形状相应地是 [batch_size, num_outputs]
        offload_logits = self.offload_head(graph_embedding) # [batch_size, num_actions]
        cpu_allocation = torch.sigmoid(self.cpu_alloc_head(graph_embedding)) # [batch_size, 1]
        bw_allocation = torch.sigmoid(self.bw_alloc_head(graph_embedding)) # [batch_size, 1]
        # 对于单个图输入 (batch_size=1)，输出形状为 [1, num_outputs]，可能需要 squeeze()
        # 但批处理时保持 [batch_size, ...] 的形状
        return offload_logits, cpu_allocation, bw_allocation


class ActorNetwork(nn.Module):
    """
    深度强化学习（DRL）中的Actor（策略）网络。

    Actor网络负责根据当前观察到的状态（任务DAG的图表示）来决定采取什么动作。
    它输出一个策略，该策略定义了在给定状态下选择每个可能动作的概率（对于离散动作）
    或连续动作的参数（例如均值和标准差）。

    结构:
    1.  **GAT基础网络 (`self.gat`)**: 与 `ILDecisionModel` 类似，使用一个 `GATNetwork`
        实例将输入的任务DAG编码为图嵌入向量。输入和输出维度与 `ILDecisionModel` 中的
        GAT相同。

    2.  **动作输出头 (Action Heads)**:
        -   `self.offload_head`: 输出卸载决策的logits。这些logits随后会通过Softmax函数
            转换为概率分布，表示选择本地执行或卸载到某个RSU的概率。
            输出维度为 `config.NUM_RSUS + 1`。
        -   `self.cpu_alloc_head`: 输出一个值，该值通过Sigmoid函数后表示建议的CPU分配
            比例的均值（对于连续动作空间）。
        -   `self.bw_alloc_head`: 输出一个值，该值通过Sigmoid函数后表示建议的带宽分配
            比例的均值（对于连续动作空间）。

    前向传播 (`forward(data)`):
        输入 `data` 是一个 `torch_geometric.data.Data` 对象（或 `Batch` 对象）。
        1. `data` 通过 `self.gat` 得到图嵌入 `graph_embedding`。
        2. `graph_embedding` 分别输入到三个动作头。
        输出:
            - `offload_prob`: 卸载决策的概率分布，形状 `[batch_size, NUM_RSUS + 1]`。
                              这是 `offload_head` 输出的logits经过Softmax处理后的结果。
            - `cpu_mean`: 建议的CPU分配比例的均值，形状 `[batch_size, 1]`，值在 (0,1) 之间。
            - `bw_mean`: 建议的带宽分配比例的均值，形状 `[batch_size, 1]`，值在 (0,1) 之间。

    在DRL的Actor-Critic架构中，Actor网络通过策略梯度方法进行更新。
    它会根据Critic网络提供的优势估计（Advantage）来调整策略，使得能够获得更高奖励的
    动作被选择的概率增加。
    """
    def __init__(self):
        super().__init__()
        gat_input_dim = config.FEATURE_DIM + 1
        self.gat = GATNetwork(gat_input_dim, config.GAT_HIDDEN_DIM, config.GAT_HIDDEN_DIM)
        self.offload_head = nn.Linear(config.GAT_HIDDEN_DIM, config.NUM_RSUS + 1)
        self.cpu_alloc_head = nn.Linear(config.GAT_HIDDEN_DIM, 1)
        self.bw_alloc_head = nn.Linear(config.GAT_HIDDEN_DIM, 1)

    def forward(self, data):
        """
        定义Actor（策略）网络的前向传播逻辑。

        该方法接收图数据，通过GAT网络提取图嵌入，然后将图嵌入传递给
        动作输出头，生成离散卸载动作的概率分布以及连续资源分配动作的参数（均值）。

        参数:
            data (torch_geometric.data.Data or torch_geometric.data.Batch):
                输入的图数据，与 `GATNetwork.forward` 的输入格式相同。

        返回:
            tuple:
                - offload_prob (torch.Tensor): 卸载决策的概率分布。
                                               形状为 `[batch_size, NUM_RSUS + 1]`。
                                               其元素和在最后一个维度上为1。
                - cpu_mean (torch.Tensor): 建议的CPU分配比例的均值。
                                           形状为 `[batch_size, 1]`。
                                           值已经过Sigmoid激活，在 (0,1) 区间内。
                - bw_mean (torch.Tensor): 建议的带宽分配比例的均值。
                                          形状为 `[batch_size, 1]`。
                                          值已经过Sigmoid激活，在 (0,1) 区间内。
        """
        graph_embedding, _ = self.gat(data) # Shape [batch_size, GAT_HIDDEN_DIM]
        offload_logits = self.offload_head(graph_embedding) # Shape [batch_size, num_actions]
        offload_prob = F.softmax(offload_logits, dim=-1) # Shape [batch_size, num_actions]
        cpu_mean = torch.sigmoid(self.cpu_alloc_head(graph_embedding)) # Shape [batch_size, 1]
        bw_mean = torch.sigmoid(self.bw_alloc_head(graph_embedding))   # Shape [batch_size, 1]
        return offload_prob, cpu_mean, bw_mean

class CriticNetwork(nn.Module):
    """
    深度强化学习（DRL）中的Critic（价值）网络。

    Critic网络的目标是评估当前策略下，处于某个特定状态（任务DAG的图表示）的期望回报（价值）。
    它学习一个状态价值函数 V(s)，这个函数预测从状态 s 开始，遵循当前策略所能获得的累积奖励。

    结构:
    1.  **GAT基础网络 (`self.gat`)**: 与 `ActorNetwork` 和 `ILDecisionModel` 类似，
        使用一个 `GATNetwork` 实例将输入的任务DAG编码为图嵌入向量。
        输入和输出维度与这些模型中的GAT相同。

    2.  **价值头 (`self.value_head`)**: 一个线性层，接收来自GAT的图嵌入作为输入，
        并输出一个标量值。这个标量值就是对输入状态的价值估计。
        输出维度为1。

    前向传播 (`forward(data)`):
        输入 `data` 是一个 `torch_geometric.data.Data` 对象（或 `Batch` 对象）。
        1. `data` 通过 `self.gat` 得到图嵌入 `graph_embedding`。
        2. `graph_embedding` 输入到 `self.value_head`。
        输出:
            - `value`: 对输入状态（或批次中每个状态）的价值估计，形状 `[batch_size, 1]`。

    在Actor-Critic方法中，Critic网络通过最小化时间差分（TD）误差来更新。
    TD误差是根据实际获得的奖励和下一个状态的价值估计来计算的。
    Critic网络学习到的价值函数 V(s) 被用于计算优势函数 A(s, a) = Q(s, a) - V(s)，
    或者直接用 TD目标值 - V(s) 作为优势的估计，这个优势值随后用于指导Actor网络的更新。
    一个好的价值估计对于稳定和加速Actor网络的学习至关重要。
    """
    def __init__(self):
        super().__init__()
        gat_input_dim = config.FEATURE_DIM + 1
        self.gat = GATNetwork(gat_input_dim, config.GAT_HIDDEN_DIM, config.GAT_HIDDEN_DIM)
        self.value_head = nn.Linear(config.GAT_HIDDEN_DIM, 1) # 输出维度为 1

    def forward(self, data):
        """
        定义Critic（价值）网络的前向传播逻辑。

        该方法接收图数据，通过GAT网络提取图嵌入，然后将图嵌入传递给
        价值头，输出对输入状态的价值估计。

        参数:
            data (torch_geometric.data.Data or torch_geometric.data.Batch):
                输入的图数据，与 `GATNetwork.forward` 的输入格式相同。

        返回:
            torch.Tensor: 对输入状态（或批次中每个状态）的价值估计。
                          形状为 `[batch_size, 1]`。
        """
        graph_embedding, _ = self.gat(data) # Shape [batch_size, GAT_HIDDEN_DIM]
        # --- 修正: value_head 输出形状已经是 [batch_size, 1] ---
        value = self.value_head(graph_embedding) # Shape [batch_size, 1]
        return value