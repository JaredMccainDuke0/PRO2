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
    def __init__(self, input_dim, hidden_dim, output_dim, heads=config.GAT_ATTENTION_HEADS, dropout=config.DROPOUT_RATE, num_layers=config.GAT_LAYERS):
        super().__init__()
        self.num_layers = num_layers; self.dropout = dropout; self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2): self.gat_layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        self.gat_layers.append(GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=dropout))

    def forward(self, data):
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
    def __init__(self):
        super().__init__()
        gat_input_dim = config.FEATURE_DIM + 1
        self.gat = GATNetwork(gat_input_dim, config.GAT_HIDDEN_DIM, config.GAT_HIDDEN_DIM)
        self.offload_head = nn.Linear(config.GAT_HIDDEN_DIM, config.NUM_RSUS + 1)
        self.cpu_alloc_head = nn.Linear(config.GAT_HIDDEN_DIM, 1)
        self.bw_alloc_head = nn.Linear(config.GAT_HIDDEN_DIM, 1)

    def forward(self, data):
        graph_embedding, _ = self.gat(data) # Shape [batch_size, GAT_HIDDEN_DIM]
        offload_logits = self.offload_head(graph_embedding) # Shape [batch_size, num_actions]
        offload_prob = F.softmax(offload_logits, dim=-1) # Shape [batch_size, num_actions]
        cpu_mean = torch.sigmoid(self.cpu_alloc_head(graph_embedding)) # Shape [batch_size, 1]
        bw_mean = torch.sigmoid(self.bw_alloc_head(graph_embedding))   # Shape [batch_size, 1]
        return offload_prob, cpu_mean, bw_mean

class CriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        gat_input_dim = config.FEATURE_DIM + 1
        self.gat = GATNetwork(gat_input_dim, config.GAT_HIDDEN_DIM, config.GAT_HIDDEN_DIM)
        self.value_head = nn.Linear(config.GAT_HIDDEN_DIM, 1) # 输出维度为 1

    def forward(self, data):
        graph_embedding, _ = self.gat(data) # Shape [batch_size, GAT_HIDDEN_DIM]
        # --- 修正: value_head 输出形状已经是 [batch_size, 1] ---
        value = self.value_head(graph_embedding) # Shape [batch_size, 1]
        return value