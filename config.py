"""
此配置文件包含了车载边缘计算（VEC）仿真项目的所有可调参数和常量。

这些参数分为几个主要类别：

1.  **仿真参数 (Simulation Parameters):**
    控制整体仿真设置，如车辆数量、RSU数量、每个时间槽的任务数量、
    子任务数量范围、总仿真时隙数以及IL和DRL的训练周期数。

2.  **任务参数 (Task Parameters):**
    定义任务的特性，例如CPU周期范围、数据大小范围以及特征向量的维度。

3.  **VEC基础设施参数 (VEC Infrastructure Parameters):**
    描述车辆和RSU的物理属性，如车辆CPU频率、RSU的CPU频率范围、
    RSU总带宽、车辆的计算和传输功率、RSU缓存容量、
    每车辆的信道带宽、噪声功率谱密度、信道增益范围和信道噪声标准差。

4.  **算法参数 (Algorithm Parameters):**
    与卸载决策算法相关的参数，包括目标函数中延迟和能耗的权重（ALPHA, BETA）、
    A-LSH缓存机制的参数（如初始桶宽d0, 负载敏感度μ, 哈希表数量L等）
    以及GAT模型的参数（如隐藏层维度、注意力头数、层数、dropout率）。

5.  **学习参数 (Learning Parameters):**
    用于模型训练的参数，如IL和DRL的学习率、损失函数中不同部分的权重。

6.  **DRL特定参数 (DRL Specific Parameters):**
    强化学习算法特有的参数，如折扣因子GAMMA、批量大小、经验回放缓冲区大小、奖励缩放因子。

7.  **专家生成参数 (Expert Generation Parameters):**
    用于生成专家数据的参数，如专家样本数量、B&B搜索的最大深度和波束宽度、启发式选择的温度系数。

8.  **绘图特定参数 (Plotting Specific Parameters):**
    专门为生成特定图表（如图3、图4、图5）而设定的参数值列表，
    例如δ (相似性阈值) 的不同取值、d0 (初始桶宽) 的不同取值以及f_j (RSU计算能力) 的范围。

9.  **约束条件 (Constraints):**
    定义系统运行的约束，例如最大可容忍延迟和最大车辆能耗。

10. **辅助函数 (Utility):**
    如 `get_noise_power` 用于根据带宽计算噪声功率。

通过修改此文件中的值，可以方便地调整仿真的行为和评估不同场景。
"""
import numpy as np

# --- Simulation Parameters ---
NUM_VEHICLES = 15               # 环境中的车辆数量
NUM_RSUS = 5                    # 环境中的路边单元（RSU）数量
NUM_TASKS_PER_SLOT = 30         # 每个仿真时隙平均生成的任务数量（实际数量可能因泊松分布而略有不同）
MIN_SUBTASKS = 3                # 每个任务DAG中最少的子任务数量
MAX_SUBTASKS = 10               # 每个任务DAG中最多的子任务数量
SIMULATION_SLOTS = 100          # 仿真运行的总时隙数量，主要用于生成图2、图3等结果
EPOCHS_IL = 100                 # 模仿学习（IL）的训练周期数
EPOCHS_DRL_PER_SLOT_UPDATE = 1  # 在DRL训练阶段，每个仿真时隙结束后执行的网络更新次数

# --- Task Parameters ---
TASK_CPU_CYCLES_MIN = 1e8       # 单个子任务所需CPU周期的最小值
TASK_CPU_CYCLES_MAX = 1e9       # 单个子任务所需CPU周期的最大值（也用作归一化基准）
TASK_DATA_SIZE_MIN = 1e6        # 整个任务的总数据量的最小值 (单位: bits)
TASK_DATA_SIZE_MAX = 1e7        # 整个任务的总数据量的最大值 (单位: bits) (也用作归一化基准)
FEATURE_DIM = 5                 # 子任务特征向量的维度（不包括原始CPU周期数，但可能包括标准化的CPU周期）

# --- VEC Infrastructure Parameters (Ref Table I & Assumptions) ---
# 注意: 下面的MIN/MAX值定义了环境初始化时RSU属性的默认随机范围，
#       可能用于图1、2、3的基线仿真。它们不再直接用于图4/5的x轴范围。
VEHICLE_CPU_FREQ = 1.5e8        # 车辆本地CPU的处理频率 (Hz, 例如 150 MHz)
RSU_CPU_FREQ_TOTAL_MIN = 200e6  # RSU总CPU处理频率的默认最小值 (Hz, 例如 200 MHz)
RSU_CPU_FREQ_TOTAL_MAX = 250e6  # RSU总CPU处理频率的默认最大值 (Hz, 例如 250 MHz)
RSU_BANDWIDTH_TOTAL = 200e6     # RSU可用的总带宽 (Hz, 例如 200 MHz)
VEHICLE_POWER_COMPUTE = 1       # 车辆本地计算时的功率消耗 (Watts) - 对应论文 Eq (5) 中的 P_i
VEHICLE_POWER_TRANSMIT = 15     # 车辆数据传输时的功率消耗 (Watts) - 对应论文 Eq (3) 中的 P_i,j^(trans)
RSU_CACHE_CAPACITY = 500        # RSU缓存中可以存储的条目数量
CHANNEL_BANDWIDTH_PER_VEHICLE = 10e6 # 分配给单个车辆进行通信的信道带宽 (Hz, 例如 10 MHz) - 简化假设，对应论文 Eq(1) 中的 b_ij
NOISE_POWER_SPECTRAL_DENSITY = 1e-20 # 噪声功率谱密度 (W/Hz) - 用于计算噪声功率 σ^2 = N0 * BW
CHANNEL_GAIN_MIN = 1e-7         # 信道增益的最小值 (路径损耗效应的占位符)
CHANNEL_GAIN_MAX = 1e-6         # 信道增益的最大值 (路径损耗效应的占位符)
CHANNEL_NOISE_STDDEV = 1e-8     # 每个时隙添加到信道增益中的高斯噪声的标准差，模拟信道波动

# --- Algorithm Parameters (Ref Table I & Assumptions) ---
ALPHA = 0.5                     # 目标函数中延迟项的权重
BETA = 0.5                      # 目标函数中能耗项的权重

# A-LSH Cache Parameters (自适应局部敏感哈希缓存参数)
CACHE_TIME_DECAY_FACTOR = 0.2   # 缓存时间衰减因子 Ω (对应论文 Eq 11) - 注意: 当前缓存策略简化为FIFO/LRU
INITIAL_BUCKET_WIDTH_D0 = 0.1   # 初始（默认）LSH桶宽度 d0 (对应论文 Eq 6)
MAX_WIDTH_ADJUST_FACTOR = 1.0   # 最大桶宽度调整因子 ψ (psi) (对应论文 Eq 6)
# --- !!! mu 值: 您上次修改为 1e-7 !!! ---
# 根据之前的讨论，您可能需要进一步调整这个值以获得 Figure 4 的预期行为
LOAD_THRESHOLD_SENSITIVITY = 2e-7 # RSU负载敏感度 μ (mu) (对应论文 Eq 6) - 需要实验调整此值 (例如, 1e-7, 5e-7, 1e-6, 1e-5 ...)
# --------------------------------------
NUM_HASH_TABLES = 4             # LSH中的哈希表数量 L
NUM_HASH_FUNCTIONS_PER_TABLE = 8 # 每个LSH哈希表中的哈希函数数量 G
REUSE_SIMILARITY_THRESHOLD = 0.1 # 子任务结果复用的相似性阈值 δ (delta) (对应论文 Eq 8 相关检查)

# GAT Model Parameters (图注意力网络模型参数 - 基于假设)
GAT_HIDDEN_DIM = 64             # GAT网络隐藏层的维度
GAT_OUTPUT_DIM_ACTOR_DISCRETE = NUM_RSUS + 1 # Actor网络离散动作（卸载目标）的输出维度 (N个RSU + 1个本地)
GAT_OUTPUT_DIM_ACTOR_CONTINUOUS = 2 # Actor网络连续动作（CPU频率和带宽分配）的输出维度 (简化：直接输出值)
GAT_OUTPUT_DIM_CRITIC = 1       # Critic网络状态值输出的维度
GAT_ATTENTION_HEADS = 4         # GAT层中的注意力头数量 Z (对应论文 Eq 22)
GAT_LAYERS = 2                  # GAT网络的层数 (由 H' 计算隐含)
DROPOUT_RATE = 0.1              # GAT层中的Dropout比率

# Learning Parameters (学习参数 - 基于假设 / 参考论文表 I)
LEARNING_RATE_IL = 2e-5         # 模仿学习（IL）的学习率
LEARNING_RATE_DRL_ACTOR = 5e-5  # 深度强化学习（DRL）Actor网络的学习率
LEARNING_RATE_DRL_CRITIC = 5e-5 # DRL Critic网络的学习率
LOSS_WEIGHT_LAMBDA1 = 1.0       # IL损失函数中卸载决策损失（交叉熵）的权重
LOSS_WEIGHT_LAMBDA2 = 0.5       # IL损失函数中CPU分配损失（MSE）的权重
LOSS_WEIGHT_LAMBDA3 = 0.5       # IL损失函数中带宽分配损失（MSE）的权重

# DRL Specific Parameters (DRL特定参数 - 基于假设)
DRL_GAMMA = 0.99                # DRL中的折扣因子γ
DRL_BATCH_SIZE = 64             # DRL训练时从经验回放缓冲区采样的批量大小
DRL_BUFFER_SIZE = 10000         # DRL经验回放缓冲区的容量
REWARD_SCALE = 1.0              # DRL奖励的缩放因子 (如果需要调整奖励幅度)

# Expert Generation Parameters (专家数据生成参数 - 基于假设)
B_B_EXPERT_SAMPLES = 1000       # 使用B&B（分支界定）或启发式生成的专家样本数量
B_B_MAX_DEPTH = 15              # B&B搜索的最大深度 (如果使用)
B_B_BEAM_WIDTH = 5              # B&B搜索的波束宽度 (如果使用)
HEURISTIC_TEMPERATURE = 0.1     # 启发式专家策略中Softmax选择的温度系数，控制随机性

# --- Plotting Specific Parameters ---
DELTA_VALUES_FIG3 = [0.05, 0.1, 0.15] # 用于图3的相似性阈值δ的测试值列表
D0_VALUES_FIG4_5 = [0.1, 0.2, 0.3]    # 用于图4和图5的初始桶宽度d0的测试值列表

# --- !!! MODIFIED: Define fj range specifically for Fig 4 & 5 !!! ---
# 这个列表控制仅为图4和图5运行的仿真中RSU计算能力fj的x轴取值。
NEW_FJ_MIN_FIG4_5 = 0.1 * 1e9 # 图4/5中fj的最小值 (0.1 GHz = 1e8 Hz)
NEW_FJ_MAX_FIG4_5 = 1.0 * 1e9 # 图4/5中fj的最大值 (1.0 GHz = 1e9 Hz)
NUM_POINTS_FIG4_5 = 10        # 在上述fj范围内生成的仿真数据点数量
F_J_VALUES_FIG4_5 = np.linspace(NEW_FJ_MIN_FIG4_5, NEW_FJ_MAX_FIG4_5, NUM_POINTS_FIG4_5) # 生成的fj值序列
# --- END MODIFICATION ---

# f_j_used_fixed_for_fig4_5 = (RSU_CPU_FREQ_TOTAL_MIN + RSU_CPU_FREQ_TOTAL_MAX) / 2 # 在动态负载场景下未使用此固定值

# Constraints (约束条件 - 参考论文 Eq 18, 19)
MAX_TOLERABLE_LATENCY = 5.0     # 最大可容忍的任务执行延迟 (秒) (ti,k_max) - 示例值
MAX_VEHICLE_ENERGY = 10.0       # 最大可容忍的车辆能耗 (焦耳) (Ei,max) - 示例值

# --- Utility ---
def get_noise_power(bandwidth, noise_density=NOISE_POWER_SPECTRAL_DENSITY):
    """Calculate noise power based on bandwidth."""
    if noise_density <= 0:
        print(f"Warning: Noise power spectral density is non-positive ({noise_density}). Using fallback small value.")
        return 1e-21
    noise = noise_density * bandwidth
    return max(1e-21, noise)