import numpy as np

# --- Simulation Parameters ---
NUM_VEHICLES = 15
NUM_RSUS = 5
NUM_TASKS_PER_SLOT = 30 # Average, can vary slightly
MIN_SUBTASKS = 3
MAX_SUBTASKS = 10
SIMULATION_SLOTS = 100 # Number of time slots to simulate for results like Fig 2, 3
EPOCHS_IL = 100        # Training epochs for IL
EPOCHS_DRL_PER_SLOT_UPDATE = 1 # Number of DRL updates per slot training phase

# --- Task Parameters ---
TASK_CPU_CYCLES_MIN = 1e8
TASK_CPU_CYCLES_MAX = 1e9
TASK_DATA_SIZE_MIN = 1e6
TASK_DATA_SIZE_MAX = 1e7 # bits
FEATURE_DIM = 5 # Simplified feature vector dimension

# --- VEC Infrastructure Parameters (Ref Table I & Assumptions) ---
# NOTE: These MIN/MAX values define the *default* random range for RSUs
# during environment initialization, used potentially in Fig 1, 2, 3 baselines.
# They are NOT directly used for the Fig 4/5 x-axis range anymore.
VEHICLE_CPU_FREQ = 1.5e8 # Hz (15 MHz)
RSU_CPU_FREQ_TOTAL_MIN = 200e6 # Hz (200 MHz) - Default Min
RSU_CPU_FREQ_TOTAL_MAX = 250e6 # Hz (250 MHz) - Default Max
RSU_BANDWIDTH_TOTAL = 200e6 # Hz (200 MHz for Mbps -> Use Shannon) - Assuming this is total BW per RSU
VEHICLE_POWER_COMPUTE = 1 # Watts (Assumed, typical for mobile CPU?) - P_i in Eq (5)
VEHICLE_POWER_TRANSMIT = 15 # Watts (Assumed, typical for V2X?) - P_i,j^(trans) in Eq (3)
RSU_CACHE_CAPACITY = 500 # items
CHANNEL_BANDWIDTH_PER_VEHICLE = 10e6 # Hz (10 MHz) - Assume RSU BW is divided equally or managed; b_ij in Eq(1) - SIMPLIFICATION
NOISE_POWER_SPECTRAL_DENSITY = 1e-20 # W/Hz (Typical value for σ^2 / BW -> σ^2 = N0 * BW)
CHANNEL_GAIN_MIN = 1e-7 # Path loss effect (Placeholder)
CHANNEL_GAIN_MAX = 1e-6 # Path loss effect (Placeholder)
CHANNEL_NOISE_STDDEV = 1e-8 # Small perturbation added each slot to channel gain

# --- Algorithm Parameters (Ref Table I & Assumptions) ---
ALPHA = 0.5 # Weight for latency in objective
BETA = 0.5  # Weight for energy in objective

# A-LSH Cache Parameters
CACHE_TIME_DECAY_FACTOR = 0.2 # Ω in Eq (11) - Note: Cache strategy simplified to FIFO/LRU for now
INITIAL_BUCKET_WIDTH_D0 = 0.1 # d0 in Eq (6) - Default/Initial value
MAX_WIDTH_ADJUST_FACTOR = 1.0 # ψ in Eq (6)
# --- !!! mu 值: 您上次修改为 1e-7 !!! ---
# 根据之前的讨论，您可能需要进一步调整这个值以获得 Figure 4 的预期行为
LOAD_THRESHOLD_SENSITIVITY = 2e-7 # μ in Eq (6) - Experiment with this value (e.g., 1e-7, 5e-7, 1e-6, 1e-5 ...)
# --------------------------------------
NUM_HASH_TABLES = 4 # L
NUM_HASH_FUNCTIONS_PER_TABLE = 8 # G
REUSE_SIMILARITY_THRESHOLD = 0.1 # δ in Eq (8) related check

# GAT Model Parameters (Assumed)
GAT_HIDDEN_DIM = 64
GAT_OUTPUT_DIM_ACTOR_DISCRETE = NUM_RSUS + 1 # Output logits for N RSUs + 1 Local
GAT_OUTPUT_DIM_ACTOR_CONTINUOUS = 2 # Output mean/std for CPU freq and BW allocation (Simplified: just output values)
GAT_OUTPUT_DIM_CRITIC = 1 # Output state value
GAT_ATTENTION_HEADS = 4 # Z in paper (Eq 22)
GAT_LAYERS = 2 # Number of GAT layers (Implied by H' calculation)
DROPOUT_RATE = 0.1

# Learning Parameters (Assumed / Ref Table I)
LEARNING_RATE_IL = 2e-5
LEARNING_RATE_DRL_ACTOR = 5e-5
LEARNING_RATE_DRL_CRITIC = 5e-5
LOSS_WEIGHT_LAMBDA1 = 1.0 # Weight for offloading decision loss (CE)
LOSS_WEIGHT_LAMBDA2 = 0.5 # Weight for CPU allocation loss (MSE)
LOSS_WEIGHT_LAMBDA3 = 0.5 # Weight for BW allocation loss (MSE)

# DRL Specific Parameters (Assumed)
DRL_GAMMA = 0.99 # Discount factor
DRL_BATCH_SIZE = 64
DRL_BUFFER_SIZE = 10000
REWARD_SCALE = 1.0 # Scale reward if needed

# Expert Generation Parameters (Assumed)
B_B_EXPERT_SAMPLES = 1000
B_B_MAX_DEPTH = 15
B_B_BEAM_WIDTH = 5
HEURISTIC_TEMPERATURE = 0.1

# --- Plotting Specific Parameters ---
DELTA_VALUES_FIG3 = [0.05, 0.1, 0.15]
D0_VALUES_FIG4_5 = [0.1, 0.2, 0.3]

# --- !!! MODIFIED: Define fj range specifically for Fig 4 & 5 !!! ---
# This list controls the x-axis values for the simulations run *only* for Fig 4 and 5.
NEW_FJ_MIN_FIG4_5 = 0.1 * 1e9 # 0.1 GHz = 1e8 Hz
NEW_FJ_MAX_FIG4_5 = 1.0 * 1e9 # 1.0 GHz = 1e9 Hz
NUM_POINTS_FIG4_5 = 10 # Number of data points to generate across the range
F_J_VALUES_FIG4_5 = np.linspace(NEW_FJ_MIN_FIG4_5, NEW_FJ_MAX_FIG4_5, NUM_POINTS_FIG4_5)
# --- END MODIFICATION ---

# f_j_used_fixed_for_fig4_5 = (RSU_CPU_FREQ_TOTAL_MIN + RSU_CPU_FREQ_TOTAL_MAX) / 2 # Not used in dynamic load scenario

# Constraints (Ref Paper Eq 18, 19)
MAX_TOLERABLE_LATENCY = 5.0 # seconds (ti,k_max) - Example value
MAX_VEHICLE_ENERGY = 10.0 # Joules (Ei,max) - Example value

# --- Utility ---
def get_noise_power(bandwidth, noise_density=NOISE_POWER_SPECTRAL_DENSITY):
    """Calculate noise power based on bandwidth."""
    if noise_density <= 0:
        print(f"Warning: Noise power spectral density is non-positive ({noise_density}). Using fallback small value.")
        return 1e-21
    noise = noise_density * bandwidth
    return max(1e-21, noise)