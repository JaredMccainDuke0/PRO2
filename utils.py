# utils.py
import numpy as np
import torch
import config
from task import TaskDAG, SubTask
import environment # <-- IMPORT ADDED HERE
import traceback # For printing errors in heuristic

# --- Calculation Functions (Based on Paper Equations) ---

def calculate_transmission_rate(vehicle_power, channel_gain, bandwidth, noise_power):
    """Calculates transmission rate using Shannon formula (Eq 1)."""
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
    """Calculates transmission delay (Eq 2)."""
    if trans_rate <= 1e-9: # Use a small epsilon instead of == 0 for float comparison
        return float('inf') # Avoid division by zero or near-zero
    return data_size / trans_rate

def calculate_transmission_energy(vehicle_power, trans_delay):
    """Calculates transmission energy (Eq 3)."""
    if np.isinf(trans_delay): # If delay is infinite, energy is also infinite (or max)
        return float('inf')
    return vehicle_power * trans_delay

def calculate_local_compute_delay(subtask: SubTask, vehicle_cpu_freq):
    """Calculates local computation delay (Eq 4)."""
    if vehicle_cpu_freq <= 1e-9:
        return float('inf')
    # Ensure cpu_cycles is positive
    if subtask.cpu_cycles <=0: return 0 # Zero cycles takes zero time
    return subtask.cpu_cycles / vehicle_cpu_freq

def calculate_local_compute_energy(vehicle_power_compute, local_delay):
    """Calculates local computation energy (Eq 5)."""
    if np.isinf(local_delay):
        return float('inf')
    return vehicle_power_compute * local_delay

def calculate_rsu_compute_delay(subtask: SubTask, allocated_rsu_freq, is_reused):
    """Calculates RSU computation delay (Part of Eq 9)."""
    if is_reused:
        return 0 # Reused tasks have negligible computation delay on RSU side
    if allocated_rsu_freq <= 1e-9:
        return float('inf')
    if subtask.cpu_cycles <= 0: return 0 # Zero cycles takes zero time

    epsilon = 1 if is_reused else 0
    return (1 - epsilon) * subtask.cpu_cycles / allocated_rsu_freq

def calculate_rsu_compute_energy(rsu_power_per_cycle, cpu_cycles, is_reused):
     """Calculates RSU computation energy (Not explicitly in paper, but needed for system total)."""
     if is_reused:
         return 0
     if cpu_cycles <= 0: return 0
     assumed_energy_per_cycle = 1e-9 # Joules/cycle (EXAMPLE VALUE)
     return cpu_cycles * assumed_energy_per_cycle


# --- Noise Power Calculation Helper ---
def get_noise_power(bandwidth, noise_density=config.NOISE_POWER_SPECTRAL_DENSITY):
    """Calculate noise power based on bandwidth."""
    if noise_density <= 0:
        return 1e-21
    noise = noise_density * bandwidth
    return max(1e-21, noise) # Ensure noise is positive


# --- A-LSH Hashing (Eq 7) ---
LSH_PARAMS = {}

def initialize_lsh_params(num_tables=config.NUM_HASH_TABLES, num_funcs=config.NUM_HASH_FUNCTIONS_PER_TABLE, feature_dim=config.FEATURE_DIM):
    """Initialize random projection vectors and offsets for LSH."""
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
    """Computes the hash value for a feature vector for a specific table (Eq 7)."""
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
    """Calculates the combined objective function value."""
    if np.isinf(latency) or np.isinf(energy):
        return float('inf')
    # Ensure non-negative components before calculation
    latency = max(0, latency)
    energy = max(0, energy)
    return alpha * latency + beta * energy

def get_vehicle_index(vehicle_id):
    """Get numerical index from vehicle ID string (e.g., 'veh_5' -> 5)."""
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
     """Get numerical index from RSU ID string (e.g., 'rsu_2' -> 2)."""
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
    Generates a plausible 'expert' policy using a STOCHASTIC heuristic
    based on estimated costs and softmax selection.
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