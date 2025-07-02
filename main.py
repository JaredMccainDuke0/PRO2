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
    """Pads a list with a specific value to reach the desired length."""
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
    Filters NaN and Inf from multiple lists simultaneously based on the validity
    across *all* provided lists for corresponding indices.
    Returns filtered lists, keeping alignment.
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
    """Generates expert data using a heuristic policy."""
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
    Runs a simulation for a given algorithm INSTANCE and parameters.
    Returns results dictionary and the AVERAGE dj calculated during the run.
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
        if 'fj' in sim_params:
            target_fj = sim_params['fj']
            print(f"  Setting RSU fj = {target_fj:.2e}")
            for rsu_instance in env.rsus:
                if hasattr(rsu_instance, 'total_cpu_frequency'):
                     # Ensure frequency is positive
                     rsu_instance.total_cpu_frequency = max(1e3, target_fj)
                else:
                     print(f"Warning: Cannot set 'total_cpu_frequency' on RSU {getattr(rsu_instance, 'id', 'N/A')}")

        if 'd0' in sim_params:
            target_d0 = sim_params['d0']
            print(f"  Setting config.INITIAL_BUCKET_WIDTH_D0 = {target_d0}")
            config.INITIAL_BUCKET_WIDTH_D0 = target_d0 # 设置全局 d0

        if 'delta' in sim_params:
             config.REUSE_SIMILARITY_THRESHOLD = sim_params['delta']
             print(f"  Setting config.REUSE_SIMILARITY_THRESHOLD = {config.REUSE_SIMILARITY_THRESHOLD}")

    # --- 重置算法状态和结果收集器 ---
    algorithm_instance.results = {'objective': [], 'latency': [], 'energy': [], 'cache_hits': []}
    if hasattr(algorithm_instance, 'all_djs_this_run'):
         algorithm_instance.all_djs_this_run = []
    else:
         print(f"Warning: Algorithm instance {algo_name} missing 'all_djs_this_run'. Creating attribute.")
         algorithm_instance.all_djs_this_run = []
    # --------------------------------

    print(f" Starting simulation loop for {config.SIMULATION_SLOTS} slots...")
    for slot in range(1, config.SIMULATION_SLOTS + 1): # Start slot count from 1
        try:
            tasks = env.step() # env.step() increments slot and resets RSU loads/dj history
            if not tasks:
                 algorithm_instance.results['objective'].append(0.0)
                 algorithm_instance.results['latency'].append(0.0)
                 algorithm_instance.results['energy'].append(0.0)
                 algorithm_instance.results['cache_hits'].append(0)
                 continue
            # run_slot handles task execution and data collection for this slot (including dj collection)
            algorithm_instance.run_slot(tasks)
        except Exception as e:
            print(f"FATAL: Error during simulation slot {env.current_slot} for {algo_name}: {e}")
            traceback.print_exc()
            algorithm_instance.results['objective'].append(np.nan)
            algorithm_instance.results['latency'].append(np.nan)
            algorithm_instance.results['energy'].append(np.nan)
            algorithm_instance.results['cache_hits'].append(0) # Or np.nan? Let's use 0

    print(f" Simulation loop finished for {algo_name}.")

    # --- 计算本次运行的平均 dj --- # REVERTED
    avg_dj_for_run = np.nan
    if hasattr(algorithm_instance, 'all_djs_this_run'):
         all_djs = algorithm_instance.all_djs_this_run
         # Filter for valid float/int numbers, excluding NaN/inf
         valid_djs = [d for d in all_djs if isinstance(d, (int, float)) and np.isfinite(d)]
         if valid_djs:
             avg_dj_for_run = np.mean(valid_djs) # Calculate mean
             print(f"  Avg dj calculated during run: {avg_dj_for_run:.4f} (from {len(valid_djs)} valid samples)")
         else:
             print("  No valid dj values collected during the run.")
    else:
        print("  Cannot calculate avg_dj: 'all_djs_this_run' attribute missing.")
    # -------------------------

    # --- 恢复原始 config 值 ---
    config.REUSE_SIMILARITY_THRESHOLD = original_delta_cfg
    config.INITIAL_BUCKET_WIDTH_D0 = original_d0_cfg
    # ------------------------

    # --- 准备返回结果 ---
    results = algorithm_instance.get_results()
    for key in ['objective', 'latency', 'energy', 'cache_hits']:
         results[key] = pad_list(results.get(key, []), config.SIMULATION_SLOTS)

    # --- 返回结果字典和平均 dj 值 --- # REVERTED
    return results, avg_dj_for_run
# --- END of modified run_simulation ---


# --- Helper function to calculate interval averages ---
# (This function remains unchanged)
def calculate_interval_averages(slots, values, interval_size=10, num_slots=config.SIMULATION_SLOTS):
    """Calculates the average value for each interval, returning interval end as x-coord."""
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
    try:
        base_env = VECEnvironment()
        if not base_env.vehicles or not base_env.rsus:
            raise RuntimeError("Environment initialization failed: No vehicles or RSUs created.")
        print(f"Environment initialized with {len(base_env.vehicles)} vehicles and {len(base_env.rsus)} RSUs.")
    except Exception as e:
        print(f"FATAL: Environment initialization error: {e}")
        traceback.print_exc()
        exit()

    # --- Store Original Config Values ---
    main_original_d0 = config.INITIAL_BUCKET_WIDTH_D0
    main_original_delta = config.REUSE_SIMILARITY_THRESHOLD
    main_original_sim_slots = config.SIMULATION_SLOTS

    # --- Generate Expert Data ---
    expert_dataset = []
    try:
        expert_dataset = generate_il_expert_dataset(config.B_B_EXPERT_SAMPLES, base_env)
        if not expert_dataset:
            print("Warning: Failed to generate any expert dataset samples. IL models will not train.")
        else:
            print(f"Successfully generated {len(expert_dataset)} expert samples.")
    except Exception as e:
        print(f"FATAL: Error during expert data generation: {e}")
        traceback.print_exc()
        # exit() # Optionally exit

    # --- Instantiate and Train IL Models ONCE ---
    print("\n--- Initializing and Training IL Models ---")
    gat_il_noreuse_model = GAT_IL_NoReuse(base_env)
    gat_reuse_il_model = GAT_REUSE_IL(base_env)

    if expert_dataset:
        try:
            print("Training GAT-IL (no reuse)...")
            gat_il_noreuse_model.train(expert_dataset)
            print("Training GAT-IL (no reuse) finished.")
        except Exception as e:
            print(f"Error training GAT_IL_NoReuse: {e}")
            traceback.print_exc()
        try:
            print("Training REUSE-GAT-IL...")
            gat_reuse_il_model.train(expert_dataset)
            print("Training REUSE-GAT-IL finished.")
        except Exception as e:
            print(f"Error training GAT_REUSE_IL: {e}")
            traceback.print_exc()
    else:
        print("Skipping IL training due to empty or failed expert dataset generation.")


    # --- Run Baseline Simulations (Fig 1 & 2) ---
    print("\n--- Running Simulations for Fig 1 & 2 ---")
    # Initialize results variables to store tuple (results_dict, avg_dj) # REVERTED
    results_gat_il_noreuse_fig12 = ({}, np.nan)
    results_gat_reuse_il_fig2 = ({}, np.nan)
    results_gat_reuse_drl_fig2 = ({}, np.nan)
    gat_reuse_drl_instance_fig2 = None # DRL instance specific to this run

    try:
        # Ensure config values are set as needed for baseline
        config.REUSE_SIMILARITY_THRESHOLD = main_original_delta
        config.INITIAL_BUCKET_WIDTH_D0 = main_original_d0
        config.SIMULATION_SLOTS = main_original_sim_slots

        print("Running GAT-IL (noreuse) for Fig 1/2 baseline...")
        results_gat_il_noreuse_fig12 = run_simulation(gat_il_noreuse_model, base_env, name_suffix="_Fig1_2_Base")

        print("Running REUSE-GAT-IL for Fig 2 baseline...")
        results_gat_reuse_il_fig2 = run_simulation(gat_reuse_il_model, base_env, name_suffix="_Fig2_Base")

        print("Running REUSE-GAT-DRL for Fig 2 baseline...")
        gat_reuse_drl_instance_fig2 = GAT_REUSE_DRL(base_env) # Create fresh DRL instance
        results_gat_reuse_drl_fig2 = run_simulation(gat_reuse_drl_instance_fig2, base_env, name_suffix="_Fig2_Base")

    except Exception as e:
        print(f"FATAL: Error during Fig 1/2 simulation runs: {e}")
        traceback.print_exc()
    finally:
        # Restore original config values if they were changed
        config.REUSE_SIMILARITY_THRESHOLD = main_original_delta
        config.INITIAL_BUCKET_WIDTH_D0 = main_original_d0
        config.SIMULATION_SLOTS = main_original_sim_slots

    # --- Run Simulations for Fig 3 (Varying Delta) ---
    print("\n--- Running Simulations for Fig 3 (Varying Delta) ---")
    results_delta_comparison = {}
    fixed_d0_fig3 = 0.2 # Define the fixed d0 for this experiment

    try:
        print(f"  Fixing d0 = {fixed_d0_fig3} for Fig 3 simulations.")
        config.SIMULATION_SLOTS = main_original_sim_slots # Ensure correct length
        d0_before_fig3_loop = config.INITIAL_BUCKET_WIDTH_D0
        config.INITIAL_BUCKET_WIDTH_D0 = fixed_d0_fig3

        for delta_val in config.DELTA_VALUES_FIG3:
            print(f"  Running simulation for delta = {delta_val}...")
            sim_params = {'delta': delta_val, 'd0': fixed_d0_fig3}
            # run_simulation now returns (results_dict, avg_dj) # REVERTED
            results_delta_comparison[delta_val] = run_simulation(
                gat_reuse_il_model, # Reuse the trained instance
                base_env,
                name_suffix=f"_Fig3_delta{delta_val}",
                sim_params=sim_params
            ) # Stores tuple (results_dict, avg_dj)

    except Exception as e:
        print(f"FATAL: Error during Fig 3 simulation runs: {e}")
        traceback.print_exc()
    finally:
        config.INITIAL_BUCKET_WIDTH_D0 = d0_before_fig3_loop # Restore d0

    # --- Modified: Data Collection for Fig 4 & 5 --- # REVERTED data storage
    print("\n--- Collecting dynamic load data for Fig 4 & 5 ---")
    results_dynamic_load_fig4_5 = {fj: {} for fj in config.F_J_VALUES_FIG4_5}
    try:
        config.SIMULATION_SLOTS = main_original_sim_slots # Ensure full simulation length
        original_d0_fig5 = config.INITIAL_BUCKET_WIDTH_D0

        print(f" Running full simulations for Fig 4/5 data ({config.SIMULATION_SLOTS} slots each)...")
        for fj_val in config.F_J_VALUES_FIG4_5:
            fj_key = float(fj_val)
            print(f"  Processing fj = {fj_key:.2e}...")
            results_dynamic_load_fig4_5[fj_key] = {}
            for d0_val in config.D0_VALUES_FIG4_5:
                print(f"    Processing d0 = {d0_val}...")
                sim_params = {'fj': fj_key, 'd0': d0_val}

                # Run simulation with REUSE-GAT-IL
                # run_simulation returns (sim_results_dict, avg_dj) # REVERTED
                sim_results_dict, avg_dj_value = run_simulation(
                    gat_reuse_il_model,
                    base_env,
                    name_suffix=f"_Fig45_dyn_fj{fj_key:.1e}_d0{d0_val}",
                    sim_params=sim_params
                )

                valid_latencies = [l for l in sim_results_dict.get('latency', []) if np.isfinite(l)]
                avg_latency = np.mean(valid_latencies) if valid_latencies else np.nan

                # Store the average latency AND the average dj value # REVERTED
                results_dynamic_load_fig4_5[fj_key][d0_val] = {
                    'avg_latency': avg_latency,
                    'avg_dj': avg_dj_value # Store the average dj here
                }
                print(f"      -> Avg Latency: {avg_latency:.4f}, Avg Dj: {avg_dj_value:.4f}") # Print avg_dj

        config.INITIAL_BUCKET_WIDTH_D0 = original_d0_fig5 # Restore d0

    except Exception as e:
        print(f"FATAL: Error during dynamic data collection for Fig 4/5: {e}")
        traceback.print_exc()
        config.INITIAL_BUCKET_WIDTH_D0 = original_d0_fig5 # Attempt restore even on error
    # --- END of reverted data collection ---


    # --- Plotting (Using reverted plot function for 4, modified 5) ---
    print("\n--- Generating Plots (Avg dj for Fig 4, styled Fig 5) ---") # Updated message
    try:
        # Fig 1: Use model instances directly
        if gat_il_noreuse_model and gat_reuse_il_model:
            plot_figure_1(gat_il_noreuse_model, gat_reuse_il_model)
        else:
            print(" Skipping Fig 1 plot due to missing model instances.")

        # Fig 2: Use results from baseline runs
        valid_fig2_data = (results_gat_il_noreuse_fig12 and results_gat_il_noreuse_fig12[0] and
                           results_gat_reuse_il_fig2 and results_gat_reuse_il_fig2[0] and
                           results_gat_reuse_drl_fig2 and results_gat_reuse_drl_fig2[0])
        if valid_fig2_data:
             plot_figure_2(results_gat_il_noreuse_fig12, results_gat_reuse_il_fig2, results_gat_reuse_drl_fig2)
        else:
             print(" Skipping Fig 2 plot due to missing baseline simulation data.")

        # Fig 3: Use results from delta comparison runs
        if results_delta_comparison:
             plot_figure_3(results_delta_comparison, fixed_d0_used=fixed_d0_fig3)
        else:
             print(" Skipping Fig 3 plot due to missing delta comparison data.")

        # Fig 4 & 5: Use results from dynamic load runs
        if results_dynamic_load_fig4_5:
            print("Plotting Figure 4 with dynamic load data (Average dj)...")
            plot_figure_4(results_dynamic_load_fig4_5) # Calls reverted plot_figure_4

            print("Plotting Figure 5 with dynamic load data (Average Latency, styled)...")
            plot_figure_5(results_dynamic_load_fig4_5) # Calls modified plot_figure_5
        else:
            print(" Skipping Fig 4 & 5 plots due to missing dynamic load data.")

    except Exception as e:
        print(f"FATAL: Plotting error: {e}")
        traceback.print_exc()
    # --- END of modified plotting section ---

    # --- Final Restoration (Optional, good practice) ---
    config.INITIAL_BUCKET_WIDTH_D0 = main_original_d0
    config.REUSE_SIMILARITY_THRESHOLD = main_original_delta
    config.SIMULATION_SLOTS = main_original_sim_slots
    print(f"\nRestored original config: d0={config.INITIAL_BUCKET_WIDTH_D0}, delta={config.REUSE_SIMILARITY_THRESHOLD}, slots={config.SIMULATION_SLOTS}")

    print("\n--- Plotting phase complete ---")
    print("--- Script execution finished ---")

