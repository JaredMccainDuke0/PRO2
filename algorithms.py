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
    def __init__(self, capacity=config.DRL_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)
    def add(self, state_data, action_details_train, reward, next_state_data, done):
        # Ensure state_data is on CPU before adding
        self.buffer.append((state_data.cpu(), action_details_train, reward, next_state_data, done))
    def sample(self, batch_size, device):
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
    def __init__(self, expert_data_list):
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
        """Train the IL model using expert data with mini-batches."""
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
                        # Initialize LSH params if not done already (should be done once)
                        if not utils.LSH_PARAMS: utils.initialize_lsh_params()

                        for l in range(config.NUM_HASH_TABLES):
                            try:
                                # get_adaptive_bucket_width is called inside compute_lsh_hash
                                hash_val = utils.compute_lsh_hash(feature_vec, rsu, l)
                                if l == 0: hash_val_for_caching = hash_val # Use first table's hash for caching

                                cached_item = rsu.cache_lookup(hash_val)
                                if cached_item is not None:
                                    cached_feature = cached_item.get('feature_vector')
                                    # Ensure both are numpy arrays before calculating norm
                                    if isinstance(cached_feature, np.ndarray) and isinstance(feature_vec, np.ndarray):
                                         dist = np.linalg.norm(cached_feature - feature_vec)
                                         if dist < config.REUSE_SIMILARITY_THRESHOLD:
                                             if dist < best_match_dist: best_match_dist = dist; cached_result = cached_item.get('result')
                                    else: print(f"Warn: Invalid feature vector types for dist calc in LSH lookup {subtask.id}. Cache:{type(cached_feature)}, Current:{type(feature_vec)}")
                            except Exception as e: print(f"Error during LSH lookup for {subtask.id} (table {l}) on RSU {rsu.id}: {e}")
                        # --- End LSH Lookup ---

                        if cached_result is not None:
                             subtask.is_reused = True; subtask.result = cached_result; cache_hit_count += 1; compute_delay = 0 # Reused delay is 0
                             # print(f"DEBUG: Subtask {subtask.id} reused on RSU {rsu.id}") # Optional debug
                        else:
                             compute_delay = utils.calculate_rsu_compute_delay(subtask, allocated_cpu, is_reused=False)
                             if np.isinf(compute_delay):
                                 raise RuntimeError(f"RSU compute failed for subtask {subtask.id} (Alloc CPU: {allocated_cpu:.2e}, Cycles: {subtask.cpu_cycles:.2e})")

                             # Only cache if computation succeeds and we have a hash
                             if not np.isinf(compute_delay) and hash_val_for_caching is not None:
                                 subtask.result = f"Computed_{subtask.id}" # Simulate result
                                 rsu.cache_add(hash_val_for_caching, subtask.result, subtask.feature_vector)
                        # --- End LSH Logic Block ---

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
    def __init__(self, env: VECEnvironment):
        super().__init__(env) # Calls corrected parent __init__
        self.name = "GAT-IL (no reuse)"
        # Training method is inherited

    # --- Override decide_and_execute to disable reuse ---
    def decide_and_execute(self, task: TaskDAG):
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
        if len(self.replay_buffer) < config.DRL_BATCH_SIZE: return 0.0, 0.0 # Return separate losses
        self.actor.train(); self.critic.train()
        sampled_data = self.replay_buffer.sample(config.DRL_BATCH_SIZE, self.device)
        if sampled_data[0] is None: print("Skipping DRL update: Sampling failed."); return 0.0, 0.0
        state_batch, actions_batch, rewards_tensor, _, dones_tensor = sampled_data # actions_batch is dict of tensors

        # --- Critic Update ---
        self.critic_optimizer.zero_grad()
        current_values = self.critic(state_batch) # Shape [batch, 1]
        # Target is just the reward since we assume 'done=True' for each task
        target_values = rewards_tensor # Shape [batch, 1]

        if current_values.shape != target_values.shape:
             print(f"Error: Critic shape mismatch. Current: {current_values.shape}, Target: {target_values.shape}")
             return 0.0, 0.0 # Skip update if shapes mismatch
        critic_loss = F.mse_loss(current_values, target_values)
        critic_loss.backward();
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0) # Clip critic gradients
        self.critic_optimizer.step();
        avg_critic_loss = critic_loss.item()

        # --- Actor Update (Policy Gradient based on Advantage) ---
        self.actor_optimizer.zero_grad()
        # Detach values from critic's graph for advantage calculation
        advantages = (target_values - current_values).detach() # Shape [batch, 1]

        # Re-evaluate actions under current policy to get log_probs
        offload_prob_current, cpu_mean_current, bw_mean_current = self.actor(state_batch) # Shapes [batch, N+1], [batch, 1], [batch, 1]

        # --- Offload Action Log Prob ---
        offload_dist_current = Categorical(offload_prob_current) # Input shape [batch, num_actions]
        action_indices_offload = actions_batch['offload'] # Shape [batch, 1] from buffer stack
        if action_indices_offload.ndim > 1: action_indices_offload = action_indices_offload.squeeze(-1) # Ensure shape [batch]
        try:
            log_probs_offload = offload_dist_current.log_prob(action_indices_offload) # Shape [batch]
        except ValueError as e:
             print(f"Error calculating log_prob for offload actions: {e}")
             print(f"  Probabilities shape: {offload_prob_current.shape}")
             print(f"  Action indices shape: {action_indices_offload.shape}")
             print(f"  Action indices values: {action_indices_offload}")
             return avg_critic_loss, 0.0 # Skip actor update

        # Actor loss: Maximize E[Adv * log(pi(a|s))] for discrete actions
        # Ensure shapes match: log_probs [batch], advantages [batch, 1] -> squeeze adv
        actor_loss_offload = -(log_probs_offload * advantages.squeeze()).mean() # Calculate mean over batch

        # --- Optional: Add MSE loss for continuous actions (guided by advantage is complex, simple MSE might suffice) ---
        # actor_loss_cpu = F.mse_loss(cpu_mean_current, actions_batch['cpu'])
        # actor_loss_bw = F.mse_loss(bw_mean_current, actions_batch['bw'])
        # Combine losses: Weights can be tuned
        # actor_loss = actor_loss_offload + 0.1 * actor_loss_cpu + 0.1 * actor_loss_bw
        actor_loss = actor_loss_offload # Use only discrete loss for simplicity

        actor_loss.backward();
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0) # Clip actor gradients
        self.actor_optimizer.step();
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