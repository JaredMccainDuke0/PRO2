# task.py
import numpy as np
import networkx as nx
import config
import traceback # For detailed error printing

class SubTask:
    """Represents a single subtask within a DAG."""
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
    """Represents a computation task modeled as a DAG."""
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
        """Returns subtask nodes in topological order, excluding virtual entry/exit."""
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
        return self.subtasks.get(subtask_id)