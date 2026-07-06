"""
Global configuration for the HAR GNN+LSTM project.
"""

# ── Dataset paths ──────────────────────────────────────────────────────────────
DATA_DIR = "data"
RAW_DIR = f"{DATA_DIR}/raw"
PROCESSED_DIR = f"{DATA_DIR}/processed"

HHAR_RAW_DIR = f"{RAW_DIR}/hhar"
PAMAP2_RAW_DIR = f"{RAW_DIR}/pamap2"

# ── Preprocessing ──────────────────────────────────────────────────────────────
WINDOW_SIZE = 128          # samples per window (~2.56 s at 50 Hz)
OVERLAP = 0.5              # 50 % overlap between consecutive windows
TARGET_SAMPLING_RATE = 50  # Hz — resample all streams to this rate

# ── HHAR specifics ────────────────────────────────────────────────────────────
HHAR_ACTIVITIES = ["bike", "sit", "stand", "walk", "stairsup", "stairsdown"]
HHAR_SENSORS = ["accel", "gyro"]

# ── PAMAP2 specifics ─────────────────────────────────────────────────────────
PAMAP2_ACTIVITIES = {
    0: "transient",
    1: "lying", 2: "sitting", 3: "standing", 4: "walking",
    5: "running", 6: "cycling", 7: "nordic_walking",
    9: "watching_TV", 10: "computer_work", 11: "car_driving",
    12: "ascending_stairs", 13: "descending_stairs",
    16: "vacuum_cleaning", 17: "ironing", 18: "folding_laundry",
    19: "house_cleaning", 20: "playing_soccer", 24: "rope_jumping",
}
PAMAP2_POSITIONS = ["wrist", "chest", "ankle"]  # IMU positions
PAMAP2_SENSOR_COLS = 40  # columns per subject file (after timestamp)

# ── Graph construction ────────────────────────────────────────────────────────
EDGE_METHOD = "fixed"   # "fixed" | "learnable" | "attention"
# For PAMAP2: wrist-chest, chest-ankle, wrist-ankle
PAMAP2_EDGES = [(0, 1), (1, 2), (0, 2)]
# For HHAR: x-y, y-z, x-z (3 axes fully connected)
HHAR_EDGES = [(0, 1), (1, 2), (0, 2)]

# ── Node feature dims (6 stats × n_channels_per_node) ────────────────────────
# PAMAP2: 18 channels / 3 nodes = 6 ch/node × 6 stats = 36 features/node
PAMAP2_NODE_FEAT_DIM = 36
# HHAR: 1 channel per axis-node × 6 stats = 6 features/node  (3 nodes: x, y, z)
HHAR_NODE_FEAT_DIM = 6

# ── Model hyperparameters ─────────────────────────────────────────────────────
GCN_HIDDEN_DIM = 64
GCN_OUTPUT_DIM = 64
LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 2
DROPOUT = 0.3
MLP_HIDDEN_DIM = 64

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
PATIENCE = 15           # early stopping patience
SEED = 42

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_PROTOCOL_HHAR = "loso"   # leave-one-subject-out
EVAL_PROTOCOL_PAMAP2 = "loso"

# ── Results ───────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
MODELS_DIR = f"{RESULTS_DIR}/models"
PLOTS_DIR = f"{RESULTS_DIR}/plots"
METRICS_DIR = f"{RESULTS_DIR}/metrics"
