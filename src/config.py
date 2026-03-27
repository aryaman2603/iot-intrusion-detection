"""
Central configuration for the IoT Intrusion Detection project.
All paths, hyperparameters, and constants live here.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DIR         = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
SAMPLE_DIR      = os.path.join(DATA_DIR, "sample")
MODELS_DIR      = os.path.join(BASE_DIR, "models")

RAW_CSV_PATH         = os.path.join(RAW_DIR,       "Stratified_data.csv")
PROCESSED_PATH       = os.path.join(PROCESSED_DIR, "ciciot2023_clean.parquet")
SAMPLE_PATH          = os.path.join(SAMPLE_DIR,    "sample_5_percent.parquet")
SAMPLE_20_PATH       = os.path.join(SAMPLE_DIR,    "sample_20_percent.parquet")
MODEL_PATH           = os.path.join(MODELS_DIR,    "lgbm_model.txt")
LABEL_ENCODER_PATH   = os.path.join(MODELS_DIR,    "lgb_label_encoder.pkl")
SCALER_PATH          = os.path.join(MODELS_DIR,    "scaler.pkl")
FEATURES_PATH        = os.path.join(MODELS_DIR,    "selected_features.json")
THRESHOLDS_PATH      = os.path.join(MODELS_DIR,    "thresholds.json")
TEST_PROBA_PATH      = os.path.join(MODELS_DIR,    "test_proba.npy")
TEST_LABELS_PATH     = os.path.join(MODELS_DIR,    "test_labels.npy")

# ── Label mapping (raw → category) ────────────────────────────────────────
LABEL_MAPPING = {
    "DDOS-ICMP_FLOOD":         "DDoS",
    "DDOS-UDP_FLOOD":          "DDoS",
    "DDOS-TCP_FLOOD":          "DDoS",
    "DDOS-PSHACK_FLOOD":       "DDoS",
    "DDOS-RSTFINFLOOD":        "DDoS",
    "DDOS-SYN_FLOOD":          "DDoS",
    "DDOS-SYNONYMOUSIP_FLOOD": "DDoS",
    "DDOS-ICMP_FRAGMENTATION": "DDoS",
    "DDOS-UDP_FRAGMENTATION":  "DDoS",
    "DDOS-ACK_FRAGMENTATION":  "DDoS",
    "DDOS-HTTP_FLOOD":         "DDoS",
    "DDOS-SLOWLORIS":          "DDoS",
    "DOS-UDP_FLOOD":           "DoS",
    "DOS-TCP_FLOOD":           "DoS",
    "DOS-SYN_FLOOD":           "DoS",
    "DOS-HTTP_FLOOD":          "DoS",
    "MIRAI-GREETH_FLOOD":      "Mirai",
    "MIRAI-UDPPLAIN":          "Mirai",
    "MIRAI-GREIP_FLOOD":       "Mirai",
    "VULNERABILITYSCAN":       "Recon",
    "RECON-HOSTDISCOVERY":     "Recon",
    "RECON-OSSCAN":            "Recon",
    "RECON-PORTSCAN":          "Recon",
    "RECON-PINGSWEEP":         "Recon",
    "MITM-ARPSPOOFING":        "Spoofing",
    "DNS_SPOOFING":            "Spoofing",
    "DICTIONARYBRUTEFORCE":    "Web_BruteForce",
    "BROWSERHIJACKING":        "Web_BruteForce",
    "COMMANDINJECTION":        "Web_BruteForce",
    "SQLINJECTION":            "Web_BruteForce",
    "XSS":                     "Web_BruteForce",
    "BACKDOOR_MALWARE":        "Web_BruteForce",
    "UPLOADING_ATTACK":        "Web_BruteForce",
    "BENIGN":                  "Benign",
}

CLASSES = ["Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web_BruteForce"]

# ── Feature selection ──────────────────────────────────────────────────────
# Top 25 features identified via LightGBM importance + RFECV on CICIoT2023.
# These cover: flow rate, packet size, flag counts, IAT statistics.
# Override by running src/feature_selection.py which writes FEATURES_PATH.
DEFAULT_FEATURES = [
    "flow_duration", "Header_Length", "Protocol Type",
    "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number",
    "cwr_flag_number", "ack_count", "syn_count", "fin_count",
    "urg_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH",
    "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP",
    "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size",
    "IAT", "Number", "Magnitue", "Radius", "Covariance",
    "Variance", "Weight",
]

# ── Training hyperparameters ───────────────────────────────────────────────
RANDOM_STATE  = 42
TEST_SIZE     = 0.15
VAL_SIZE      = 0.10   # fraction of train used for early stopping

LGBM_PARAMS = {
    "objective":         "multiclass",
    "num_class":         len(CLASSES),
    "metric":            "multi_logloss",
    "boosting_type":     "gbdt",
    "n_estimators":      500,          # reduced from 1000 — early stopping handles the rest
    "learning_rate":     0.1,          # increased from 0.05 — faster convergence on M2
    "num_leaves":        63,           # reduced from 127 — less memory, faster per tree
    "max_depth":         7,
    "min_child_samples": 100,          # increased — prevents overfitting on minority classes
    "subsample":         0.7,
    "subsample_freq":    1,            # required for subsample to take effect
    "colsample_bytree":  0.7,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "class_weight":      "balanced",
    "n_jobs":            -1,           # uses all M2 Pro performance cores
    "random_state":      RANDOM_STATE,
    "verbose":           -1,
}

LGBM_FIT_PARAMS = {
    "callbacks": [],   # populated in training.py with early_stopping + log_evaluation
}

N_CV_FOLDS = 3   # switch to 5 for final production run

# ── Active training data path ──────────────────────────────────────────────
# Use SAMPLE_PATH during development (fast), PROCESSED_PATH for final model.
TRAIN_DATA_PATH = SAMPLE_20_PATH    # ~1.6M rows, good balance of speed vs coverage