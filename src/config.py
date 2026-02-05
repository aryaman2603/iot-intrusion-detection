import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

TRAIN_DATA_PATH = os.path.join(DATA_DIR, "processed", "ciciot2023_clean.parquet")
MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.json")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

SELECTED_FEATURES = ['Header_Length', 'Protocol Type', 'Rate', 'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 
                     'psh_flag_number', 'ack_flag_number', 'ack_count', 'syn_count', 
                     'rst_count', 'TCP', 'ICMP', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'IAT', 'Number']

