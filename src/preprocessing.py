import polars as pl
import os


RAW_CSV_NAME = "Stratified_data.csv" 


DATA_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
SAMPLE_DIR = "data/sample"


INPUT_FILE_PATH = os.path.join(DATA_DIR, RAW_CSV_NAME)
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, "ciciot2023_clean.parquet")
SAMPLE_FILE_PATH = os.path.join(SAMPLE_DIR, "sample_5_percent.parquet")

label_mapping = {
        
        'DDOS-ICMP_FLOOD': 'DDoS',
        'DDOS-UDP_FLOOD': 'DDoS',
        'DDOS-TCP_FLOOD': 'DDoS',
        'DDOS-PSHACK_FLOOD': 'DDoS',
        'DDOS-RSTFINFLOOD': 'DDoS',
        'DDOS-SYN_FLOOD': 'DDoS',
        'DDOS-SYNONYMOUSIP_FLOOD': 'DDoS',
        'DDOS-ICMP_FRAGMENTATION': 'DDoS',
        'DDOS-UDP_FRAGMENTATION': 'DDoS',
        'DDOS-ACK_FRAGMENTATION': 'DDoS',
        'DDOS-HTTP_FLOOD': 'DDoS',
        'DDOS-SLOWLORIS': 'DDoS',
        
        
        'DOS-UDP_FLOOD': 'DoS',
        'DOS-TCP_FLOOD': 'DoS',
        'DOS-SYN_FLOOD': 'DoS',
        'DOS-HTTP_FLOOD': 'DoS',
        
        
        'MIRAI-GREETH_FLOOD': 'Mirai',
        'MIRAI-UDPPLAIN': 'Mirai',
        'MIRAI-GREIP_FLOOD': 'Mirai',
        
        
        'VULNERABILITYSCAN': 'Recon',
        'RECON-HOSTDISCOVERY': 'Recon',
        'RECON-OSSCAN': 'Recon',
        'RECON-PORTSCAN': 'Recon',
        'RECON-PINGSWEEP': 'Recon',
        
        
        'MITM-ARPSPOOFING': 'Spoofing',
        'DNS_SPOOFING': 'Spoofing',
        
        
        'DICTIONARYBRUTEFORCE': 'Web_BruteForce',
        'BROWSERHIJACKING': 'Web_BruteForce',
        'COMMANDINJECTION': 'Web_BruteForce',
        'SQLINJECTION': 'Web_BruteForce',
        'XSS': 'Web_BruteForce',
        'BACKDOOR_MALWARE': 'Web_BruteForce',
        'UPLOADING_ATTACK': 'Web_BruteForce',
        
        
        'BENIGN': 'Benign'
    }

def process_single_csv():
    print(f"Connecting to raw file: {INPUT_FILE_PATH}")
    
    
    q = (
        pl.scan_csv(INPUT_FILE_PATH, infer_schema_length=10000)

        .rename({"Label": "label"})
        
        .with_columns(pl.col("label").replace(label_mapping).alias("label_category"))
        
        .with_columns(pl.all().exclude("label", "label_category").cast(pl.Float32))
        
        
        .filter(pl.all_horizontal(pl.all().exclude("label", "label_category").is_finite()))
        .drop_nulls()
    )

    print(" Streaming data to Parquet (this will take 2-5 mins)...")
    q.sink_parquet(OUTPUT_FILE_PATH)
    print(f" Cleaned data saved to: {OUTPUT_FILE_PATH}")

def create_sample_file():
    print(" Creating a 5% sample for quick experiments...")
    
    df = pl.read_parquet(OUTPUT_FILE_PATH)
    
    
    try:
        sample_df = df.sample(fraction=0.05, shuffle=True, seed=42)
        sample_df.write_parquet(SAMPLE_FILE_PATH)
        print(f" Sample saved to: {SAMPLE_FILE_PATH} (Rows: {sample_df.height})")
    except Exception as e:
        print(f" Could not sample automatically: {e}")

if __name__ == "__main__":
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    
    if os.path.exists(INPUT_FILE_PATH):
        process_single_csv()
        create_sample_file()
    else:
        print(f"ERROR: File not found at {INPUT_FILE_PATH}")
        