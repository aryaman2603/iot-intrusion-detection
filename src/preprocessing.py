import polars as pl
import os


RAW_CSV_NAME = "Stratified_data.csv" 


DATA_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
SAMPLE_DIR = "data/sample"


INPUT_FILE_PATH = os.path.join(DATA_DIR, RAW_CSV_NAME)
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, "ciciot2023_clean.parquet")
SAMPLE_FILE_PATH = os.path.join(SAMPLE_DIR, "sample_5_percent.parquet")

def process_single_csv():
    print(f"Connecting to raw file: {INPUT_FILE_PATH}")
    
    
    q = (
        pl.scan_csv(INPUT_FILE_PATH, infer_schema_length=10000)

        .rename({"Label": "label"})
        
        
        .with_columns(pl.all().exclude("label").cast(pl.Float32))
        
        
        .filter(pl.all_horizontal(pl.all().exclude("label").is_finite()))
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
        