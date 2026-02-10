import os
import pandas as pd
import glob

def import_data():
    source_dir = "data/raw"
    target_dir = "data/raw"
    
    # Get all .dat files
    files = glob.glob(os.path.join(source_dir, "*.dat"))
    
    print(f"Found {len(files)} files in {source_dir}")
    
    for file_path in files:
        filename = os.path.basename(file_path)
        name_no_ext = os.path.splitext(filename)[0]
        target_path = os.path.join(target_dir, name_no_ext + ".csv")
        
        print(f"Processing {filename} -> {target_path}...")
        
        try:
            # Read whitespace-delimited file
            # Skip the first line if it's just headers not matching pandas expectation or just let pandas handle it
            # The file has: Frequency Int1 Int2 Mag Phs(rad)
            df = pd.read_csv(file_path, sep=r'\s+', engine='python')
            
            # Select relevant columns
            # Frequency is already Frequency
            # Mag -> Amp
            # Phs(rad) -> Phase
            
            # Check column names
            # Using clean mapping
            rename_map = {
                'Frequency': 'Frequency',
                'Mag': 'Amp',
                'Phs(rad)': 'Phase'
            }
            
            # Filter and rename
            df_out = df[['Frequency', 'Mag', 'Phs(rad)']].rename(columns=rename_map)
            
            # Save to CSV
            df_out.to_csv(target_path, index=False)
            print(f"  Saved {len(df_out)} rows.")
            
        except Exception as e:
            print(f"  Error processing {filename}: {e}")

if __name__ == "__main__":
    import_data()
