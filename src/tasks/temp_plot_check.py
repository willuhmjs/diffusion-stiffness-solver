import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def plot_predicted_vs_true_k():
    print("Loading data...")
    # Load data
    try:
        results_df = pd.read_csv('results/batch_inference_results.csv')
        fracture_df = pd.read_csv('data/metadata/fracture_toughness.csv')
        properties_df = pd.read_csv('data/metadata/specimen_properties.csv')
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return

    # Prepare lists to store matched data
    predicted_ks = []
    gc_values = []
    specimen_labels = []
    thicknesses = []

    print(f"Processing {len(results_df)} inference results...")

    for index, row in results_df.iterrows():
        # The CSV has 'Specimen', 'Location' columns directly now (based on `batch_inference.py` output)
        if 'Specimen' in row and 'Location' in row:
             # Specimen column might be "Spec1" or just "1"
             spec_val = str(row['Specimen'])
             if "Spec" in spec_val:
                 spec_num = int(spec_val.replace("Spec", ""))
             else:
                 spec_num = int(spec_val)
             
             loc_num = int(row['Location'])
        elif 'Filename' in row:
            filename = row['Filename']
            # Parse Specimen and Location from filename (e.g., Spec3_Loc1_Rep1.csv)
            match = re.search(r"Spec(\d+)_Loc(\d+)", filename, re.IGNORECASE)
            if not match:
                continue
            spec_num = int(match.group(1))
            loc_num = int(match.group(2))
        else:
            print(f"Skipping row {index}: Could not identify Specimen/Location")
            continue

        k_pred = row['Predicted_K']
        
        # 1. Look up Fracture Toughness (G_c)
        # Check if column is 'Specimen' or 'Specimen_ID'
        spec_col = 'Specimen_ID' if 'Specimen_ID' in fracture_df.columns else 'Specimen'
        g_row = fracture_df[fracture_df[spec_col] == spec_num]
        
        if g_row.empty:
            continue
            
        # Column is likely 'Avg_G1C'
        if 'Avg_G1C' in g_row:
             g_c = g_row.iloc[0]['Avg_G1C']
        else:
             # Fallback
             try:
                col = [c for c in fracture_df.columns if "G1C" in c or "Energy" in c][0]
                g_c = g_row.iloc[0][col]
             except:
                 print(f"Skipping Spec{spec_num}: Could not find G1C column")
                 continue
        
        # 2. Look up Thickness (h)
        h_row = properties_df[(properties_df['Specimen'] == spec_num) & (properties_df['Location'] == loc_num)]
        if h_row.empty:
            print(f"Skipping Spec{spec_num} Loc{loc_num}: No thickness data.")
            continue
        
        h_val = h_row.iloc[0]['Thickness, m']
        
        predicted_ks.append(k_pred)
        gc_values.append(g_c)
        specimen_labels.append(f"S{spec_num}")
        thicknesses.append(h_val)

    # Plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(gc_values, predicted_ks, c=thicknesses, cmap='viridis', s=100, alpha=0.8, edgecolors='k')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Bondline Thickness (m)')
    
    # Label points
    # Jitter slightly to avoid overlap
    seen_coords = {}
    for i, txt in enumerate(specimen_labels):
        x = gc_values[i]
        y = predicted_ks[i]
        
        # Simple jitter
        if (x,y) in seen_coords:
            y *= 1.1 # Shift up slightly
        seen_coords[(x,y)] = True
        
        plt.annotate(txt, (x, y), fontsize=8, alpha=0.7)
        
    plt.xlabel('Fracture Energy Gc (J/m^2)')
    plt.ylabel('Predicted Stiffness K (N/m^3)')
    plt.yscale('log')
    plt.xscale('log')
    plt.title(f'Predicted Stiffness vs Fracture Energy\n(Color = Thickness used in Inference)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    out_path = 'results/predicted_k_vs_fracture_toughness.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    plot_predicted_vs_true_k()
