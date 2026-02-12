import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Constants
RESULTS_PATH = 'results/batch_inference_results.csv'
FRACTURE_PATH = 'data/metadata/fracture_toughness.csv'
OUTPUT_PLOT_PATH = 'results/predicted_k_vs_fracture_toughness.png'
OUTPUT_CSV_PATH = 'results/fracture_energy_analysis.csv'


def load_data():
    """Loads the required datasets."""
    for path in [RESULTS_PATH, FRACTURE_PATH]:
        if not os.path.exists(path):
            print(f"Error: File not found at {path}")
            return None, None

    try:
        inference_df = pd.read_csv(RESULTS_PATH)
        fracture_df = pd.read_csv(FRACTURE_PATH)
        return inference_df, fracture_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def parse_specimen(val):
    """Parses specimen ID, handling 'Spec1' format."""
    if isinstance(val, str) and val.startswith('Spec'):
        return int(val.replace('Spec', ''))
    try:
        return int(val)
    except (ValueError, TypeError):
        return val


def process_data(inference_df, fracture_df):
    """Merges inference results with fracture toughness data."""
    inference_df['Specimen_ID'] = inference_df['Specimen'].apply(parse_specimen)
    
    # Merge with Fracture Toughness (G1c)
    merged_df = pd.merge(
        inference_df, 
        fracture_df[['Specimen_ID', 'Avg_G1C', 'StDev_G1C']], 
        on='Specimen_ID', 
        how='left'
    )
    
    merged_df = merged_df.dropna(subset=['Avg_G1C'])
    
    # Aggregate by specimen (median K per specimen to reject outlier predictions)
    specimen_summary = merged_df.groupby('Specimen_ID').agg(
        Mean_K=('Predicted_K', 'median'),
        Std_K=('Predicted_K', lambda x: np.percentile(x, 75) - np.percentile(x, 25)),
        Avg_G1C=('Avg_G1C', 'first'),
        StDev_G1C=('StDev_G1C', 'first')
    ).reset_index()
    
    return merged_df, specimen_summary


def plot_results(specimen_summary, output_path):
    """Generates correlation plot between G1c and Predicted K."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scale K by 10^14 for display
    scale_factor = 1e14
    y_scaled = specimen_summary['Mean_K'] / scale_factor
    y_err_scaled = specimen_summary['Std_K'] / scale_factor
    
    # Plot with error bars
    ax.errorbar(
        specimen_summary['Avg_G1C'], 
        y_scaled,
        xerr=specimen_summary['StDev_G1C'],
        yerr=y_err_scaled,
        fmt='o', 
        capsize=5, 
        markersize=8,
        color='#2E5090',
        ecolor='#2E5090', 
        elinewidth=1, 
        capthick=1
    )
    
    # Linear regression
    x = specimen_summary['Avg_G1C'].values
    y = y_scaled.values
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs
    
    # R² calculation
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Fit line
    x_fit = np.linspace(0, 2500, 100)
    y_fit = slope * x_fit + intercept
    
    ax.plot(x_fit, y_fit, ':', color='#2E5090', linewidth=1.5)
    
    # Add equation and R²
    eq_text = f'y = {slope:.4f}x + {intercept:.3f}\nR² = {r_squared:.4f}'
    ax.text(0.5, 0.35, eq_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top')
    
    # Axis settings
    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 10)
    ax.set_xlabel(r'$G_{IC}$, J/m²', fontsize=12)
    ax.set_ylabel(r'Interfacial Stiffness N/m³ ($\times 10^{14}$)', fontsize=12)
    ax.grid(True, which="major", ls="-", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


def plot_fracture_energy_task():
    print("Executing fracture energy plotting task...")
    
    inference_df, fracture_df = load_data()
    if inference_df is None:
        return

    merged_df, specimen_summary = process_data(inference_df, fracture_df)
    
    if merged_df.empty:
        print("No valid data points after merging.")
        return

    print("\nSpecimen Summary:")
    print(specimen_summary.to_string(index=False))
    
    plot_results(specimen_summary, OUTPUT_PLOT_PATH)
    
    merged_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nAnalysis data saved to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    plot_fracture_energy_task()
