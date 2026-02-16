import sys
import io
import contextlib
import time
import os
import re
import math
import matplotlib.pyplot as plt
import src.core.utils as utils
from src.tasks.generate import generate_data_task
from src.tasks.train import train_model
from src.tasks.visualize import visualize_training
from src.tasks.evaluate import evaluate_parameter_recovery
from src.tasks.inference import infer_task
from src.tasks.verify import verify_task
from src.tasks.uncertainty import uncertainty_task
from src.tasks.batch_inference import run_batch_inference

# We remove the capture_output context manager usage to allow live output (including progress bars)
# to be seen by the user.

def run_pipeline():
    print("\n" + "="*80)
    print("DIFFUSION STIFFNESS SOLVER - AUTOMATED PIPELINE REPORT")
    print("="*80 + "\n")
    
    report = {
        "generation": {"status": "PENDING", "duration": 0},
        "training": {"status": "PENDING", "duration": 0},
        "visualization": {"status": "PENDING", "duration": 0},
        "evaluation": {"status": "PENDING", "duration": 0},
        "inference": {"status": "PENDING", "duration": 0},
        "verification": {"status": "PENDING", "duration": 0},
        "uncertainty": {"status": "PENDING", "duration": 0},
        "batch_inference": {"status": "PENDING", "duration": 0}
    }

    start_total = time.time()

    # --- STEP 1: DATA GENERATION ---
    print("🔹 Running Task: DATA GENERATION...")
    t0 = time.time()
    try:
        generate_data_task()
        report["generation"]["status"] = "SUCCESS"
    except Exception as e:
        report["generation"]["status"] = "FAILED"
        print(f"Error: {e}")
    report["generation"]["duration"] = time.time() - t0
    print(f"   Status: {report['generation']['status']} ({report['generation']['duration']:.2f}s)")

    # --- STEP 2: TRAINING ---
    print("\n🔹 Running Task: MODEL TRAINING...")
    t0 = time.time()
    try:
        train_model() 
        report["training"]["status"] = "SUCCESS"
    except Exception as e:
        report["training"]["status"] = "FAILED"
        print(f"Error: {e}")
    report["training"]["duration"] = time.time() - t0
    print(f"   Status: {report['training']['status']} ({report['training']['duration']:.2f}s)")

    # --- STEP 3: VISUALIZATION ---
    print("\n🔹 Running Task: VISUALIZATION...")
    t0 = time.time()
    try:
        visualize_training()
        report["visualization"]["status"] = "SUCCESS"
    except Exception as e:
        report["visualization"]["status"] = "FAILED"
        print(f"Error: {e}")
    report["visualization"]["duration"] = time.time() - t0
    print(f"   Status: {report['visualization']['status']} ({report['visualization']['duration']:.2f}s)")

    # --- STEP 4: PARAMETER RECOVERY EVALUATION ---
    print("\n🔹 Running Task: PARAMETER RECOVERY EVALUATION...")
    t0 = time.time()
    try:
        evaluate_parameter_recovery()
        report["evaluation"]["status"] = "SUCCESS"
    except Exception as e:
        report["evaluation"]["status"] = "FAILED"
        print(f"Error: {e}")
    report["evaluation"]["duration"] = time.time() - t0
    print(f"   Status: {report['evaluation']['status']} ({report['evaluation']['duration']:.2f}s)")

    # --- STEP 5: INFERENCE (Default Spec4) ---
    print("\n🔹 Running Task: INFERENCE (Default Specimen)...")
    t0 = time.time()
    try:
        infer_task()
        report["inference"]["status"] = "SUCCESS"
    except Exception as e:
        report["inference"]["status"] = "FAILED"
        print(f"Error: {e}")
    report["inference"]["duration"] = time.time() - t0
    print(f"   Status: {report['inference']['status']} ({report['inference']['duration']:.2f}s)")

    # --- STEP 6: VERIFICATION ---
    print("\n🔹 Running Task: VERIFICATION...")
    t0 = time.time()
    try:
        verify_task()
        report["verification"]["status"] = "SUCCESS"
    except Exception as e:
        report["verification"]["status"] = "FAILED"
        print(f"Error: {e}")
    report["verification"]["duration"] = time.time() - t0
    print(f"   Status: {report['verification']['status']} ({report['verification']['duration']:.2f}s)")

    # --- STEP 7: UNCERTAINTY ANALYSIS ---
    print("\n🔹 Running Task: UNCERTAINTY ANALYSIS...")
    t0 = time.time()
    try:
        # Find all specimen files
        data_dir = 'data/raw'
        found_pairs = utils.find_specimen_pairs(data_dir)
        
        if not found_pairs:
            print("No matching Spec/Ref pairs found. Running default...")
            uncertainty_task(num_samples=50)
        else:
            num_plots = len(found_pairs)
            cols = 5
            rows = math.ceil(num_plots / cols)
            
            print(f"Found {num_plots} pairs. Running uncertainty analysis for all (Grid {rows}x{cols})...")
            
            # Create two summary figures
            fig_k, axes_k = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), squeeze=False)
            fig_p, axes_p = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), squeeze=False)
            
            axes_k_flat = axes_k.flatten()
            axes_p_flat = axes_p.flatten()
            
            for i, pair in enumerate(found_pairs):
                 loc = pair['loc_path']
                 ref = pair['ref_path']
                 print(f"[{i+1}/{num_plots}] Processing {os.path.basename(loc)}...")
                 
                 uncertainty_task(loc_path=loc, ref_path=ref, num_samples=50, ax_k=axes_k_flat[i], ax_phase=axes_p_flat[i])

            # Hide empty subplots
            for i in range(num_plots, len(axes_k_flat)):
                axes_k_flat[i].axis('off')
                axes_p_flat[i].axis('off')
            
            fig_k.tight_layout()
            fig_p.tight_layout()
            
            os.makedirs("results", exist_ok=True)
            fig_k.savefig('results/uncertainty_summary_k_dist.png')
            fig_p.savefig('results/uncertainty_summary_phase_fit.png')
            print("Saved summary plots to results/uncertainty_summary_*.png")
            
            plt.close(fig_k)
            plt.close(fig_p)

        report["uncertainty"]["status"] = "SUCCESS"
    except Exception as e:
        report["uncertainty"]["status"] = "FAILED"
        print(f"Error: {e}")
    report["uncertainty"]["duration"] = time.time() - t0
    print(f"   Status: {report['uncertainty']['status']} ({report['uncertainty']['duration']:.2f}s)")

    # --- STEP 8: BATCH INFERENCE ---
    print("\n🔹 Running Task: BATCH INFERENCE...")
    t0 = time.time()
    try:
        print("   Running Batch Inference on all specimens...")
        run_batch_inference()
        report["batch_inference"]["status"] = "SUCCESS"
    except Exception as e:
        report["batch_inference"]["status"] = "FAILED"
        print(f"Error: {e}")
    report["batch_inference"]["duration"] = time.time() - t0
    print(f"   Status: {report['batch_inference']['status']} ({report['batch_inference']['duration']:.2f}s)")

    total_duration = time.time() - start_total

    print("\n" + "="*80)
    print("📝  FINAL REPORT")
    print("="*80)
    
    print(f"\nTOTAL PIPELINE DURATION: {total_duration:.2f}s\n")
    
    for task, info in report.items():
        print(f"{task.upper():<20} | Status: {info['status']:<10} | Duration: {info['duration']:.2f}s")
    
    print("\n" + "="*80)
    print("END OF REPORT")
    print("="*80)

if __name__ == "__main__":
    run_pipeline()
