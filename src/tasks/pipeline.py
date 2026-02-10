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

@contextlib.contextmanager
def capture_output():
    new_out, new_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err

def run_pipeline():
    print("\n" + "="*80)
    print("🚀  DIFFUSION STIFFNESS SOLVER - AUTOMATED PIPELINE REPORT  🚀")
    print("="*80 + "\n")
    
    report = {
        "generation": {"status": "PENDING", "output": "", "duration": 0},
        "training": {"status": "PENDING", "output": "", "duration": 0},
        "visualization": {"status": "PENDING", "output": "", "duration": 0},
        "evaluation": {"status": "PENDING", "output": "", "duration": 0},
        "inference": {"status": "PENDING", "output": "", "duration": 0},
        "verification": {"status": "PENDING", "output": "", "duration": 0},
        "uncertainty": {"status": "PENDING", "output": "", "duration": 0}
    }

    start_total = time.time()

    # --- STEP 1: DATA GENERATION ---
    print("🔹 Running Task: DATA GENERATION...")
    t0 = time.time()
    try:
        with capture_output() as (out, err):
            generate_data_task()
        report["generation"]["status"] = "SUCCESS"
        report["generation"]["output"] = out.getvalue()
    except Exception as e:
        report["generation"]["status"] = "FAILED"
        report["generation"]["output"] = str(e)
    report["generation"]["duration"] = time.time() - t0
    print(f"   Status: {report['generation']['status']} ({report['generation']['duration']:.2f}s)")

    # --- STEP 2: TRAINING ---
    print("🔹 Running Task: MODEL TRAINING...")
    t0 = time.time()
    try:
        with capture_output() as (out, err):
            train_model() 
        report["training"]["status"] = "SUCCESS"
        full_log = out.getvalue()
        report["training"]["output"] = "..." + full_log[-1000:] if len(full_log) > 1000 else full_log
    except Exception as e:
        report["training"]["status"] = "FAILED"
        report["training"]["output"] = str(e)
    report["training"]["duration"] = time.time() - t0
    print(f"   Status: {report['training']['status']} ({report['training']['duration']:.2f}s)")

    # --- STEP 3: VISUALIZATION ---
    print("🔹 Running Task: VISUALIZATION...")
    t0 = time.time()
    try:
        with capture_output() as (out, err):
            visualize_training()
        report["visualization"]["status"] = "SUCCESS"
        report["visualization"]["output"] = out.getvalue()
    except Exception as e:
        report["visualization"]["status"] = "FAILED"
        report["visualization"]["output"] = str(e)
    report["visualization"]["duration"] = time.time() - t0
    print(f"   Status: {report['visualization']['status']} ({report['visualization']['duration']:.2f}s)")

    # --- STEP 4: PARAMETER RECOVERY EVALUATION ---
    print("🔹 Running Task: PARAMETER RECOVERY EVALUATION...")
    t0 = time.time()
    eval_metrics = None
    try:
        with capture_output() as (out, err):
            eval_metrics = evaluate_parameter_recovery()
        report["evaluation"]["status"] = "SUCCESS"
        report["evaluation"]["output"] = out.getvalue()
    except Exception as e:
        report["evaluation"]["status"] = "FAILED"
        report["evaluation"]["output"] = str(e)
    report["evaluation"]["duration"] = time.time() - t0
    print(f"   Status: {report['evaluation']['status']} ({report['evaluation']['duration']:.2f}s)")

    # --- STEP 5: INFERENCE (Default Spec4) ---
    print("🔹 Running Task: INFERENCE (Default Specimen)...")
    t0 = time.time()
    try:
        with capture_output() as (out, err):
            infer_task()
        report["inference"]["status"] = "SUCCESS"
        report["inference"]["output"] = out.getvalue()
    except Exception as e:
        report["inference"]["status"] = "FAILED"
        report["inference"]["output"] = str(e)
    report["inference"]["duration"] = time.time() - t0
    print(f"   Status: {report['inference']['status']} ({report['inference']['duration']:.2f}s)")

    # --- STEP 6: VERIFICATION ---
    print("🔹 Running Task: VERIFICATION...")
    t0 = time.time()
    try:
        with capture_output() as (out, err):
            verify_task()
        report["verification"]["status"] = "SUCCESS"
        report["verification"]["output"] = out.getvalue()
    except Exception as e:
        report["verification"]["status"] = "FAILED"
        report["verification"]["output"] = str(e)
    report["verification"]["duration"] = time.time() - t0
    print(f"   Status: {report['verification']['status']} ({report['verification']['duration']:.2f}s)")

    # --- STEP 7: UNCERTAINTY ANALYSIS ---
    print("🔹 Running Task: UNCERTAINTY ANALYSIS...")
    t0 = time.time()
    try:
        with capture_output() as (out, err):
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

        report["uncertainty"] = {"status": "SUCCESS", "output": out.getvalue()}
    except Exception as e:
        report["uncertainty"] = {"status": "FAILED", "output": str(e)}
    report["uncertainty"]["duration"] = time.time() - t0
    print(f"   Status: {report['uncertainty']['status']} ({report['uncertainty']['duration']:.2f}s)")

    # --- STEP 8: BATCH INFERENCE ---
    print("🔹 Running Task: BATCH INFERENCE...")
    
    try:
        # Run batch inference to ensure results exist
        # We suppress output unless error to keep pipeline clean
        # But run_batch_inference prints a lot. Let's capture it.
        print("   Running Batch Inference on all specimens...")
        with capture_output() as (out, err):
            run_batch_inference()
        
    except Exception as e:
        print(f"   Status: FAILED ({e})")

    total_duration = time.time() - start_total

    print("\n" + "="*80)
    print("📝  FINAL REPORT")
    print("="*80)
    
    print(f"\nTOTAL PIPELINE DURATION: {total_duration:.2f}s\n")
    
    print("--- SECTION 1: DATA GENERATION ---")
    print(f"STATUS: {report['generation']['status']}")
    print("LOGS:")
    print(report['generation']['output'].strip())
    print("-" * 40)

    print("\n--- SECTION 2: MODEL TRAINING ---")
    print(f"STATUS: {report['training']['status']}")
    print("LOGS (Last 1000 chars):")
    print(report['training']['output'].strip())
    print("-" * 40)
    
    print("\n--- SECTION 3: VISUALIZATION ---")
    print(f"STATUS: {report['visualization']['status']}")
    print("LOGS:")
    print(report['visualization']['output'].strip())
    print("-" * 40)

    print("\n--- SECTION 4: PARAMETER RECOVERY ---")
    print(f"STATUS: {report['evaluation']['status']}")
    if eval_metrics:
        print(f"RMSE (Log Scale): {eval_metrics.get('rmse_log', 'N/A')}")
        print(f"Weak Bond RMSE:   {eval_metrics.get('weak_rmse_log', 'N/A')}")
    print("LOGS:")
    print(report['evaluation']['output'].strip())
    print("-" * 40)

    print("\n--- SECTION 5: INFERENCE RESULTS ---")
    print(f"STATUS: {report['inference']['status']}")
    print("LOGS:")
    print(report['inference']['output'].strip())
    print("-" * 40)

    print("\n--- SECTION 6: VERIFICATION & PHYSICS CHECK ---")
    print(f"STATUS: {report['verification']['status']}")
    print("LOGS:")
    print(report['verification']['output'].strip())
    print("-" * 40)

    print("\n--- SECTION 7: UNCERTAINTY ANALYSIS ---")
    print(f"STATUS: {report['uncertainty']['status']}")
    print("LOGS:")
    print(report['uncertainty']['output'].strip())
    print("-" * 40)

    print("\n" + "="*80)
    print("END OF REPORT")
    print("="*80)

if __name__ == "__main__":
    run_pipeline()
