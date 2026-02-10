import sys
import os

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

def main():
    parser = argparse.ArgumentParser(description="Diffusion Stiffness Solver CLI")
    parser.add_argument("--config", type=str, help="Path to config file", default=None)
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate Data
    subparsers.add_parser("generate", help="Generate synthetic training data")

    # Train Model
    subparsers.add_parser("train", help="Train the diffusion model")

    # Inference
    infer_parser = subparsers.add_parser("infer", help="Run inference on experimental data")
    infer_parser.add_argument("--loc", type=str, help="Path to Location CSV (optional, uses default if not provided)")
    infer_parser.add_argument("--ref", type=str, help="Path to Reference CSV (optional, uses default if not provided)")

    # Verify (Closed-Loop)
    subparsers.add_parser("verify", help="Run closed-loop verification on theoretical curves")

    # Uncertainty Analysis
    unc_parser = subparsers.add_parser("uncertainty", help="Run uncertainty analysis with multi-sampling")
    unc_parser.add_argument("--loc", type=str, help="Path to Location CSV")
    unc_parser.add_argument("--ref", type=str, help="Path to Reference CSV")
    unc_parser.add_argument("--samples", type=int, default=100, help="Number of samples (default: 100)")

    # Pipeline (All-in-One)
    subparsers.add_parser("pipeline", help="Run the full pipeline (Gen -> Train -> Infer -> Verify) and generate a report")

    args = parser.parse_args()

    # Set config path before importing modules that use it
    if args.config:
        os.environ['CONFIG_PATH'] = args.config

    # Lazy imports to ensure config is loaded correctly
    from src.tasks.generate import generate_data_task
    from src.tasks.train import train_model
    from src.tasks.inference import infer_task
    from src.tasks.verify import verify_task
    from src.tasks.uncertainty import uncertainty_task
    from src.tasks.pipeline import run_pipeline

    if args.command == "generate":
        generate_data_task()
    elif args.command == "train":
        train_model()
    elif args.command == "infer":
        infer_task(args.loc, args.ref)
    elif args.command == "uncertainty":
        uncertainty_task(args.loc, args.ref, args.samples)
    elif args.command == "verify":
        verify_task()
    elif args.command == "pipeline":
        run_pipeline()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()