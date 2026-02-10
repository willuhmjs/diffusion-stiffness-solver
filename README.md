# Diffusion Stiffness Solver

A Conditional Diffusion Model for solving the ultrasonic inverse problem in tri-layer adhesive structures. This framework maps swept-frequency phase spectra to interfacial stiffness parameters to detect bond degradation.

## Overview

This project implements a machine learning pipeline to solve the inverse problem for ultrasonic testing (UT) of adhesive bonds. Specifically, it infers the "interfacial stiffness" ($K$) of an Aluminum-Adhesive-Aluminum bond from frequency sweep phase measurements (3-8 MHz).

The core architecture uses a **Conditional Diffusion Model** (Conditioned on 1D Phase Curves) to estimate the posterior distribution of stiffness values given noisy spectral observations.

**Key Features:**
*   **Physics-Based Data Generation:** Uses the Transfer Matrix Method (TMM) to simulate realistic training data with configurable material properties.
*   **Conditional Diffusion Model:** A 1D ResNet-based conditioning network guides a diffusion process to recover stiffness parameters.
*   **Weak Bond Detection:** Optimized to detect early-stage degradation ($10^{11} - 10^{13}$ N/m³).
*   **Automated Pipeline:** End-to-end script for data generation, training, evaluation, and inference.

## Project Structure

```
├── config/             # Configuration files (accurate.yaml, default.yaml)
├── data/               # Raw and Processed Data
├── results/            # Plots, Logs, and Evaluation Metrics
├── src/
│   ├── core/           # Core Logic (Physics, Model, Diffusion, Config)
│   ├── tasks/          # Executable Tasks (Train, Inference, Verify)
│   └── main.py         # Entry Point
├── NEXT_MODEL_CONTEXT.md # Handoff instructions for next steps
```

## Quick Start

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Full Pipeline (Recommended):**
    This uses the accurate configuration for GenericAlloy/GenericAdhesive materials.
    ```bash
    python3 src/main.py --config config/accurate.yaml pipeline
    ```

3.  **Run Inference on Experimental Data:**
    Predict stiffness for a specific specimen (e.g., Specimen 1, Location 1).
    ```bash
    python3 src/main.py --config config/accurate.yaml infer --loc data/raw/Spec1_Loc1_Rep1.csv --ref data/raw/Spec1_Ref_Rep1.csv
    ```

## Configuration

*   **`config/default.yaml`**: Generic Aluminum/Epoxy parameters.
*   **`config/accurate.yaml`**: Specific parameters for **GenericAlloy Substrate** and **Generic Adhesive Adhesive** (based on nominal values). **Use this for current experiments.**

## Physics Model

The forward model uses the Transfer Matrix Method for a Tri-Layer system (Substrate-Adhesive-Substrate).
*   **Inputs:** Frequencies, Stiffness ($K_{top}, K_{bottom}$), Adhesive Properties ($c, \alpha, \rho$).
*   **Outputs:** Reflection Phase Spectrum.

## Next Steps (Missing Labels)

We are currently waiting for ground truth stiffness values for the experimental data in `data/raw`. 
See **`NEXT_MODEL_CONTEXT.md`** for detailed instructions on how to proceed once those labels are available.
