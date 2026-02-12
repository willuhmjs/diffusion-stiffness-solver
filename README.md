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

## Quick Start

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Full Pipeline (Recommended):**
    This uses the default configuration for GenericAlloy/GenericAdhesive materials.
    ```bash
    python3 src/main.py pipeline
    ```

3.  **Run Inference on Experimental Data:**
    Predict stiffness for a specific specimen (e.g., Specimen 1, Location 1).
    ```bash
    python3 src/main.py infer --loc data/raw/Spec1_Loc1_Rep1.csv --ref data/raw/Spec1_Ref_Rep1.csv
    ```

    Or run batch inference on all files in `data/raw`:
    ```bash
    python3 src/main.py batch_inference
    ```

## User Guide

### 1. Input Data Formats

The solver accepts experimental data in **CSV** (`.csv`) or **DAT** (`.dat`) formats.

#### File Structure
Each file must contain frequency sweep data with the following information:
- **Frequency**: The frequency of the ultrasonic wave (typically in MHz).
- **Amplitude/Magnitude**: Signal strength (optional for phase inversion but good for checks).
- **Phase**: The phase angle (in radians or degrees).

#### Supported Column Names
 The system automatically maps common column headers to internal names. Ensure your files use one of the following conventions:

| Data Type | Supported Column Headers | Internal Name |
|-----------|--------------------------|---------------|
| **Frequency** | `Frequency`, `Freq(MHz)` | `Frequency` |
| **Amplitude** | `Amp`, `Mag`, `Magnitude` | `Amp` |
| **Phase** | `Phase`, `Phs`, `Phs(rad)` | `Phase` |

#### File Pair Requirements
Inference requires **two files** for each measurement location to cancel out system effects:
1. **Location File**: The measurement taken at the bond location (e.g., `Spec1_Loc1_Rep1.dat`).
2. **Reference File**: A measurement taken on a reference sample (e.g., `Spec1_Ref_Rep1.dat`).

### 2. Configuration

*   **`config/default.yaml`**: The main configuration file containing material properties (Substrate/Adhesive), frequency sweep ranges, and training hyperparameters.

## Project Structure

```
├── config/             # Configuration files (default.yaml)
├── data/               # Raw and Processed Data
├── results/            # Plots, Logs, and Evaluation Metrics
├── src/
│   ├── core/           # Core Logic (Physics, Model, Diffusion, Config)
│   ├── tasks/          # Executable Tasks (Train, Inference, Verify)
│   └── main.py         # Entry Point
```

## Physics Model

The forward model uses the Transfer Matrix Method for a Tri-Layer system (Substrate-Adhesive-Substrate).
*   **Inputs:** Frequencies, Stiffness ($K_{top}, K_{bottom}$), Adhesive Properties ($c, \alpha, \rho$).
*   **Outputs:** Reflection Phase Spectrum.
