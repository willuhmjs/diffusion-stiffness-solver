# Diffusion Stiffness Solver

A Conditional Diffusion Model for solving the ultrasonic inverse problem in tri-layer adhesive structures. This framework maps swept-frequency phase spectra to interfacial stiffness parameters to detect bond degradation.

## Overview

This project implements a machine learning pipeline to solve the inverse problem for ultrasonic testing (UT) of adhesive bonds. Specifically, it infers the "interfacial stiffness" ($K$) of an Aluminum-Adhesive-Aluminum bond from frequency sweep phase measurements (3-8 MHz).

The core architecture uses a **Conditional Diffusion Model** (Conditioned on 1D Phase Curves) to estimate the posterior distribution of stiffness values given noisy spectral observations.

**Key Features:**
*   **Physics-Based Data Generation:** Uses the Transfer Matrix Method (TMM) to simulate realistic training data with configurable material properties (AA7075, FM209).
*   **Conditional Diffusion Model:** A 1D ResNet-based conditioning network guides a diffusion process to recover stiffness parameters.
*   **Weak Bond Detection:** Optimized to detect early-stage degradation ($10^{11} - 10^{13}$ N/m³).
*   **Automated Pipeline:** End-to-end script for data generation, training, evaluation, and inference.

## Project Structure

```
├── config/             # Configuration files (Physics, Training, Data)
├── data/               # Raw and Processed Data
├── results/            # Plots, Logs, and Evaluation Metrics
├── src/
│   ├── core/           # Core Logic (Physics, Model, Diffusion, Config)
│   ├── tasks/          # Executable Tasks (Train, Inference, Verify)
│   └── main.py         # Entry Point
```

## Quick Start

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Full Pipeline:**
    (Generates data, trains model, and evaluates)
    ```bash
    python3 src/main.py pipeline
    ```

3.  **Run Inference on Specific File:**
    ```bash
    python3 src/main.py inference --file data/raw/Spec1_Loc3_Rep1.csv
    ```

## Physics Model

The forward model uses the Transfer Matrix Method for a Tri-Layer system (Substrate-Adhesive-Substrate).
*   **Inputs:** Frequencies, Stiffness ($K_{top}, K_{bottom}$), Adhesive Properties ($c, \alpha, \rho$).
*   **Outputs:** Reflection Phase Spectrum.
