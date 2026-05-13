# SpinDFT: High-Throughput Magnetic Exchange Pipeline for 2D $CrI_3$

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Quantum ESPRESSO](https://img.shields.io/badge/Quantum_ESPRESSO-v7.0+-003366.svg)](https://www.quantum-espresso.org/)

This repository contains **SpinDFT**, a fully automated, Python-based orchestrator designed to run massive, high-accuracy first-principles calculations on High-Performance Computing (HPC) clusters. 

Using high-throughput Density Functional Theory (DFT), Wannier90, and TB2J, this project systematically calculates the magnetic exchange interactions of 2D Chromium Triiodide ($CrI_3$) under applied mechanical strain. It serves as the automated data-generation backbone for training machine learning potentials (like [DSpinGNN](https://github.com/isamabdullah88/DSpinGNN.git)) by revealing how localized structural distortions modulate Heisenberg super-exchange parameters.

---

## Key Technical Features

* **End-to-End Automation**: Parses ASE databases, programmatically builds explicit `pw.x` and `wannier90.x` inputs, and executes MPI commands sequentially across hundreds of strained configurations.
* **Dynamic Fermi Energy Calibration**: Automatically parses and tracks the exact Fermi level from the initial SCF output to dynamically calculate the precise number of empty conduction bands required for the NSCF step. This strictly prevents band over-runs, under-runs, and fatal "empty shells" errors during Wannierization.
* **Crash-Proof Diagonalization**: Automatically enforces Conjugate Gradient (`cg`) diagonalization during NSCF steps to prevent LAPACK segmentation faults when calculating massive arrays of empty conduction bands.
* **Intelligent Storage Management**: Implements iterative, Just-In-Time (JIT) deletion protocols to manage massive wavefunction and charge density files. This allows high-throughput DFT calculations over dense k-meshes to run safely within strict HPC disk-quota environments.
* **Dual-Output Logging**: Features a custom logger that outputs clean, high-level progress to the terminal while silently recording deep physics parameters, convergence metrics, and MPI tracebacks to a comprehensive `SpinDFT_<date>_<time>.log` file.
---

## The Full Physics Pipeline

The orchestrator executes the following rigorous computational sequence for every structural strain configuration in the database:

1.  **Structural Relaxation (Variable-Cell / Fixed-Cell)**: Relaxes the initial pristine or strained lattice using a standard k-mesh (e.g., $6 \times 6 \times 1$) to find the local thermodynamic minimum.
2.  **Self-Consistent Field (SCF)**: Calculates the converged ground-state charge and spin density matrices on a dense k-mesh (e.g., $10 \times 10 \times 1$) with a Hubbard $U$ correction to account for localized Cr $3d$-orbitals.
3.  **Non-Self-Consistent Field (NSCF)**: Computes the highly-excited empty bands over an ultra-dense K-mesh (e.g., $36 \times 36 \times 1$) required for accurate Wannierization.
4.  **Wannier Extraction (`pw2wannier90.x`)**: Extracts the overlap and projection matrices from the NSCF wavefunctions.
5.  **Wannierization (`wannier90.x`)**: Maximally localizes the Cr-$d$ and I-$p$ orbitals to generate the highly accurate real-space tight-binding Hamiltonian (`_hr.dat`).
6.  **Magnetic Exchange (`TB2J`)**: Computes the localized Heisenberg exchange parameters ($J$) using the magnetic force theorem and outputs the final, machine-readable `exchange.xml`.

---

## Prerequisites & Dependencies

To run this pipeline, your HPC environment must have the following installed and accessible in the system `$PATH`:

* **Python 3.8+**
* **ASE (Atomic Simulation Environment)**
* **Quantum ESPRESSO** (v7.0+ strongly recommended for HDF5 support)
* **Wannier90** (v3.0+)
* **TB2J** (Tight-Binding to J)

---

## Usage

**1. Configure the Environment** Ensure your pseudopotentials (e.g., SSSP Efficiency) are located in the designated directory and your initial structural database (`.db`) is generated.

**2. Run the Orchestrator**
Execute the main python script, pointing it toward output database path and desired scratch/working directory:
```bash
python run.py --dbpath /path/to/structures.db --wkdir /scratch/user/tmp --prerelaxed_dir /path/to/scf_runs
```

**3. Monitor Execution**
You can safely watch the real-time pipeline execution, including iteration-level convergence details, via the generated log file:
```bash
tail -f SpinDFTLogs/SpinDFT_<date>_<time>.log
```

--

## License
This project is open-source and available under the MIT License.

--

**Author**: Isam Balghari

**Institution**: Lahore University of Management Sciences (LUMS)

**Degree**: MS Physics

Research Focus: Spintronics, Strain Engineering, and Machine Learning Potentials
