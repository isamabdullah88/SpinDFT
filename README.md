# High-Throughput Magnetic Exchange Pipeline for 2D $CrI_3$

Using high-throughput Density Functional Theory (DFT), Wannier90, and TB2J, this project systematically calculates the magnetic exchange interactions of 2D Chromium Triiodide (CrI₃) under applied mechanical strain to reveal how structural distortions modulate its Heisenberg parameters.

## 🚀 Overview

This repository contains a fully automated, Python-based orchestrator designed to run massive, high-accuracy First-Principles calculations on High-Performance Computing (HPC) clusters. It seamlessly links Quantum ESPRESSO, Wannier90, and TB2J to extract Goodenough-Kanamori exchange interactions ($J$) from atomic structures.

### Key Technical Features

- **End-to-End Automation**: Parses ASE databases, builds explicit pw.x and wannier90.x inputs, and executes MPI commands sequentially.

- **HPC Quota Survival System**: Features advanced disk-management for strict quotas (< 20 GB). Uses "Surgical Copying" of HDF5 charge densities and "Double Just-In-Time (JIT) Deletion" to instantly destroy massive 15+ GB wavefunction files the millisecond they are no longer needed.

- **Crash-Proof Diagonalization**: Automatically enforces Conjugate Gradient (cg) diagonalization during NSCF steps to prevent LAPACK segmentation faults when calculating large numbers of empty conduction bands.

- **Dual-Output Logging**: Custom logger that outputs clean progress to the terminal while silently recording deep physics and MPI tracebacks to a pipeline_execution.log file.

## ⚙️ The Computational Pipeline

The code executes the following rigorous sequence for every structural strain configuration:

- **Surgical SCF Injection**: Copies only the lightweight (~5MB) ground-state charge and spin density .hdf5 files from a pre-calculated SCF run, strictly ignoring heavy wavefunctions.

- **NSCF Calculation**: Computes the highly-excited empty bands (using disk_io = 'low' to write only compressed HDF5 data) over a dense $8 \times 8 \times 1$ K-mesh.

- **Wannier Extraction (pw2wannier90.x)**: Extracts overlap and projection matrices.

- **JIT Cleanup 1**: Violently deletes the massive pwscf.save/ directory to rescue HPC disk quota.

- **Wannierization (wannier90.x)**: Maximally localizes the Cr-$d$ and I-$p$ orbitals to generate the real-space Hamiltonian (_hr.dat).

- **JIT Cleanup 2**: Deletes the heavy .mmn, .amn, and .eig matrices.

- **Magnetic Exchange (TB2J)**: Computes the Heisenberg exchange parameters using the magnetic force theorem and outputs the final exchange.xml.

## 🛠️ Prerequisites & Dependencies

To run this pipeline, your HPC environment must have the following installed and accessible in the system $PATH:

- **Python 3.8+**

- **ASE (Atomic Simulation Environment)**

- **Quantum ESPRESSO** (v7.0+ recommended for HDF5 support)

- **Wannier90** (v3.0+)

- **TB2J** (Tight-Binding to J)

## 💻 Usage

**Configure the Environment**: Ensure your pseudopotentials are in the designated directory and your structural database (.db) is ready.

**Run the Orchestrator**:
```
python tb2j_runner.py --dbpath /path/to/structures.db --wkdir /scratch/user/tmp --prerelaxed_dir /path/to/scf_runs
```

**Monitor the Logs**:
You can watch the real-time pipeline execution safely via the generated log file:
```
tail -f pipeline_execution.log
```

## ⚠️ Important Note on K-point Meshes and Disk Space

Due to the strict relationship between K-point density and wavefunction file size, this pipeline is highly tuned for an $8 \times 8 \times 1$ mesh (64 k-points). This generates a peak disk footprint of ~12 GB, allowing it to run safely within a standard 20 GB HPC scratch quota. Using denser grids (e.g., $10 \times 10 \times 1$) will cause the intermediate .hdf5 files to exceed 18 GB, requiring a larger HPC disk allocation.

## 📜 License

This project is open-source and available under the MIT License.

## 🎓 About the Author

Isam Balghari MS Physics Candidate

Research Interests: Spintronics, Magnetic Materials, and Machine Learning Potentials.
