# SpinDFT: High-Throughput Magnetic Exchange Pipeline for 2D CrI₃

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Quantum ESPRESSO](https://img.shields.io/badge/Quantum_ESPRESSO-v7.0+-003366.svg)](https://www.quantum-espresso.org/)
[![Wannier90](https://img.shields.io/badge/Wannier90-v3.0+-orange.svg)](https://wannier.org/)
[![TB2J](https://img.shields.io/badge/TB2J-latest-green.svg)](https://tb2j.readthedocs.io/)

**SpinDFT** is a fully automated, Python-based high-throughput orchestrator for running large-scale first-principles calculations on HPC clusters. It systematically computes the strain-dependent magnetic exchange interactions of monolayer CrI₃ using a rigorous DFT+U → Wannier90 → TB2J pipeline, generating the training dataset that underpins the [DSpinGNN](https://github.com/isamabdullah88/DSpinGNN) machine learning framework.

---

## Overview

Understanding how localized structural distortions modulate Heisenberg superexchange parameters in 2D van der Waals magnets requires quantum mechanical accuracy across hundreds of structural configurations — a task that demands careful automation, robust error handling, and efficient storage management. SpinDFT addresses all three, enabling systematic sampling of the full biaxial, uniaxial, and shear strain phase space of monolayer CrI₃ with minimal manual intervention.

Each run produces per-configuration exchange couplings J(r), total energies, and atomic forces that are consumed directly by DSpinGNN for physics-informed machine learning potential training.

---

## Key Features

**End-to-End Automation**
Parses ASE structural databases, programmatically constructs `pw.x` and `wannier90.x` input files, and executes the full six-step MPI pipeline sequentially across hundreds of strained configurations without manual input.

**Dynamic Fermi Energy Calibration**
Automatically parses and tracks the exact Fermi level from the SCF output to determine the precise number of empty conduction bands required for the NSCF step. This eliminates band over-runs, under-runs, and the "empty shells" errors that commonly crash Wannierization on novel systems.

**Crash-Proof Diagonalization**
Enforces Conjugate Gradient (`cg`) diagonalization during NSCF steps to prevent LAPACK segmentation faults when computing large arrays of empty conduction bands — a known failure mode with the default Davidson algorithm at high band counts.

**Intelligent Storage Management**
Implements Just-In-Time (JIT) deletion of wavefunction and charge density files immediately after each step consumes them. This enables high-throughput DFT over dense k-meshes to operate safely within HPC disk quota constraints that would otherwise be exceeded within the first few dozen configurations.

**Dual-Output Logging**
A custom logger writes clean, high-level progress to the terminal while simultaneously recording full physics parameters, convergence metrics, timing data, and MPI tracebacks to a timestamped `SpinDFT_<date>_<time>.log` file for post-hoc diagnostics.

---

## Physics Pipeline

The orchestrator executes the following sequence for every structural configuration in the database:

```
Structure (ASE .db)
       │
       ▼
1. Structural Relaxation   ←  variable-cell or fixed-cell (6×6×1 k-mesh)
       │
       ▼
2. SCF Calculation         ←  ground-state charge/spin density, DFT+U (10×10×1)
       │
       ▼
3. NSCF Calculation        ←  empty bands for Wannierization (36×36×1)
       │
       ▼
4. pw2wannier90.x          ←  overlap (M) and projection (A) matrices
       │
       ▼
5. Wannier90               ←  maximally-localized Cr-d / I-p Wannier functions
       │
       ▼
6. TB2J                    ←  Heisenberg J(r) via magnetic force theorem
       │
       ▼
   exchange.xml  +  energies  +  forces   →   DSpinGNN training dataset
```

**Step details:**

1. **Structural Relaxation** — Relaxes the strained cell to the local energy minimum. Variable-cell relaxation is used for the pristine structure; fixed-cell relaxation is applied to strained configurations where the strain is prescribed.

2. **SCF** — Converges the ground-state charge and spin density using GGA-PBE with a Hubbard U = 3.0 eV on Cr 3d orbitals (`DFT+U`, Dudarev scheme) and Grimme D3 van der Waals corrections. Produces the reference Fermi energy.

3. **NSCF** — Computes a large number of unoccupied bands over a dense k-mesh. Band count is set dynamically from the Fermi level parsed in step 2.

4. **pw2wannier90** — Extracts Bloch overlaps and projections onto Cr-d and I-p atomic orbitals.

5. **Wannier90** — Minimizes the spread of the Wannier functions to produce a real-space tight-binding Hamiltonian (`seedname_hr.dat`).

6. **TB2J** — Applies the Liechtenstein magnetic force theorem to the Wannier Hamiltonian on a dense k-mesh (36×36×1) to extract isotropic nearest-neighbour exchange couplings J₁(r) and write `exchange.xml`.

---

## Output Dataset

For each configuration, SpinDFT produces:

| Output | Description |
|---|---|
| `exchange.xml` | TB2J exchange coupling file; J₁ (meV) per Cr–Cr bond |
| SCF total energy | E (Ry), parsed and stored in the ASE database |
| Atomic forces | F (Ry/Bohr), parsed from SCF output |
| Relaxed structure | Updated ASE `.db` entry with relaxed coordinates |
| `SpinDFT_*.log` | Full run log with convergence metrics and timings |

The complete dataset (345 training + 61 validation + 61 test configurations) is publicly available at the repository linked below.

---

## Prerequisites

The following must be installed and accessible in your HPC `$PATH`:

| Dependency | Version | Purpose |
|---|---|---|
| Python | 3.8+ | Orchestration and parsing |
| ASE | latest | Structure database and I/O |
| Quantum ESPRESSO | 7.0+ | DFT relaxation, SCF, NSCF |
| Wannier90 | 3.0+ | Wannier function generation |
| TB2J | latest | Exchange coupling extraction |
| SSSP pseudopotentials | Efficiency v1.3.0 | Cr and I pseudopotentials |

Quantum ESPRESSO v7.0 or later is strongly recommended for HDF5 wavefunction support, which reduces I/O overhead on large NSCF calculations.

---

## Installation

```bash
git clone https://github.com/isamabdullah88/SpinDFT.git
cd SpinDFT
pip install -r requirements.txt
```

Ensure your pseudopotential directory (e.g., SSSP Efficiency) is set in the configuration file before running.

---

## Usage

### 1. Prepare Your Structural Database

Generate or provide an ASE `.db` file containing the initial structural configurations. Strained structures can be generated using the included strain-sampling utility:

```bash
python generate_structures.py --pristine CrI3_pristine.cif --strains biaxial uniaxial shear --range -0.05 0.05 --steps 11
```

### 2. Configure the Environment

Edit `config.yaml` to set your pseudopotential path, HPC scratch directory, MPI executable, and convergence parameters:

```yaml
pseudo_dir: /home/user/pseudopotentials/sssp_efficiency
scratch_dir: /scratch/user/spindft_runs
mpi_cmd: mpirun -np 32
hubbard_u: 3.0   # eV, applied to Cr d-orbitals
```

### 3. Run the Orchestrator

```bash
python run.py \
  --dbpath /path/to/structures.db \
  --wkdir /scratch/user/tmp \
  --prerelaxed_dir /path/to/scf_runs
```

### 4. Monitor Execution

```bash
tail -f SpinDFTLogs/SpinDFT_<date>_<time>.log
```

The terminal shows high-level progress (configuration index, current step, convergence status). The log file contains the full record including Fermi energies, band counts, Wannier spreads, and any MPI errors.

### 5. Collect Outputs

Completed exchange couplings, energies, and forces are written back into the ASE database and as individual `exchange.xml` files in each configuration's working directory.

---

## Project Context

SpinDFT is the data-generation backbone of a two-repository research project:

| Repository | Role |
|---|---|
| **SpinDFT** (this repo) | High-throughput DFT pipeline; generates the labelled dataset |
| [**DSpinGNN**](https://github.com/isamabdullah88/DSpinGNN) | Physics-informed GNN; trained on the SpinDFT dataset for mesoscale simulation |

The full methodology and results are described in:

> Isam Abdullah Balghari, Muhammad Faryad, Muhammad Sabieh Anwar,
> *"DSpinGNN: A Physics-Informed Equivariant Graph Neural Network for Dynamic
> Magnetic Exchange Prediction in Strain-Deformed Monolayer CrI₃,"*
> submitted to Physical Review Materials (2026).

---

## Citation

If you use SpinDFT in your research, please cite the accompanying paper:

```bibtex
@article{balghari2026dspingnn,
  author  = {Isam Abdullah Balghari and Muhammad Faryad and Muhammad Sabieh Anwar},
  title   = {{DSpinGNN}: A Physics-Informed Equivariant Graph Neural Network
             for Dynamic Magnetic Exchange Prediction in Strain-Deformed
             Monolayer {CrI}$_3$},
  journal = {Physical Review Materials},
  year    = {2026},
  note    = {Submitted},
}
```

---

## Authors

**Isam Abdullah Balghari**
Department of Physics, LUMS
✉ isamabdullah88@gmail.com

**Supervisor: Dr. Muhammad Sabieh Anwar**
Department of Physics, LUMS
[Personal](https://physlab.org/muhammad-sabieh-anwar-personal/)

**Co-supervisor: Dr. Muhammad Faryad**
Department of Physics, LUMS
[Personal](https://lums.edu.pk/lums_employee/4010)

---

## License

This project is licensed under the GNU General Public License v3.0.
See [LICENSE](LICENSE) for details.
