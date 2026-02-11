# MS Thesis: Machine Learning Enabled Spin-Lattice Dynamics for 2D Materials

## 🔬 Project Overview

This repository contains the implementation and research for my MS Physics thesis. The project focuses on determining the renormalized Curie temperature ($T_c$) in 2D magnetic materials by integrating Spin-Lattice Dynamics (SLD) with advanced Machine Learning (ML) potentials.

The core challenge addressed here is the coupling between the crystal lattice (vibrations/strain) and the magnetic subsystem (spin orientations). By utilizing a custom-built Graph Neural Network, we can simulate these dynamics at a fraction of the cost of traditional ab initio methods.

## Key Research Goals

* **DSpinGNN Development**: A novel Spin-Disentangled Equivariant Graph Neural Network architecture designed specifically to model magnetic materials.

* **Strain-Spin Coupling**: Capturing how mechanical strain and lattice vibrations influence the magnetic exchange interactions ($J_{ij}$).

* **$T_c$ Prediction**: Calculating the renormalized Curie temperature by accounting for dynamic lattice effects that static models often overlook.

## 🛠 Model Architecture: DSpinGNN

The DSpinGNN (Spin-Disentangled Equivariant Graph Neural Network) is designed to handle the distinct symmetries of magnetic systems. Unlike standard GNNs, it disentangles spatial coordinates from spin degrees of freedom to maintain physical equivariance.

### Implementation Highlights

* **From-Scratch Reproductions**: This project is informed by scratch implementations and benchmarking of NequIP and DTNN.

* **Equivariance**: Built using the e3nn library to ensure the model respects $E(3)$ symmetry.

* **Hamiltonian Integration**: The model predicts the potential energy surface and magnetic exchange parameters used in the SLD equations of motion.

The Hamiltonian used for the dynamics is defined as:

$$H = H_{lattice}(\{R_i\}) + \sum_{i \lt j} J_{ij}(R_{ij}) \mathbf{S}_i \cdot \mathbf{S}_j$$

## 📂 Repository Structure
```
├── src/
│   ├── models/             # DSpinGNN, NequIP, and DTNN implementations
│   ├── dynamics/           # Spin-Lattice Dynamics (SLD) integration scripts
│   └── layers/             # Equivariant message-passing layers
├── data/                   # DFT datasets for 2D materials (CrI3, CrGeTe3, etc.)
├── notebooks/              # Analysis of phase transitions and Tc extraction
├── tests/                  # Unit tests for model equivariance
├── requirements.txt        # Software dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites

* Python 3.9+

* PyTorch & PyTorch Geometric

* e3nn (for Euclidean Equivariance)

* ASE (Atomic Simulation Environment)


### Installation

Clone this repository:
```
git clone [https://github.com/your-username/thesis-strain-dynamics.git](https://github.com/your-username/thesis-strain-dynamics.git)
```

Install dependencies:
```
pip install -r requirements.txt
```

### Running Simulations

To train the DSpinGNN model on your dataset:
```
python src/train.py --config configs/dspin_config.yaml
```

To run the spin-lattice dynamics simulation for $T_c$ extraction:
```
python src/dynamics/run_sld.py --temp_range 0 500 --steps 100000
```

## 📊 Results & Analysis

The project demonstrates that lattice-driven renormalization significantly shifts the predicted Curie temperature in 2D systems compared to rigid-lattice approximations.

Phase Transition Analysis: Extraction of $T_c$ through susceptibility peaks and Binder cumulants.

Strain Effects: Analysis of how uniaxial and biaxial strain tunes the magnetic order through the $J(R)$ relationship.

## 🎓 About the Author

Isam Balghari MS Physics Candidate

Research Interests: Spintronics, Magnetic Materials, and Machine Learning Potentials.
