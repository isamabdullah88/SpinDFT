import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.calculators.espresso import Espresso, EspressoProfile
from tqdm import tqdm

# --- Settings ---
pseudo_dir = './' 
# Use your M1 core count (e.g., 6 or 8)
qe_profile = EspressoProfile(command='mpirun -np 6 pw.x', pseudo_dir=pseudo_dir)

# Shared parameters
base_params = {
    'control': {'calculation': 'scf', 'pseudo_dir': pseudo_dir, 'outdir': './tmp', 'disk_io': 'none'},
    'system': {'ecutwfc': 30, 'ibrav': 0},
    'electrons': {'mixing_beta': 0.7, 'conv_thr': 1.0e-6}
}

distances = np.linspace(0.5, 2.5, 15)  
E_singlet = []
E_triplet = []

print("Starting Spin-Dependent Scan...")

for d in tqdm(distances, desc="Scanning Bonds"):
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, d]], cell=[10, 10, 10], pbc=True)
    
    # --- 1. SINGLET (Non-Magnetic / Anti-parallel) ---
    # nspin=1 is the default (non-polarized)
    atoms.calc = Espresso(profile=qe_profile, 
                          pseudopotentials={'H': 'H.pbe-rrkjus_psl.1.0.0.UPF'},
                          input_data=base_params, kpts=None)
    E_singlet.append(atoms.get_potential_energy())

    # --- 2. TRIPLET (Ferromagnetic / Parallel Spins) ---
    # We copy the base params and add SPIN settings
    triplet_params = base_params.copy()
    triplet_params['system']['nspin'] = 2              # Enable Spin
    triplet_params['system']['tot_magnetization'] = 2.0 # 2 electrons both UP (Total Mag = 2)
    
    # IMPORTANT: We must tell QE to start with a magnetic guess
    triplet_params['system']['starting_magnetization(1)'] = 1.0 

    atoms.calc = Espresso(profile=qe_profile, 
                          pseudopotentials={'H': 'H.pbe-rrkjus_psl.1.0.0.UPF'},
                          input_data=triplet_params, kpts=None)
    E_triplet.append(atoms.get_potential_energy())

# --- Plotting ---
plt.figure(figsize=(8, 5))
plt.plot(distances, E_singlet, 'o-', label='Singlet (↑↓) - Bonding')
plt.plot(distances, E_triplet, 's-', color='red', label='Triplet (↑↑) - Repulsive')

plt.title('Exchange Energy: Singlet vs Triplet H2')
plt.xlabel('Bond Length (Å)')
plt.ylabel('Energy (eV)')
plt.legend()
plt.grid(True)
plt.savefig('spin_comparison.png')
plt.show()