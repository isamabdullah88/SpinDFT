import numpy as np
from ase import Atoms
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.db import connect
from tqdm import tqdm
import os

# --- 1. Configuration ---
# EXACT filenames you provided
pseudos = {
    'Cr': 'cr_pbe_v1.5.uspp.F.UPF',
    'I':  'I.pbe-n-kjpaw_psl.0.2.UPF'
}
pseudo_dir = './'  # Ensure your .UPF files are in the same folder as this script!

# M1 Pro Optimization: Using 6 P-Cores
profile = EspressoProfile(command='mpirun -np 8 pw.x', pseudo_dir=pseudo_dir)

# DFT Parameters
input_data = {
    'control': {
        'calculation': 'scf',
        'restart_mode': 'from_scratch',
        'pseudo_dir': pseudo_dir,
        'outdir': './tmp',
        'tprnfor': True,      # Forces needed for GNN
        'tstress': True,
        'disk_io': 'none'     # Save SSD
    },
    'system': {
        'ecutwfc': 30,        # Increased slightly for Iodine PAW
        'ecutrho': 240,       # 10x ecutwfc (Required for Cr USPP stability)
        'occupations': 'smearing',
        'smearing': 'mv',     # Marzari-Vanderbilt smearing (good for metals/magnetism)
        'degauss': 0.01,
        'nspin': 2,           # Enable Spin
        'starting_magnetization(1)': 0.6, # Cr atoms (Magnetic)
        'starting_magnetization(2)': 0.0, # I atoms (Non-magnetic)
    },
    'electrons': {
        'mixing_beta': 0.3,   
        'conv_thr': 1.0e-4,
        'diagonalization': 'cg'
    }
}

# --- 2. Build CrI3 Monolayer ---
a = 6.86
c = 18.0

cell = [[a, 0, 0],
        [-a/2, a * np.sqrt(3)/2, 0],
        [0, 0, c]]

# Fractional coordinates
positions = [
    [0.3333, 0.6667, 0.5000], # Cr
    [0.6667, 0.3333, 0.5000], # Cr
    [0.3333, 0.0000, 0.5700], # I
    [0.0000, 0.3333, 0.5700], # I
    [0.6667, 0.6667, 0.5700], # I
    [0.6667, 0.0000, 0.4300], # I
    [0.0000, 0.6667, 0.4300], # I
    [0.3333, 0.3333, 0.4300]  # I
]
cart_positions = np.dot(positions, cell)
atoms_ideal = Atoms('Cr2I6', positions=cart_positions, cell=cell, pbc=[True, True, True])

# --- 3. Dataset Generation Loop ---
db_name = 'cri3_dataset.db'
n_samples = 5

print(f"Generating {n_samples} samples using Cr USPP & I PAW...")

with connect(db_name) as db:
    for i in tqdm(range(n_samples)):
        atoms = atoms_ideal.copy()
        
        # Perturb atomic positions (Rattle)
        atoms.rattle(stdev=0.05, seed=i)
        
        atoms.calc = Espresso(profile=profile, 
                              pseudopotentials=pseudos, 
                              input_data=input_data,
                              kpts=(3, 3, 1)) 
        
        try:
            atoms.get_potential_energy()
            # Save Magnetic Moments too!
            mag_moms = atoms.get_magnetic_moments()
            db.write(atoms, data={'mag_moments': mag_moms, 'step': i})
            
        except Exception as e:
            print(f"Step {i} failed: {e}")

print("Done. Verify with: ase gui cri3_dataset.db")