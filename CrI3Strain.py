import numpy as np
from ase.io import read
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.db import connect
from tqdm import tqdm
import json
import os

# --- Configuration ---
PSEUDOS = {
    'Cr': 'cr_pbe_v1.5.uspp.F.UPF',
    'I':  'I.pbe-n-kjpaw_psl.0.2.UPF'
}
PSEUDO_DIR = './SSSP_1.3.0_PBE_efficiency/' 

# Profile for M1/Parallel execution
PROFILE = EspressoProfile(command='mpirun -np 8 pw.x', pseudo_dir=PSEUDO_DIR)

# DFT Parameters for CrI3 (Magnetism enabled)
INPUT_DATA = json.load(open('Data/CrI3_DFT_Input.json'))

def generate_strained_dataset(cifpath, dbpath, strain_range=(-0.05, 0.05),
                              num_steps=20):
    """
    Generates strained structures and immediately runs QE calculations.
    Args:
        cifpath: Path to the relaxed CrI3.cif
        dbpath: Output database filename
        strain_range: Tuple of (min_strain, max_strain)
        num_steps: How many data points to generate
    """
    # Read cif into 'Atoms' object
    atoms_base = read(cifpath)
    
    strains = np.linspace(strain_range[0], strain_range[1], num_steps)
    
    print(f"Starting DFT calculations for {num_steps} biaxial strained structures...")

    # Open database connection
    with connect(dbpath) as db:
        for eps in tqdm(strains):
            atoms_strained = atoms_base.copy()
            
            # 2. Apply Strain
            original_cell = atoms_strained.get_cell()
            strain_matrix = np.array([[1 + eps, 0, 0], [0, 1 + eps, 0], [0, 0, 1]])
            new_cell = np.dot(original_cell, strain_matrix)
            atoms_strained.set_cell(new_cell, scale_atoms=True)
            
            # 3. Attach Calculator
            atoms_strained.calc = Espresso(
                profile=PROFILE,
                pseudopotentials=PSEUDOS,
                input_data=INPUT_DATA,
                kpts=(2, 2, 1)  # K-points
            )
            
            structid = f"CrI3_Biaxial_{eps:.4f}"
            
            # 4. Run Calculation & Save
            energy = atoms_strained.get_potential_energy()
            
            # Magnetic data
            mag_moms = atoms_strained.get_magnetic_moments()
            forces = atoms_strained.get_forces()
            stress = atoms_strained.get_stress()
            
            # Save ASE database
            db.write(atoms_strained, data={
                'strain_value': eps,
                'id': structid,
                'mag_moments': mag_moms,
                'energy': energy,
                'forces': forces,
                'stress': stress
            })
                
    print(f"Done. Data saved to {dbpath}")

if __name__ == "__main__":
    generate_strained_dataset("Data/CrI3.cif", "Data/CrI3_Strained1.db", num_steps=5)