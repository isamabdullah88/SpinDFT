import os
import numpy as np
from ase import Atoms
from ase.io import read

from .config import PHASE
import logging

class CrI3:
    def __init__(self, a=6.913154, c=19.260000, dz=1.618870, prerelaxed_dir=None):
        """
        Initialize the CrI3 class with the pristine lattice parameters.
        
        :param a: In-plane lattice parameter (Angstroms)
        :param c: Out-of-plane lattice parameter (Angstroms)
        :param dz: Physical thickness (distance Cr to I plane) in Angstroms
        :param prerelaxed_dir: Path to the directory containing pre-relaxed atom files (optional).
        """
        self.prerelaxed_dir = prerelaxed_dir

        # Fractional offset for the 'c' vector
        z_offset = dz / c 
        
        # Fractional coordinates relative to the center (0.5)
        z_high = 0.5 + z_offset
        z_low  = 0.5 - z_offset

        # Define the hexagonal simulation box
        cell = [[a, 0, 0],
                [-a/2, a * np.sqrt(3)/2, 0],
                [0, 0, c]]

        # Scaled (fractional) positions
        scaled_positions = [
            [0.3333, 0.6667, 0.5000], # Cr
            [0.6667, 0.3333, 0.5000], # Cr
            [0.3333, 0.0000, z_high], # I
            [0.0000, 0.3333, z_high], # I
            [0.6667, 0.6667, z_high], # I
            [0.6667, 0.0000, z_low],  # I
            [0.0000, 0.6667, z_low],  # I
            [0.3333, 0.3333, z_low]   # I
        ]

        # Using scaled_positions allows ASE to automatically handle the Cartesian math
        self.batoms = Atoms('Cr2I6', 
                           scaled_positions=scaled_positions, 
                           cell=cell, 
                           pbc=[True, True, True])
        
        self.logprefix = "[CrI3]"
        self.logger = logging.getLogger("SpinDFT")

    def write_baseline(self, filename="Data/CrI3_relaxed.cif"):
        """Saves the pristine 0% strain unit cell."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.batoms.write(filename)
        self.logger.info(f"{self.logprefix} Success! Baseline '{filename}' has been created.")

    def strain_atoms(self, stntype, stnvalue):
        """
        Returns the Atoms object for a specific strain type and value.
        """
        if stntype not in ['Biaxial', 'Uniaxial_X', 'Shear_XY']:
            raise ValueError("stntype must be either 'Biaxial', 'Uniaxial_X', or 'Shear_XY'.")

        if self.prerelaxed_dir is not None:
            filename = f"Strain_{stntype}_{stnvalue:.4f}.json"
            filepath = os.path.join(self.prerelaxed_dir, PHASE, filename)
            
            if os.path.exists(filepath):
                self.logger.info(f"{self.logprefix} Found relaxed structure. Loading {filename}...")
                return read(filepath)
            else:
                self.logger.info(f"{self.logprefix} No pre-relaxed file found at {filepath}. Generating mathematically...")

        # Generate unrelaxed strained structure mathematically
        self.logger.info(f"{self.logprefix} Applying {stnvalue} {stntype} strain to base structure...")
        
        atoms = self.batoms.copy()
        
        # Get the original cell vectors
        cell = atoms.get_cell()
        
        # Apply strain
        if stntype == 'Biaxial':
            # Apply in-plane biaxial strain to both 'a' and 'b' lattice vectors
            cell[0] *= (1.0 + stnvalue)
            cell[1] *= (1.0 + stnvalue)
        
        elif stntype == 'Uniaxial_X':
            # Apply uniaxial strain to the 'a' lattice vector only
            cell[0] *= (1.0 + stnvalue)

        elif stntype == 'Shear_XY':
            # Apply shear by adding a component of the second lattice vector to the first
            cell[0] += cell[1] * stnvalue
        
        # Scale atoms
        atoms.set_cell(cell, scale_atoms=True)
        
        return atoms

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    # Initialize the class, optionally pointing it to a directory with pre-relaxed files
    cri3_manager = CrI3(prerelaxed_dir="relaxed_atoms_dir")
    
    # Write the pristine baseline
    cri3_manager.write_baseline()
    
    # Request a 2% BIAXIAL strain
    # If "relaxed_atoms_dir/strain_biaxial_0.020.json" is missing, it scales mathematically
    biaxial_atoms = cri3_manager.strain_atoms(stntype="Biaxial", stnvalue=0.02)
    print("\nBiaxial Cell:\n", biaxial_atoms.get_cell())
    
    # Request a 2% UNIAXIAL strain
    uniaxial_atoms = cri3_manager.strain_atoms(stntype="Uniaxial_X", stnvalue=0.02)
    print("\nUniaxial Cell:\n", uniaxial_atoms.get_cell())

    # Request a 2% SHEAR strain
    shear_atoms = cri3_manager.strain_atoms(stntype="Shear_XY", stnvalue=0.02)
    print("\nShear Cell:\n", shear_atoms.get_cell())