import numpy as np
from ase import Atoms

class CrI3Atom:
    def __init__(self, a=6.913154, c=19.260000, dz=1.618870):
        # Physical thickness (distance Cr to I plane) in Angstroms
        # Updated to perfectly match your vc-relax (U=3.0) ground state
        dz_physical = dz
        
        # Fractional offset for the 'c' vector
        z_offset = dz_physical / c 
        
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
        self.atoms = Atoms('Cr2I6', 
                           scaled_positions=scaled_positions, 
                           cell=cell, 
                           pbc=[True, True, True])

    def write(self, filename="Data/CrI3_relaxed.cif"):
        self.atoms.write(filename)
        print(f"Success! '{filename}' has been created.")

# If you want to quickly generate the perfect baseline file:
if __name__ == "__main__":
    import os
    os.makedirs("Data", exist_ok=True)
    baseline = CrI3Atom()
    baseline.write("Data/CrI3_relaxed.cif")