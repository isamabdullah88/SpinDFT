import numpy as np
from ase import Atoms


class CrI3Atom:
    def __init__(self, a=6.867, c=18.0):
        a = 7.348
        c = 19.260
        # Physical thickness (distance Cr to I plane) in Angstroms
        dz_physical = 1.348
        
        # Fractional offset for any 'c'
        z_offset = dz_physical / c 
        
        # Fractional coordinates relative to the center (0.5)
        z_high = 0.5 + z_offset
        z_low  = 0.5 - z_offset

        cell = [[a, 0, 0],
                [-a/2, a * np.sqrt(3)/2, 0],
                [0, 0, c]]

        positions = [
            [0.3333, 0.6667, 0.5000], # Cr
            [0.6667, 0.3333, 0.5000], # Cr
            [0.3333, 0.0000, z_high], # I
            [0.0000, 0.3333, z_high], # I
            [0.6667, 0.6667, z_high], # I
            [0.6667, 0.0000, z_low],  # I
            [0.0000, 0.6667, z_low],  # I
            [0.3333, 0.3333, z_low]   # I
        ]

        # Convert to Cartesian
        cart_positions = np.dot(positions, cell)

        # Create the Atoms Object
        self.atoms = Atoms('Cr2I6', positions=cart_positions, cell=cell, pbc=[True, True, True])

    def write(self, filename="CrI3.cif"):
        self.atoms.write(filename)

        print(f"Success! '{filename}' has been created.")