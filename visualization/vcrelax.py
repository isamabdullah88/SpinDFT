import numpy as np
from ase.db import connect

# --- Config ---
DB_PATH = 'VCRELAX.db'

def extract_detailed_parameters():
    try:
        db = connect(DB_PATH)
        # Select the row with the lowest energy (the final relaxed structure)
        # relaxed_row = min(db.select(), key=lambda x: x.energy)
        relaxed_row = db.get(id=1)
    except Exception as e:
        print(f"Error accessing database: {e}")
        return

    # 1. Get Atoms object and Cell info
    atoms = relaxed_row.toatoms()
    cell = atoms.get_cell()
    
    # Lengths (a, b, c) and Angles (alpha, beta, gamma)
    # These describe the SHAPE OF THE BOX (90, 90, 120)
    a, b, c = cell.lengths()
    alpha, beta, gamma = cell.angles()

    # 2. Extract Positions
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    # 3. Calculate dz_physical (Thickness)
    # We identify Cr and I atoms by their symbols
    cr_indices = [i for i, s in enumerate(symbols) if s == 'Cr']
    i_indices = [i for i, s in enumerate(symbols) if s == 'I']
    
    z_coords_cr = [positions[i][2] for i in cr_indices]
    z_coords_i  = [positions[i][2] for i in i_indices]

    # Average Cr plane height
    z_cr_avg = np.mean(z_coords_cr)
    
    # Split Iodine atoms into top and bottom planes
    z_i_top = [z for z in z_coords_i if z > z_cr_avg]
    z_i_bot = [z for z in z_coords_i if z < z_cr_avg]
    
    dz_top = np.mean(z_i_top) - z_cr_avg if z_i_top else 0
    dz_bot = z_cr_avg - np.mean(z_i_bot) if z_i_bot else 0
    dz_avg = (dz_top + dz_bot) / 2

    # 4. Calculate Representative Bond Angles (Local Chemistry)
    # This is what you see in the GUI when clicking atoms
    # We'll take the first Cr and find its nearest I neighbors
    cr_idx = cr_indices[0]
    distances = atoms.get_distances(cr_idx, i_indices, mic=True)
    # Get indices of the 6 nearest Iodines (the octahedron)
    neighbor_i_indices = [i_indices[idx] for idx in np.argsort(distances)[:6]]
    
    # Calculate an I-Cr-I angle (between the first two neighbors)
    # Format: get_angle(atom1, vertex, atom2)
    bond_angle_i_cr_i = atoms.get_angle(neighbor_i_indices[0], cr_idx, neighbor_i_indices[1], mic=True)

    # Calculate a Cr-I-Cr angle (Superexchange angle)
    # Find two Cr neighbors for the first Iodine
    i_idx = i_indices[0]
    cr_distances = atoms.get_distances(i_idx, cr_indices, mic=True)
    cr_neighbors = [cr_indices[idx] for idx in np.argsort(cr_distances)[:2]]
    bond_angle_cr_i_cr = atoms.get_angle(cr_neighbors[0], i_idx, cr_neighbors[1], mic=True)

    # --- Print Results ---
    print(f"--- RELAXED STRUCTURE DETAILS (ID: {relaxed_row.id}) ---")
    print(f"Lattice Constants (Å):  a = {a:.6f}, c = {c:.6f}")
    
    print("\n--- CELL PARAMETERS (The Box) ---")
    print(f"Cell Angles (deg):      α = {alpha:.2f}°, β = {beta:.2f}°, γ = {gamma:.2f}°")
    print("Note: These define the simulation box, not the atom bonds.")

    print("\n--- ATOMIC GEOMETRY (The Bonds) ---")
    print(f"Physical Thickness dz:  {dz_avg:.6f} Å")
    print(f"I-Cr-I Bond Angle:      {bond_angle_i_cr_i:.2f}° (What you see in GUI)")
    print(f"Cr-I-Cr Bond Angle:     {bond_angle_cr_i_cr:.2f}° (Superexchange angle)")
    
    print(f"\nTotal Potential Energy: {relaxed_row.energy:.6f} eV")
    
    print("\nAtomic Positions (Cartesian):")
    print(f"{'Index':<6} | {'Symbol':<4} | {'X':>10} | {'Y':>10} | {'Z':>10}")
    print("-" * 50)
    for i, (p, s) in enumerate(zip(positions, symbols)):
        print(f"{i:<6} | {s:<4} | {p[0]:>10.4f} | {p[1]:>10.4f} | {p[2]:>10.4f}")

if __name__ == "__main__":
    extract_detailed_parameters()