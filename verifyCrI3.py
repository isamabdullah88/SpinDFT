import matplotlib.pyplot as plt
import numpy as np
from ase.db import connect

db_name = 'DataSets/CrI3_Strained_Biaxial10.db'

strains = []
energies = []
mag_moms_cr = []
max_forces = []

print(f"{'ID':<20} | {'Strain':<8} | {'Energy (eV)':<12} | {'Cr Spin (µB)':<12} | {'Max Force (eV/Å)':<15}")
print("-" * 80)

with connect(db_name) as db:
    for row in db.select():
        # 1. Extract Strain
        # If 'strain_value' key is missing, try to infer from ID or set to 0
        s = row['strain_value']
            
        # 2. Extract Energy
        e = row.energy
        
        # 3. Extract Magnetic Moment (Check first Cr atom)
        # We handle cases where data might be missing
        if 'mag_moments' in row.data:
            m = row.data['mag_moments'][0] # First atom (Cr)
        else:
            m = 0.0
            
        # 4. Extract Forces
        # Calculate the magnitude of force on each atom: sqrt(fx^2 + fy^2 + fz^2)
        # Then take the MAXIMUM force in the crystal.
        if 'forces' in row.data:
            forces = row.data['forces']
            # Linear algebra norm along axis 1 (atoms)
            f_norms = np.linalg.norm(forces, axis=1)
            f_max = np.max(f_norms)
        else:
            f_max = 0.0
        
        strains.append(s)
        energies.append(e)
        mag_moms_cr.append(m)
        max_forces.append(f_max)
        
        # Print Table Row
        # Highlight bad magnetism in the printout
        mag_warning = " (!)" if abs(m) < 1.0 else ""
        print(f"{row.id:<20} | {s*100:>6.2f}% | {e:>12.4f} | {m:>12.4f}{mag_warning} | {f_max:>15.4f}")

# --- Sort data for plotting (database might be out of order) ---
sorted_indices = np.argsort(strains)
strains = np.array(strains)[sorted_indices]
energies = np.array(energies)[sorted_indices]
mag_moms_cr = np.array(mag_moms_cr)[sorted_indices]
max_forces = np.array(max_forces)[sorted_indices]

print('strains: ', strains)
print('energies: ', energies)
print('mag_moms_cr: ', mag_moms_cr)

# --- Plotting ---
plt.figure(figsize=(15, 5))

# Plot 1: Energy
plt.subplot(1, 3, 1)
plt.plot(strains, energies, 'o-', c='b')
plt.xlabel('Strain')
plt.ylabel('Total Energy (eV)')
plt.title('Energy Landscape')
plt.grid(True)

# Plot 2: Magnetism (The most critical check for CrI3)
plt.subplot(1, 3, 2)
plt.plot(strains, mag_moms_cr, 's-', c='r')
plt.axhline(y=3.0, color='gray', linestyle='--', label='Expected (3.0)')
plt.xlabel('Strain')
plt.ylabel('Cr Magnetic Moment (µB)')
plt.title('Spin Stability')
plt.legend()
plt.grid(True)

# Plot 3: Forces (The "Happiness" check)
plt.subplot(1, 3, 3)
plt.plot(strains, max_forces, '^-', c='k')
plt.xlabel('Strain')
plt.ylabel('Max Force (eV/Å)')
plt.title('Atomic Forces')
plt.grid(True)

plt.tight_layout()
plt.show()