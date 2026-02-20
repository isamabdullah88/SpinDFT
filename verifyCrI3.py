import matplotlib.pyplot as plt
import numpy as np
from ase.db import connect


def sanitycheck(dbpath, plot=True):
    strains = []
    energies = []
    mag_moms_cr = []
    max_forces = []

    print(f"{'ID':<20} | {'Strain':<8} | {'Energy (eV)':<12} | {'Cr Spin (µB)':<12} | {'Max Force (eV/Å)':<15}")
    print("-" * 80)

    with connect(dbpath) as db:
        for row in db.select():
            
            s = row['strain_value']
            
            e = row.energy
            
            if 'mag_moments' in row.data:
                m = row.data['mag_moments'][0] # First atom (Cr)
            else:
                m = 0.0
                
            # Max Force Calculation
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
            
            mag_warning = " (!)" if abs(m) < 1.0 else ""
            print(f"{row.id:<20} | {s*100:>6.2f}% | {e:>12.4f} | {m:>12.4f}{mag_warning} | {f_max:>15.4f}")

    # --- Sort data for plotting (database might be out of order) ---
    sorted_indices = np.argsort(strains)
    strains = np.array(strains)[sorted_indices]
    energies = np.array(energies)[sorted_indices]
    mag_moms_cr = np.array(mag_moms_cr)[sorted_indices]
    max_forces = np.array(max_forces)[sorted_indices]

    if not plot:
        return strains, energies, mag_moms_cr, max_forces
    
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

    return strains, energies, mag_moms_cr, max_forces



if __name__ == "__main__":
    dbpathFM = 'DataSets/CrI3/FM/CrI3_Uniaxial_VC_FM.db'
    dbpathAFM = 'DataSets/CrI3/AFM/CrI3_Uniaxial_VC_AFM.db'

    strainFM, energiesFM, _, maxforcesFM = sanitycheck(dbpathFM, plot=True)
    strainAFM, energiesAFM, _, maxforcesAFM = sanitycheck(dbpathAFM, plot=True)

    plt.figure(figsize=(10, 5))
    plt.plot(strainFM, energiesFM, 'o-', label='FM', c='b')
    plt.plot(strainAFM, energiesAFM, 's-', label='AFM', c='r')
    
    plt.xlabel('Strain Index')
    plt.ylabel('Total Energy (eV)')
    plt.title('Energy Comparison: FM vs AFM')
    plt.legend()
    plt.grid(True)
    plt.show()