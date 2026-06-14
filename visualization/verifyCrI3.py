import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ase.db import connect
import os

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
    
    straintype = 'Uniaxial_X'
    dbpathFM = f'./DataSets/HPC-Pre_VCRelax/Relaxed-6x6/Relaxed-{straintype}/FM/Relaxed-{straintype}-FM.db'
    dbpathAFM = f'./DataSets/HPC-Pre_VCRelax/Relaxed-6x6/Relaxed-{straintype}/AFM/Relaxed-{straintype}-AFM.db'
    
    print("DB exists:", os.path.exists(dbpathFM))
    print("DB exists:", os.path.exists(dbpathAFM))
    print('FM:  ', dbpathFM)
    print('AFM: ', dbpathAFM)

    strainFM, energiesFM, _, maxforcesFM = sanitycheck(dbpathFM, plot=False)
    strainAFM, energiesAFM, _, maxforcesAFM = sanitycheck(dbpathAFM, plot=False)

    minFM = np.argmin(energiesFM)
    print('minFM absolute:', strainFM[minFM], energiesFM[minFM])

    # Set the minimum energy where strain is zero for each dataset
    strainFM -= strainFM[minFM]  # Shift FM strains so that min is at 0%
    strainAFM -= strainAFM[minFM]  # Shift AFM strains similarly

    # Calculate Relative Energy (ΔE) to normalize the Y-axis
    global_min = min(energiesFM[minFM], energiesAFM[minFM])

    # --- Package Data for Seaborn ---
    df_fm = pd.DataFrame({
        f'{straintype} Strain (%)': strainFM * 100,
        'Relative Energy (eV)': energiesFM - global_min,
        'Magnetic State': 'Ferromagnetic (FM)'
    })
    
    df_afm = pd.DataFrame({
        f'{straintype} Strain (%)': strainAFM * 100,
        'Relative Energy (eV)': energiesAFM - global_min,
        'Magnetic State': 'Antiferromagnetic (AFM)'
    })
    
    # Combine the two datasets into one table
    df = pd.concat([df_fm, df_afm], ignore_index=True)
    
    # 'paper' context, clean ticks
    sns.set_theme(context="paper", style="ticks")

    # Font and Axis sizing for APS formatting (8-10 pt)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.labelsize': 9,       # Axis titles
        'xtick.labelsize': 8,      # X-axis numbers
        'ytick.labelsize': 8,      # Y-axis numbers
        'legend.fontsize': 7.5,    # Legend text
        'axes.linewidth': 0.8,     # Bounding box line thickness
        'xtick.major.width': 0.8,  # Tick thickness
        'ytick.major.width': 0.8
    })

    # 3. Figure Size: PRB single column is exactly 3.375 inches wide. 
    fig, ax = plt.subplots(figsize=(3.375, 2.6), dpi=600)

    # 4. Color Palette (Explicit dictionary: FM = Red, AFM = Blue)
    prb_palette = {
        'Ferromagnetic (FM)': '#d62728',     # Red
        'Antiferromagnetic (AFM)': '#1f77b4' # Blue
    } 

    # Draw the plot
    sns.lineplot(
        data=df,
        x=f'{straintype} Strain (%)',
        y='Relative Energy (eV)',
        hue='Magnetic State',     
        style='Magnetic State',   
        markers=['o', 's'],       
        dashes=False,             
        linewidth=1.2,            
        markersize=5,             
        palette=prb_palette,      
        ax=ax
    )

    # Aesthetics
    ax.set_xlabel(f'Uniaxial (X) Strain (%)')
    ax.set_ylabel(r'Relative Energy, $\Delta$E (eV)')
    
    # Add a subtle grid
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.3, color='gray')
    
    # Despine removes the top and right bounding box lines
    sns.despine()

    # Clean up the legend 
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, frameon=False, 
              loc='upper center', title=None)

    # Use tight layout to prevent clipped labels
    plt.tight_layout(pad=0.5)
    
    # Export as a true vector PDF with transparent background
    plt.savefig(f'Relaxed-{straintype}.pdf', format='pdf', transparent=True, bbox_inches='tight')
    print(f"PRB-formatted vector plot saved successfully as Relaxed-{straintype}.pdf!")
    
    plt.show()