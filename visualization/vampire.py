import numpy as np
import matplotlib.pyplot as plt
import os

def plot_vampire_output(filepath='output'):
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Error: Could not find '{filepath}'. Make sure the script is in the Vampire folder.")
        return

    # Load the data, automatically ignoring all lines that start with '#'
    print(f"Loading data from {filepath}...")
    data = np.loadtxt(filepath, comments='#')

    # Extract columns (Python is 0-indexed)
    T = data[:, 0]          # Column 1: Temperature
    M_Cr1 = data[:, 4]      # Column 5: Mean Magnetization length for Cr1
    M_Cr2 = data[:, 8]      # Column 9: Mean Magnetization length for Cr2

    # Average the two sublattices to get the total system magnetization
    M_total = (M_Cr1 + M_Cr2) / 2.0

    # Set up a beautiful, publication-quality plot
    plt.figure(figsize=(9, 6))
    
    # Plot the main average magnetization
    plt.plot(T, M_total, marker='o', color='#1f77b4', linestyle='-', 
             linewidth=2.5, markersize=8, label='Total Magnetization')
    
    # Optional: Plot the individual sublattices as faint dashed lines to prove they overlap
    plt.plot(T, M_Cr1, marker='', color='red', linestyle='--', alpha=0.4, label='Cr1 Sublattice')
    plt.plot(T, M_Cr2, marker='', color='green', linestyle=':', alpha=0.4, label='Cr2 Sublattice')

    # Formatting the plot
    plt.title('Monte Carlo Phase Transition (CrI$_3$ at -1% Strain)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Temperature (K)', fontsize=14)
    plt.ylabel('Normalized Magnetization $\\langle |M| \\rangle$', fontsize=14)
    
    # Grid, limits, and legend
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, max(T) + 5)
    plt.ylim(-0.05, 1.05)
    plt.axhline(0, color='black', linewidth=1.2) # X-axis line
    
    plt.legend(fontsize=12, loc='upper right')
    
    # Add a text annotation pointing out the approximate Tc
    plt.annotate(f'Approx $T_c$ ≈ 65 K', 
                 xy=(65, 0.18), xytext=(75, 0.4),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    
    # Save and show
    save_path = 'Tc_Phase_Transition.png'
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved successfully as '{save_path}'")
    plt.show()

if __name__ == "__main__":
    vampirepath = 'DataSets/CrI3-Relax/FM/_strain_uniaxial_x_-0.0100/tmp/TB2J_results/Vampire/output'
    plot_vampire_output(vampirepath)