import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.calculators.espresso import Espresso, EspressoProfile
from tqdm import tqdm  # <--- Progress bar library

# --- 1. Settings ---
# Path to your pseudopotentials folder (Ensure this path is correct!)
pseudo_dir = './' 

# Define how to run the code using the new "Profile" method
# Change '4' to the number of P-Cores you want to use (e.g., 6 or 8 for M1 Pro)
qe_profile = EspressoProfile(command='mpirun -np 4 pw.x', pseudo_dir=pseudo_dir)

# --- 2. Define Calculation Parameters ---
calc_params = {
    'control': {
        'calculation': 'scf',
        'pseudo_dir': pseudo_dir,
        'outdir': './tmp',
        'disk_io': 'none'  # Optimization: Don't write wavefunctions to disk (saves SSD life)
    },
    'system': {
        'ecutwfc': 30,  
        'ibrav': 0,     
        'occupations': 'smearing', # Added smearing to be safe
        'smearing': 'gauss',
        'degauss': 0.01
    },
    'electrons': {
        'mixing_beta': 0.7,
        'conv_thr': 1.0e-6
    }
}

# --- 3. Scan Loop ---
distances = np.linspace(0.5, 2.5, 15) 
energies = []
forces_list = []

# Using tqdm to wrap the loop creates the progress bar
print("Starting H2 Bond Scan...")
for d in tqdm(distances, desc="Calculating Steps", unit="step"):
    
    # Create Hydrogen dimer
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, d]], cell=[10, 10, 10], pbc=True)
    
    # Attach the calculator with the new Profile
    atoms.calc = Espresso(profile=qe_profile, 
                          pseudopotentials={'H': 'H.pbe-rrkjus_psl.1.0.0.UPF'},
                          tstress=True, tprnfor=True,
                          input_data=calc_params,
                          kpts=None) 

    # Run Calculation
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    
    energies.append(e)
    forces_list.append(f[1, 2]) 

# --- 4. Results & Visualization ---
print("\nScan Complete!")
print(f"{'Dist (A)':<10} {'Energy (eV)':<15} {'Force (eV/A)'}")
print("-" * 40)
for d, e, f in zip(distances, energies, forces_list):
    print(f"{d:<10.2f} {e:<15.5f} {f:.5f}")

plt.figure(figsize=(8, 5))
plt.plot(distances, energies, 'o-', label='DFT Energy (QE)')
plt.title('H2 Dissociation Curve on M1 Pro')
plt.xlabel('Bond Length (Å)')
plt.ylabel('Total Energy (eV)')
plt.grid(True)
plt.legend()
plt.savefig('h2_pes.png')
print("\nPlot saved as 'h2_pes.png'")
plt.show()