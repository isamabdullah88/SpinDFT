import os
import shutil
import subprocess
import numpy as np
from ase.db import connect
from ase.calculators.espresso import Espresso, EspressoProfile

# --- Configuration ---
# Must match your system paths exactly
PSEUDO_DIR = './' 
PSEUDOS = {'Cr': 'cr_pbe_v1.5.uspp.F.UPF', 'I': 'I.pbe-n-kjpaw_psl.0.2.UPF'}

# Commands
MPI_CMD = 'mpirun -np 8'
PW_CMD = f'{MPI_CMD} pw.x'
W90_CMD = f'{MPI_CMD} wannier90.x'
PW2WAN_CMD = f'{MPI_CMD} pw2wannier90.x'

# TB2J settings
# Spin-Orbit Coupling (SOC) is REQUIRED for DMI and Anisotropy (K)
# If you run without SOC, you only get J.
USE_SOC = True 

def run_tb2j_for_structure(atoms, struct_id, work_dir="tb2j_work"):
    """
    Runs the full QE -> Wannier90 -> TB2J pipeline for a single structure.
    """
    
    # 1. Prepare Directory
    calc_dir = os.path.join(work_dir, struct_id)
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudos to calc_dir (QE sometimes needs them locally)
    for p in PSEUDOS.values():
        if not os.path.exists(os.path.join(calc_dir, p)):
            shutil.copy(os.path.join(PSEUDO_DIR, p), calc_dir)

    prefix = 'cri3'
    
    # --- 2. SCF Calculation (Ground State) ---
    # We need a well-converged charge density
    input_data_scf = {
        'control': {
            'calculation': 'scf',
            'prefix': prefix,
            'pseudo_dir': '.',
            'outdir': './tmp',
            'disk_io': 'low'
        },
        'system': {
            'ecutwfc': 40, # Higher cutoff for Wannier
            'ecutrho': 320,
            'occupations': 'smearing',
            'smearing': 'mv',
            'degauss': 0.01,
            'nspin': 2 if not USE_SOC else 4, # 4 = Non-collinear (SOC)
            'lspinorb': USE_SOC,              # Enable SOC
            'noncolin': USE_SOC,
            # Initial Magnetic Moments (Aligned along Z for FM)
            'starting_magnetization(1)': 0.6, # Cr
            'angle1(1)': 0.0,                 # Angle from z-axis (important for SOC)
            'starting_magnetization(2)': 0.0, # I
        },
        'electrons': {'conv_thr': 1.0e-6} # Tight convergence
    }
    
    profile = EspressoProfile(command=PW_CMD, pseudo_dir='.')
    atoms.calc = Espresso(input_data=input_data_scf, pseudopotentials=PSEUDOS, 
                          kpts=(6, 6, 1), profile=profile, directory=calc_dir)
    
    print(f"[{struct_id}] Running SCF...")
    atoms.get_potential_energy() # Triggers execution

    # --- 3. NSCF Calculation (More Bands) ---
    # Wannier needs unoccupied bands. 
    # Rule of thumb: Num_Bands >= Num_Wannier_Projections + few extra
    # Cr d(5) + I p(3). Cr2I6 -> (2*5 + 6*3) = 28 bands minimum. 
    # Let's compute 50 to be safe.
    
    input_data_nscf = input_data_scf.copy()
    input_data_nscf['control'].update({'calculation': 'nscf'})
    input_data_nscf['system'].update({'nbnd': 50, 'nosym': True}) # nosym important for W90
    
    atoms.calc.set(input_data=input_data_nscf)
    print(f"[{struct_id}] Running NSCF...")
    atoms.get_potential_energy()

    # --- 4. Wannier90 Pre-processing ---
    # Create .win file
    win_content = f"""
num_bands = 50
num_wann = 28
dis_num_iter = 100
num_iter = 0        ! We don't need maximal localization for TB2J, just disentanglement usually
iprint = 2

begin projections
Cr:d
I:p
end projections

mp_grid = 6 6 1

begin unit_cell_cart
{atoms.cell[0][0]} {atoms.cell[0][1]} {atoms.cell[0][2]}
{atoms.cell[1][0]} {atoms.cell[1][1]} {atoms.cell[1][2]}
{atoms.cell[2][0]} {atoms.cell[2][1]} {atoms.cell[2][2]}
end unit_cell_cart

begin atoms_cart
"""
    for atom in atoms:
        win_content += f"{atom.symbol} {atom.position[0]} {atom.position[1]} {atom.position[2]}\n"
    win_content += "end atoms_cart\n"

    with open(os.path.join(calc_dir, f'{prefix}.win'), 'w') as f:
        f.write(win_content)

    print(f"[{struct_id}] Running Wannier90 -pp...")
    subprocess.run(f"{W90_CMD} -pp {prefix}", shell=True, cwd=calc_dir, check=True)

    # --- 5. PW2Wannier90 ---
    pw2wan_in = f"""&inputpp 
   outdir = './tmp'
   prefix = '{prefix}'
   seedname = '{prefix}'
   spin_component = 'none' 
   write_mmn = .true.
   write_amn = .true.
   write_unk = .false.
/
"""
    with open(os.path.join(calc_dir, f'{prefix}.pw2wan'), 'w') as f:
        f.write(pw2wan_in)

    print(f"[{struct_id}] Running PW2Wannier90...")
    subprocess.run(f"{PW2WAN_CMD} < {prefix}.pw2wan > pw2wan.out", shell=True, cwd=calc_dir, check=True)

    # --- 6. Wannier90 Main Run ---
    print(f"[{struct_id}] Running Wannier90...")
    subprocess.run(f"{W90_CMD} {prefix}", shell=True, cwd=calc_dir, check=True)

    # --- 7. TB2J ---
    # This extracts the parameters.
    # We use the command line interface 'wannier2J.py' provided by TB2J.
    # Note: 'elements' arg helps TB2J identify magnetic ions
    print(f"[{struct_id}] Running TB2J...")
    
    tb2j_cmd = (
        f"wannier2J.py --posfile {prefix}.win --prefix {prefix} "
        f"--elements Cr --efermi {get_fermi_energy(calc_dir)} "
        f"--kmesh 6 6 1"
    )
    subprocess.run(tb2j_cmd, shell=True, cwd=calc_dir, check=True)
    
    print(f"[{struct_id}] Success! J values in {calc_dir}/TB2J_results")

def get_fermi_energy(calc_dir):
    # Quick helper to grep Fermi energy from SCF output
    # (TB2J needs this to define the occupancy)
    try:
        with open(os.path.join(calc_dir, "espresso.pwo"), 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "Fermi" in line:
                    return float(line.split()[-2])
    except:
        return 0.0 # Fallback

if __name__ == "__main__":
    # Read from the database generated by your previous script
    db_name = 'cri3_strained.db'
    
    with connect(db_name) as db:
        for row in db.select():
            print(f"\n--- Processing {row.id} (Strain: {row.strain_value:.3f}) ---")
            atoms = row.toatoms()
            
            # Run the heavy pipeline
            try:
                run_tb2j_for_structure(atoms, row.id)
            except Exception as e:
                print(f"Error processing {row.id}: {e}")