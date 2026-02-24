import os
from ase.io import read, write
from ase.calculators.espresso import Espresso, EspressoProfile

# --- Config ---
PSEUDO_DIR = os.path.abspath('./SSSP_1.3.0_PBE_efficiency/')
PSEUDOS = {'Cr': 'cr_pbe_v1.5.uspp.F.UPF', 'I': 'I.pbe-n-kjpaw_psl.0.2.UPF'}

# Variable Cell Relax Parameters
INPUT_DATA = {
    'control': {
        'calculation': 'vc-relax',  # Varies cell size + atomic positions
        'restart_mode': 'from_scratch',
        'outdir': './tmp_relax',
        'pseudo_dir': '.',
        'tprnfor': True,
        'tstress': True,
        'disk_io': 'low'
    },
    'system': {
        'ecutwfc': 40, 'ecutrho': 320,
        'occupations': 'fixed', 
        # 'smearing': 'mv', 'degauss': 0.01,
        'nspin': 2, 'nosym': True,
        'starting_magnetization(1)': 3.0, 'starting_magnetization(2)': 0.0,
        # 'constrained_magnetization': 'total', # Constraint Type
        'tot_magnetization': 6.0,             # Target Value (Bohr Magnetons)
    },
    'electrons': {'conv_thr': 1.0e-5, 'mixing_beta': 0.2, 'diagonalization': 'cg'},
    'ions': {'ion_dynamics': 'bfgs'},
    'cell': {'cell_dynamics': 'bfgs', 'press_conv_thr': 0.1} # Target 0 pressure
}

def run_relax():
    # Load your current base
    atoms = read('DataSets/CrI3.cif')
    
    # Setup Calculator (Manual patching logic embedded)
    # We use a simplified manual run here for clarity
    from ase.io import write as ase_write
    import subprocess
    
    work_dir = "relax_run"
    os.makedirs(work_dir, exist_ok=True)
    
    # Copy pseudos
    import shutil
    for p in PSEUDOS.values():
        shutil.copy(os.path.join(PSEUDO_DIR, p), work_dir)
        
    input_path = os.path.join(work_dir, 'espresso.pwi')
    output_path = os.path.join(work_dir, 'espresso.pwo')
    
    # Write Input
    ase_write(input_path, atoms, format='espresso-in', input_data=INPUT_DATA, 
              pseudopotentials=PSEUDOS, kpts=(4, 4, 1))
    
    # Patch Hubbard
    with open(input_path, 'a') as f:
        f.write("\nHUBBARD (atomic)\nU Cr-3d 3.0\n\n")
        
    # Run
    print("Running vc-relax to find equilibrium lattice constant...")
    # Using 4 cores for the relax
    cmd = f"mpirun -np 8 pw.x -in espresso.pwi > espresso.pwo"
    
    # Force single thread
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    
    subprocess.run(cmd, shell=True, cwd=work_dir, env=env, check=True)
    
    print("Done! Check relax_run/espresso.pwo for the final cell parameters.")
    print("Look for 'CELL_PARAMETERS' at the end of the file.")

if __name__ == "__main__":
    run_relax()