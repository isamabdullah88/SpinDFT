import numpy as np
import os
import shutil
import subprocess
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from ase.io import read
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.db import connect

# --- Configuration ---
# Update this if your JSON or CIF paths change
CONFIG_PATHS = {
    'json_input': 'Data/CrI3_DFT_Input.json',
    'cif_input': 'Data/CrI3.cif',
    'db_output': 'Data/CrI3_Strained_TB2J.db',
    'pseudo_dir': './SSSP_1.3.0_PBE_efficiency/'
}

PSEUDOS = {
    'Cr': 'cr_pbe_v1.5.uspp.F.UPF',
    'I':  'I.pbe-n-kjpaw_psl.0.2.UPF'
}

# --- Dynamic Resource Allocation ---
# We optimize for 8-atom CrI3. 
# QE scales best up to ~8-16 cores for this size.
# We split the machine into chunks of 'CORES_PER_JOB'.
TOTAL_CORES = os.cpu_count()
CORES_PER_JOB = 8 # Optimal for small unit cells. 
NUM_WORKERS = max(1, TOTAL_CORES // CORES_PER_JOB)

print(f"--- Resource Optimization ---")
print(f"Total Cores Detected: {TOTAL_CORES}")
print(f"Running {NUM_WORKERS} concurrent jobs with {CORES_PER_JOB} cores each.")

# Try to load JSON, fall back to default if file missing
try:
    INPUT_DATA_SCF = json.load(open(CONFIG_PATHS['json_input']))
except FileNotFoundError:
    print(f"WARNING: {CONFIG_PATHS['json_input']} not found. Using default internal config.")
    INPUT_DATA_SCF = {
        'control': {
            'calculation': 'scf', 
            'restart_mode': 'from_scratch',
            'outdir': './tmp', 
            'tprnfor': True, 'tstress': True, 'disk_io': 'low'
        },
        'system': {
            'ecutwfc': 40, 'ecutrho': 320,
            'occupations': 'smearing', 'smearing': 'mv', 'degauss': 0.01,
            'nspin': 2,
            'starting_magnetization(1)': 1.0, 
            'lda_plus_u': True, 'lda_plus_u_kind': 0,
            'Hubbard_U(1)': 3.0, 'Hubbard_U(2)': 0.0,
        },
        'electrons': {'conv_thr': 1.0e-5, 'mixing_beta': 0.3}
    }

# Ensure PSEUDO_DIR is set correctly in input data
INPUT_DATA_SCF['control']['pseudo_dir'] = CONFIG_PATHS['pseudo_dir']

def run_cmd(cmd, cwd):
    # Helper to run shell commands efficiently
    subprocess.run(cmd, shell=True, cwd=cwd, check=True, stdout=subprocess.DEVNULL)

def run_full_pipeline(args):
    """
    Runs the full pipeline: SCF -> NSCF -> Wannier90 -> TB2J
    """
    eps, base_struct_dict, worker_id = args
    
    # Define MPI command for this specific worker
    cmd_prefix = f"mpirun -np {CORES_PER_JOB}"

    # Unique directory for this worker/strain
    worker_dir = f"./calc_strain_{eps:.4f}"
    os.makedirs(worker_dir, exist_ok=True)
    
    # Copy Pseudos to worker dir
    for p in PSEUDOS.values():
        src = os.path.join(CONFIG_PATHS['pseudo_dir'], p)
        if os.path.exists(src) and not os.path.exists(os.path.join(worker_dir, p)):
            shutil.copy(src, worker_dir)

    # Reconstruct Atoms
    from ase import Atoms
    atoms = Atoms.fromdict(base_struct_dict)
    
    # Apply Strain (Biaxial)
    cell = atoms.get_cell()
    strain_matrix = np.array([[1+eps, 0, 0], [0, 1+eps, 0], [0, 0, 1]])
    atoms.set_cell(np.dot(cell, strain_matrix), scale_atoms=True)
    
    # 1. Run SCF (Cheap Screening & Forces)
    local_input = INPUT_DATA_SCF.copy()
    local_input['control']['outdir'] = './tmp'
    
    # Use the specific MPI command
    profile = EspressoProfile(command=f"{cmd_prefix} pw.x", pseudo_dir='.')
    
    # Use (4,4,1) k-points for better magnetic convergence
    atoms.calc = Espresso(profile=profile, pseudopotentials=PSEUDOS, 
                          input_data=local_input, kpts=(4, 4, 1), directory=worker_dir)
    
    result_data = {
        'strain': eps,
        'id': f"CrI3_Biaxial_{eps:.4f}",
        'status': 'INIT'
    }

    try:
        # Trigger SCF
        energy = atoms.get_potential_energy()
        moms = atoms.get_magnetic_moments()
        forces = atoms.get_forces()
        stress = atoms.get_stress()
        
        result_data.update({
            'energy': energy,
            'mag_moments': moms,
            'forces': forces,
            'stress': stress,
            'atoms': atoms
        })
        
        cr_mom = moms[0] # Assume Cr is first
        
        # --- SCREENING ---
        if abs(cr_mom) < 2.0:
            shutil.rmtree(worker_dir) # Cleanup
            result_data['status'] = 'SKIPPED (Non-Magnetic)'
            return result_data
            
    except Exception as e:
        result_data['status'] = f"SCF_FAIL: {str(e)}"
        return result_data

    # 2. Run NSCF (Heavy)
    input_nscf = local_input.copy()
    input_nscf['control']['calculation'] = 'nscf'
    input_nscf['system']['nbnd'] = 40 
    input_nscf['system']['nosym'] = True
    
    atoms.calc.set(input_data=input_nscf)
    try:
        atoms.get_potential_energy()
    except:
        result_data['status'] = "NSCF_FAIL"
        return result_data

    # 3. Wannier90 Prep
    prefix = 'espresso'
    
    win_content = f"""
num_bands = 40
num_wann = 28
dis_num_iter = 100
num_iter = 0
iprint = 2
begin projections
Cr:d
I:p
end projections
mp_grid = 4 4 1
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
    
    with open(os.path.join(worker_dir, f"{prefix}.win"), 'w') as f:
        f.write(win_content)

    # 4. Run Wannier Pipeline
    try:
        run_cmd(f"{cmd_prefix} wannier90.x -pp {prefix}", worker_dir)
        
        with open(os.path.join(worker_dir, f"{prefix}.pw2wan"), 'w') as f:
            f.write(f"&inputpp outdir='./tmp' prefix='{prefix}' write_mmn=.true. write_amn=.true. /")
            
        run_cmd(f"{cmd_prefix} pw2wannier90.x < {prefix}.pw2wan", worker_dir)
        run_cmd(f"{cmd_prefix} wannier90.x {prefix}", worker_dir)
        
        # 5. Run TB2J
        # ADDED: --np flag to make TB2J utilize the assigned cores
        run_cmd(f"wannier2J.py --posfile {prefix}.win --prefix {prefix} --elements Cr --kmesh 4 4 1 --np {CORES_PER_JOB}", worker_dir)
        
        # 6. Check Results
        result_xml = os.path.join(worker_dir, 'TB2J_results', 'exchange.xml')
        if os.path.exists(result_xml):
            os.makedirs("./results", exist_ok=True)
            shutil.copy(result_xml, f"./results/exchange_{eps:.4f}.xml")
            result_data['status'] = 'SUCCESS'
        else:
            result_data['status'] = 'TB2J_FAIL'
            
    except Exception as e:
        result_data['status'] = f"W90_FAIL: {str(e)}"
    finally:
        shutil.rmtree(os.path.join(worker_dir, 'tmp'), ignore_errors=True)
        
    return result_data


def main():
    # 1. Load Structure
    try:
        atoms_base = read(CONFIG_PATHS['cif_input'])
        print(f"Loaded structure from {CONFIG_PATHS['cif_input']}")
    except FileNotFoundError:
        print("CIF not found, generating dummy CrI3 for testing...")
        from ase.build import mx2
        atoms_base = mx2(formula='CrI3', kind='2H', a=6.86, thickness=3.0)
        atoms_base.cell[2][2] = 20.0
    
    # 2. Setup Strains
    strains = np.linspace(-0.05, 0.05, 12) # Generate 12 data points
    base_dict = atoms_base.asdict()
    
    # 3. Parallel Execution
    tasks = [(eps, base_dict, i % NUM_WORKERS) for i, eps in enumerate(strains)]
    
    print(f"Starting Production Run...")
    
    os.makedirs(os.path.dirname(CONFIG_PATHS['db_output']), exist_ok=True)
    
    with connect(CONFIG_PATHS['db_output']) as db, ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for res in executor.map(run_full_pipeline, tasks):
            print(f"Strain {res['strain']:.4f}: {res['status']}")
            
            if 'energy' in res:
                db.write(res['atoms'], 
                         key_value_pairs={
                             'strain_value': res['strain'],
                             'id': res['id'],
                             'energy': res['energy'],
                             'pipeline_status': res['status']
                         },
                         data={
                             'mag_moments': res['mag_moments'],
                             'forces': res['forces'],
                             'stress': res['stress']
                         })

if __name__ == "__main__":
    main()