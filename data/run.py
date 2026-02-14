import numpy as np
from .scf import SCF
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from ase.db import connect
from tqdm import tqdm
from .strain import prepare_cell_tasks


def multiworker():
    # strains = np.linspace(-0.1, 0.1, 30) # Generate 12 data points
    # strains = [0]
    
    TOTAL_CORES = os.cpu_count() - 4
    CORES_PER_JOB = 1 # Optimal for small unit cells. 
    NUM_WORKERS = max(1, TOTAL_CORES // CORES_PER_JOB)
    # NUM_WORKERS = 1

    print(f"--- Resource Optimization ---")
    print(f"Total Cores Detected: {TOTAL_CORES}")
    print(f"Running {NUM_WORKERS} concurrent jobs with {CORES_PER_JOB} cores each.")

    worker_dir = f"./DataSets/CrI3/"
    os.makedirs(worker_dir, exist_ok=True)
    scf = SCF(worker_dir=worker_dir)
    
    # 3. Parallel Execution
    # tasks = [(eps, CORES_PER_JOB) for eps in strains]
    tasks = prepare_cell_tasks(scf.atoms.get_cell(), num_workers=NUM_WORKERS, num_total=11)
    
    print(f"Starting Production Run...")
    
    dbpath = './DataSets/CrI3_Strained_Biaxial_AFM.db'
    # os.makedirs(dbpath, exist_ok=True)
    
    # scf.run((0.0, 1)) # Test run for single strain value
    # exit()

    with connect(dbpath) as db, ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for res in tqdm(executor.map(scf.run, tasks), total=len(tasks)):
            tqdm.write(f"Strain {res['strain']:.4f}: {res['status']}")
            
            if res['status'] != 'SUCCESS':
                tqdm.write(f"Skipping database write for strain {res['strain']:.4f} due to status: {res['status']}")
                continue

            db.write(
                res['atoms'],
                key_value_pairs={
                    'strain_value': res['strain'],
                    'dataid': res['id'],
                    'pipeline_status': res['status']
                },
                data={
                    'mag_moments': res['mag_moments'],
                    'forces': res['forces'],
                    'stress': res['stress']
                }
            )
            

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    multiworker()