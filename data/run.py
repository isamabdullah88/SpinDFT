import numpy as np
from .scf import SCF
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from ase.db import connect
from tqdm import tqdm
from .strain import prep_strains
from .config import INPUT_SCF, PHASE, KPTS, VCRELAX, WKDIR, RELAXED_DIR

def writedb(db, res):
    if res['status'] != 'SUCCESS':
        tqdm.write(f"Skipping database write for strain {res['strain']:.4f} due to status: {res['status']}")
        return

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
            'stress': res['stress'],
            'scf_parameters': INPUT_SCF,
            'kpoints': KPTS
        }
    )

def multiworker():
    TOTAL_CORES = os.cpu_count() - 2

    if VCRELAX:
        CORES_PER_JOB = 8 # Optimal for small unit cells. 
    else:
        CORES_PER_JOB = 2 # Optimal for small unit cells.
    NWORKERS = max(1, TOTAL_CORES // CORES_PER_JOB)

    os.environ['OMP_NUM_THREADS'] = str(CORES_PER_JOB)
    # os.environ['MKL_NUM_THREADS'] = '1'
    # os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    
    print(f"--- Resource Optimization ---")
    print(f"Total Cores Detected: {TOTAL_CORES}")
    print(f"Running {NWORKERS} concurrent jobs with {CORES_PER_JOB} cores each.")

    # wkdir = f"./DataSets/CrI3/FM-Test"
    dbpath = os.path.join(WKDIR, f"CrI3_Uniaxial_{PHASE}.db")
    os.makedirs(WKDIR, exist_ok=True)

    scf = SCF(WKDIR, NWORKERS, KPTS, phase=PHASE, relaxed_dir=RELAXED_DIR)

    if VCRELAX:
        res = scf.run(None, VCRELAX)
        print(f"VC-Relax Test Run: {res['status']}")
        writedb(connect(dbpath), res)
        return
    
    # Generate strain tasks
    tasks = prep_strains(count = 4)
    
    print(f"Starting Production Run...")
    

    with connect(dbpath) as db, ProcessPoolExecutor(max_workers=NWORKERS) as executor:
        for res in tqdm(executor.map(scf.run, tasks), total=len(tasks)):
            tqdm.write(f"Strain {res['strain']:.4f}: {res['status']}")
            
            writedb(db, res)
            

if __name__ == "__main__":
    multiworker()