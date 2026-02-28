import os
import time
from ase.db import connect
from tqdm import tqdm

from qe import SCF
from config import prep_strains
from config import INPUT_SCF, PHASE, KPTS, VCRELAX, RELAX, SOC, NSCF_NBNDS, WANNIER_NBNDS
from tb2j import Exchange, WorkspaceManager

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

def run(dbpath, wkdir, prerelaxed_dir, ncalculations=15, coresperjob=6):
    TOTAL_CORES = os.cpu_count() - 2

    if VCRELAX:
        CORES_PER_JOB = TOTAL_CORES # Optimal for small unit cells. 
    else:
        CORES_PER_JOB = coresperjob # Optimal for small unit cells.
    NWORKERS = max(1, TOTAL_CORES // CORES_PER_JOB)

    os.environ['OMP_NUM_THREADS'] = '1'
    
    print(f"--- Resource Optimization ---")
    print(f"Total Cores Detected: {TOTAL_CORES}")
    print(f"Running {NWORKERS} concurrent jobs with {CORES_PER_JOB} cores each.")
    print("-----------------------------")
    print(f"Using K-Points: {KPTS}")
    print(f"Phase: {PHASE}")
    print(f"VC-Relax Enabled: {VCRELAX}")
    print(f"Relaxation: {RELAX}")

    wkdir = os.path.join(wkdir, PHASE)

    workspace = WorkspaceManager(wkdir)

    scf = SCF(wkdir, KPTS, phase=PHASE, prerelaxed_dir=prerelaxed_dir, cores_per_job=CORES_PER_JOB)

    if VCRELAX:
        res = scf.run(None, VCRELAX)
        print(f"VC-Relax Run: {res['status']}")
        writedb(connect(dbpath), res)
        return
    

    # Generate strain tasks
    tasks = prep_strains(count = ncalculations)
    
    print(f"Starting Production Run...")
    
    with connect(dbpath) as db:
        for task in tasks:
            strain, stntype = task

            workspace.setwkdir(strain)


            # startt = time.time()
            res = scf.run((strain, stntype))
            print(f"Strain {strain:.4f} ({stntype}): {res['status']}")
            writedb(db, res)

            # workspace.cleanscf()

            # print(f"Running Exchange Pipeline for Strain {strain:.4f} ({stntype})...")
            exchangepl = Exchange(
                kpts=KPTS, 
                soc=False, 
                numcores=CORES_PER_JOB,
                nscf_nbnds=NSCF_NBNDS,
                wannier_nbnds=WANNIER_NBNDS
            )
            exchangepl.run(res['atoms'], workspace.tmpdir)

            # workspace.cleanwannier()

            endt = time.time()
            print(f"Completed TB2J and Wannier90 pipeline for Strain {strain:.4f} ({stntype}) in {endt - startt:.2f} seconds")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run DFT calculations for strained CrI3.")
    parser.add_argument('--WKDIR', type=str, required=True, help='Working directory for calculations and database.')
    parser.add_argument('--PRERELAXED_DIR', type=str, default=None, help='Directory containing pre-relaxed structures (optional).')
    parser.add_argument('--DBPATH', type=str, required=True, help='Path to the SQLite database file.')
    parser.add_argument('--N_CALCULATIONS', type=int, default=15, help='Number of strained configurations to compute.')
    parser.add_argument('--CORES_PER_JOB', type=int, default=6, help='Number of cores to use per job.')
    args = parser.parse_args()
    
    startt = time.time()
    run(dbpath=args.DBPATH, wkdir=args.WKDIR, prerelaxed_dir=args.PRERELAXED_DIR,
        ncalculations=args.N_CALCULATIONS, coresperjob=args.CORES_PER_JOB)
    endt = time.time()
    print(f"Total runtime: {endt - startt:.2f} seconds")