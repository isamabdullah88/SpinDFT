import os
import time
from ase.db import connect
from logger import getlogger

from qe import SCF
from config import prep_strains
from config import PHASE, KPTS, VCRELAX, RELAX, SOC, NSCF_NBNDS, STRAIN_TYPE
from exchange import Exchange, WorkspaceManager

log = getlogger("SpinDFT")
log.info(f"Starting SpinDFT pipeline with phase: {PHASE}, VCRELAX: {VCRELAX}, Relaxation: {RELAX}")


def run(dbpath, wkdir, prerelaxed_dir, ncalculations=15, coresperjob=6):
    TOTAL_CORES = os.cpu_count() - 2

    if VCRELAX:
        CORES_PER_JOB = TOTAL_CORES # Optimal for small unit cells. 
    else:
        CORES_PER_JOB = coresperjob # Optimal for small unit cells.
    NWORKERS = max(1, TOTAL_CORES // CORES_PER_JOB)

    os.environ['OMP_NUM_THREADS'] = '1'

    # TODO: DO this to prevent cache memeory error after long runs
    # THE FIX: Disable OpenMPI's buggy "vader" shared memory module
    # env["OMPI_MCA_btl_vader_single_copy_mechanism"] = "none"
    
    # # Optional: Prevent OpenBLAS from spawning too many threads in pure python scripts
    # if "wann2J.py" in cmd or serial:
    #     env["OMP_NUM_THREADS"] = "1"
    #     env["OPENBLAS_NUM_THREADS"] = "1"
    
    # Format the configuration into a single beautiful text block
    configsummary = (
        "\n"
        "==================================================\n"
        "             PIPELINE CONFIGURATION               \n"
        "==================================================\n"
        " [ Resource Optimization ]\n"
        f"   Total Cores        :  {TOTAL_CORES}\n"
        f"   Concurrent Jobs    :  {NWORKERS}\n"
        f"   Cores per Job      :  {CORES_PER_JOB}\n"
        "\n"
        " [ Physics Parameters ]\n"
        f"   Phase              :  {PHASE}\n"
        f"   K-Points           :  {KPTS}\n"
        f"   VC-Relax           :  {VCRELAX}\n"
        f"   Relaxation         :  {RELAX}\n"
        "=================================================="
    )
    
    # Log the entire block exactly once
    log.info(configsummary)

    wkdir = os.path.join(wkdir, PHASE)

    workspace = WorkspaceManager(wkdir)

    scf = SCF(wkdir, KPTS, phase=PHASE, prerelaxed_dir=prerelaxed_dir, cores_per_job=CORES_PER_JOB)

    if VCRELAX:
        res = scf.run(None, VCRELAX)
        log.info(f"VC-Relax Run: {res['status']}")
        scf.writedb(connect(dbpath), res)
        return
    
    # Generate strain tasks
    tasks = prep_strains(count = ncalculations)
    
    log.info(f"Starting Production Run...")
    
    with connect(dbpath) as db:
        for task in tasks:
            strain, stntype = task

            log.info("\n\n" + "="*100)
            log.info(f"Processing Strain {strain:.4f} ({stntype})")
            workspace.setwkdir(strain, STRAIN_TYPE)

            startt = time.time()

            res = scf.run((strain, stntype))
            scf.writedb(db, res)

            log.info(f"Strain {strain:.4f} ({stntype}): {res['status']}")

            workspace.cleanscf()

            exchangepl = Exchange(
                kpts=KPTS, 
                soc=SOC, 
                numcores=CORES_PER_JOB,
                nscf_nbnds=NSCF_NBNDS
            )
            exchangepl.run(res['atoms'], workspace.tmpdir)

            workspace.cleanwannier()

            endt = time.time()
            log.info(f"Completed TB2J and Wannier90 pipeline for Strain {strain:.4f} ({stntype}) in {endt - startt:.2f} seconds")
            log.info("="*100 + "\n\n")

        log.info("\n\n\n\n" + "="*100)
        log.info("All CALCULATIONS COMPLETED!")
        log.info("="*100)

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
    log.info(f"Total runtime: {endt - startt:.2f} seconds")