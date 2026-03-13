import os
import logging
from ase.db import connect

from .workspace import WorkspaceManager
from .wannier90 import Wannier90
from .TB2J import TB2JExchange
from qe import NSCF
from config import KPTS, TB2J_KPTS, INPUT_SCF, NSCF_NBNDS, PHASE


class Exchange:
    """Executes the NSCF and Wannier/TB2J steps for a single configuration."""
    def __init__(self, kpts, soc, numcores, nscf_nbnds):
        self.kpts = kpts
        self.soc = soc
        self.numcores = numcores
        self.nscf_nbnds = nscf_nbnds

        self.logger = logging.getLogger("SpinDFT")

    def run(self, atoms, wkdir):
        self.logger.info(f"[Exchange] {'-'*50}")
        self.logger.info(f"[Exchange] Executing Pipeline for Wkdir: {wkdir}")
        
        # Step 1: Explicit NSCF Calculation (using buffer bands)
        nscf = NSCF(
            atoms=atoms, 
            INPUT_SCF=INPUT_SCF, 
            wkdir=wkdir, 
            kmesh=self.kpts, 
            soc=self.soc, 
            nbnds=self.nscf_nbnds
        )
        nscf.run(self.numcores)

        # Wannier90
        wannier = Wannier90(
            wkdir=wkdir,
            kmesh=KPTS,
            soc=self.soc, 
            nscf_nbnds=self.nscf_nbnds
        )
        wannier.run(atoms, self.numcores)

        # TB2J
        self.tb2j = TB2JExchange(
            wkdir,
            TB2J_KPTS, 
            self.soc,
            self.numcores
        )
        self.tb2j.run()

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # Paths
    # outdir = f"./DataSets/TB2J/{PHASE}"
    wkdir = f"./DataSets/Test/{PHASE}"
    dbpath = "DataSets/Test/Test.db"
    
    # Update this path if your scf.save files are stored per-strain
    precomputed_scf_path = "./pwscf.save" 
    
    NUMCORES = 8

    workspace = WorkspaceManager(wkdir)
    
    # Initialize and run master
    exchangepl = Exchange(
        kpts=KPTS, 
        soc=False, 
        numcores=NUMCORES,
        nscf_nbnds=NSCF_NBNDS,
        wannier_nbnds=WANNIER_NBNDS
    )

    exchangepl.logger.info(f"Connecting to database: {dbpath}")
    with connect(dbpath) as db:
        # Iterate through all database rows
        for row in db.select():
        # row = db.get(0)
            exchangepl.logger.info(f'Processing row ID: {row.id} with strain: {row["strain_value"]}')
            atoms = db.get_atoms(row.id)
            strain = row['strain_value']

            workspace.setwkdir(strain)

            exchangepl.run(atoms, workspace.tmpdir)

            workspace.cleanwannier()