import os
import logging
from ase.db import connect

from .workspace import WorkspaceManager
from .wannier90 import Wannier90
from qe import NSCF
from config import KPTS, INPUT_SCF, NSCF_NBNDS, WANNIER_NBNDS, PHASE


class Exchange:
    """Executes the NSCF and Wannier/TB2J steps for a single configuration."""
    def __init__(self, kpts, soc, numcores, nscf_nbnds, wannier_nbnds):
        self.kpts = kpts
        self.soc = soc
        self.numcores = numcores
        self.nscf_nbnds = nscf_nbnds
        self.wannier_nbnds = wannier_nbnds

        self.logger = logging.getLogger("SpinDFT")

    def run(self, atoms, wkdir):
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Executing Pipeline for Wkdir: {wkdir}")
        self.logger.info(f"{'='*50}")
        
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

        # Step 2: Wannier90 & TB2J (using perfectly matched subset of bands)
        wannier = Wannier90(
            wkdir=wkdir,
            kmesh=self.kpts, 
            soc=self.soc, 
            nbnds=self.wannier_nbnds
        )
        result = wannier.run(atoms, self.numcores)
        
        # Step 3: Conditional Cleanup
        # If the Wannier/TB2J run returns a SUCCESS status, delete the junk.
        # if isinstance(result, dict) and result.get('status') == 'SUCCESS':
        #     self.cleanup()
        # else:
        #     self.logger.warning("\nWARNING: Pipeline step failed. Retaining heavy files for debugging purposes!")
            
        # return result

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