import os
import shutil
import glob
from ase.db import connect

from config import INPUT_SCF, KPTS, PHASE, NSCF_NBNDS, WANNIER_NBNDS
from .wannier90 import Wannier90
from qe import NSCF

class WorkspaceManager:
    """Manages the creation of directories and injection of charge densities."""
    def __init__(self, base_wkdir, base_outdir):
        self.base_wkdir = os.path.abspath(base_wkdir)
        self.base_outdir = os.path.abspath(base_outdir)

    def initwsp(self, strain):
        """Creates temporary and output directories for a specific strain."""
        strain_wkdir = os.path.join(self.base_wkdir, f"Strain_Uniaxial_X_{strain:.4f}", "tmp")
        
        os.makedirs(strain_wkdir, exist_ok=True)
        
        return strain_wkdir

    # def inject_scf_density(self, scf_source_path, target_wkdir):
    #     """Copies the precomputed SCF charge density to the working directory."""
    #     if not scf_source_path or not os.path.exists(scf_source_path):
    #         print(f"WARNING: SCF source not found at {scf_source_path}. NSCF may fail!")
    #         return False

    #     dest_save = os.path.join(target_wkdir, "pwscf.save")
    #     if os.path.exists(dest_save):
    #         shutil.rmtree(dest_save)
            
    #     shutil.copytree(scf_source_path, dest_save)
    #     print(f"Copied pre-computed SCF charge density from {scf_source_path}")
    #     return True


class Exchange:
    """Executes the NSCF and Wannier/TB2J steps for a single configuration."""
    def __init__(self, atoms, strain, wkdir, kpts, soc, numcores, nscf_nbnds, wannier_nbnds):
        self.atoms = atoms
        self.strain = strain
        self.wkdir = wkdir
        self.kpts = kpts
        self.soc = soc
        self.numcores = numcores
        self.nscf_nbnds = nscf_nbnds
        self.wannier_nbnds = wannier_nbnds

    def cleanup(self):
        """Surgically removes massive intermediate files to save HPC quota."""
        print("\n--- Initiating HPC Quota Cleanup ---")
        
        # 1. Delete the Quantum ESPRESSO wavefunctions/charge density
        qe_save_dir = os.path.join(self.wkdir, "pwscf.save")
        if os.path.exists(qe_save_dir):
            shutil.rmtree(qe_save_dir)
            print(f"[Deleted] QE Save Directory: {qe_save_dir}")
            
        # 2. Delete Wannier90/TB2J heavy matrices
        heavy_extensions = ['*.mmn', '*.amn', '*.chk', '*.eig', '*.nnkp', '*wfc*']
        deleted_count = 0
        
        for ext in heavy_extensions:
            # Search for the heavy files in the working directory
            for filepath in glob.glob(os.path.join(self.wkdir, ext)):
                try:
                    os.remove(filepath)
                    deleted_count += 1
                except OSError as e:
                    print(f"Warning: Could not delete {filepath} - {e}")
                    
        print(f"[Deleted] {deleted_count} heavy Wannier90/QE matrix files.")
        print("--- Cleanup Complete: Kept lightweight logs and TB2J results ---\n")

    def run(self):
        print(f"\n{'='*50}")
        print(f"Executing Pipeline for Strain: {self.strain:.4f}")
        print(f"{'='*50}")
        
        # Step 1: Explicit NSCF Calculation (using buffer bands)
        nscf = NSCF(
            atoms=self.atoms, 
            INPUT_SCF=INPUT_SCF, 
            wkdir=self.wkdir, 
            kmesh=self.kpts, 
            soc=self.soc, 
            nbnds=self.nscf_nbnds
        )
        nscf.run(self.numcores)

        # Step 2: Wannier90 & TB2J (using perfectly matched subset of bands)
        wannier = Wannier90(
            wkdir=self.wkdir,
            kmesh=self.kpts, 
            soc=self.soc, 
            nbnds=self.wannier_nbnds
        )
        result = wannier.run(self.atoms, self.numcores)
        
        # Step 3: Conditional Cleanup
        # If the Wannier/TB2J run returns a SUCCESS status, delete the junk.
        if isinstance(result, dict) and result.get('status') == 'SUCCESS':
            self.cleanup()
        else:
            print("\nWARNING: Pipeline step failed. Retaining heavy files for debugging purposes!")
            
        return result


class TB2J:
    """Master controller that connects to the database and dispatches jobs."""
    def __init__(self, wkdir, kpts=(2, 2, 1), soc=False, numcores=8):
        # self.dbpath = dbpath
        self.kpts = kpts
        self.soc = soc
        self.numcores = numcores
        self.workspace = WorkspaceManager(wkdir, wkdir)
        
        # Explicit band configuration directly exposed to the master orchestrator!
        # Easily tweak the buffer bands right here:
        self.nscf_nbnds = NSCF_NBNDS
        self.wannier_nbnds = WANNIER_NBNDS

    def run(self, atoms, strain):
        print(f"Global K-Mesh enforced: {self.kpts}")
        print(f"Band Configuration: NSCF={self.nscf_nbnds}, Wannier={self.wannier_nbnds}")
        
        
        
        # Setup directories
        strain_wkdir = self.workspace.initwsp(strain)
        print(f"Working Dir: {strain_wkdir}")
        
        # Inject precomputed SCF density
        # scf_source = scf_save_dir_template
        # self.workspace.inject_scf_density(scf_source, strain_wkdir)

        # Execute pipeline
        pipeline = Exchange(
            atoms=atoms,
            strain=strain,
            wkdir=strain_wkdir,
            kpts=self.kpts,
            soc=self.soc,
            numcores=self.numcores,
            nscf_nbnds=self.nscf_nbnds,
            wannier_nbnds=self.wannier_nbnds
        )
        pipeline.run()

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # Paths
    # outdir = f"./DataSets/TB2J/{PHASE}"
    wkdir = f"./DataSets/Test/{PHASE}"
    dbpath = "DataSets/Test/Test.db"
    
    # Update this path if your scf.save files are stored per-strain
    precomputed_scf_path = "./pwscf.save" 
    
    NUMCORES = 8
    
    # Initialize and run master
    pipeline = TB2J(
        wkdir=wkdir,
        kpts=KPTS, 
        soc=False, 
        numcores=NUMCORES
    )

    print(f"Connecting to database: {dbpath}")
    with connect(dbpath) as db:
        # Iterate through all database rows
        for row in db.select():
        # row = db.get(0)
            print('row')
            atoms = db.get_atoms(row.id)
            strain = row['strain_value']

            pipeline.run(atoms, strain)