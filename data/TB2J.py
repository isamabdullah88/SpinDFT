import os
import shutil
from ase.db import connect

from .config import INPUT_SCF, KPTS, PHASE
from .wannier90 import Wannier90
from .nscf import NSCF

class WorkspaceManager:
    """Manages the creation of directories and injection of charge densities."""
    def __init__(self, base_wkdir, base_outdir):
        self.base_wkdir = os.path.abspath(base_wkdir)
        self.base_outdir = os.path.abspath(base_outdir)

    def prepare_strain_workspace(self, strain_val):
        """Creates temporary and output directories for a specific strain."""
        strain_wkdir = os.path.join(self.base_wkdir, f"_strain_uniaxial_x_{strain_val:.4f}", "tmp")
        strain_outdir = os.path.join(self.base_outdir, f"_strain_uniaxial_x_{strain_val:.4f}")
        
        os.makedirs(strain_wkdir, exist_ok=True)
        os.makedirs(strain_outdir, exist_ok=True)
        
        return strain_wkdir, strain_outdir

    def inject_scf_density(self, scf_source_path, target_wkdir):
        """Copies the precomputed SCF charge density to the working directory."""
        if not scf_source_path or not os.path.exists(scf_source_path):
            print(f"WARNING: SCF source not found at {scf_source_path}. NSCF may fail!")
            return False

        dest_save = os.path.join(target_wkdir, "pwscf.save")
        if os.path.exists(dest_save):
            shutil.rmtree(dest_save)
            
        shutil.copytree(scf_source_path, dest_save)
        print(f"Copied pre-computed SCF charge density from {scf_source_path}")
        return True


class StrainPipeline:
    """Executes the NSCF and Wannier/TB2J steps for a single configuration."""
    def __init__(self, atoms, strain_val, wkdir, outdir, kpts, soc, numcores, nscf_nbnds, wannier_nbnds):
        self.atoms = atoms
        self.strain_val = strain_val
        self.wkdir = wkdir
        self.outdir = outdir
        self.kpts = kpts
        self.soc = soc
        self.numcores = numcores
        self.nscf_nbnds = nscf_nbnds
        self.wannier_nbnds = wannier_nbnds

    def run(self):
        print(f"\n{'='*50}")
        print(f"Executing Pipeline for Strain: {self.strain_val:.4f}")
        print(f"{'='*50}")
        
        # Step 1: Explicit NSCF Calculation (using buffer bands)
        qe_runner = NSCF(
            atoms=self.atoms, 
            INPUT_SCF=INPUT_SCF, 
            wkdir=self.wkdir, 
            kmesh=self.kpts, 
            soc=self.soc, 
            nbnds=self.nscf_nbnds
        )
        qe_runner.run(self.numcores)

        # Step 2: Wannier90 & TB2J (using perfectly matched subset of bands)
        wannier_runner = Wannier90(
            INPUT_SCF=INPUT_SCF, 
            wkdir=self.wkdir, 
            outdir=self.outdir, 
            kmesh=self.kpts, 
            soc=self.soc, 
            nbnds=self.wannier_nbnds
        )
        result = wannier_runner.run(self.atoms, self.numcores)
        
        return result


class TB2JMaster:
    """Master controller that connects to the database and dispatches jobs."""
    def __init__(self, dbpath, wkdir, outdir, kpts=(2, 2, 1), soc=False, numcores=8):
        self.dbpath = dbpath
        self.kpts = kpts
        self.soc = soc
        self.numcores = numcores
        self.workspace = WorkspaceManager(wkdir, outdir)
        
        # Explicit band configuration directly exposed to the master orchestrator!
        # Easily tweak the buffer bands right here:
        self.nscf_nbnds = 100
        self.wannier_nbnds = 80

    def run_all(self, scf_save_dir_template=None):
        print(f"Connecting to database: {self.dbpath}")
        print(f"Global K-Mesh enforced: {self.kpts}")
        print(f"Band Configuration: NSCF={self.nscf_nbnds}, Wannier={self.wannier_nbnds}")
        
        with connect(self.dbpath) as db:
            # Iterate through all database rows (Restored from the hardcoded `row = db.get(4)`)
            for row in db.select():
                atoms = db.get_atoms(row.id)
                strain_val = row.get('strain_value', 0.0) 
                
                # Setup directories
                strain_wkdir, strain_outdir = self.workspace.prepare_strain_workspace(strain_val)
                print(f"Working Dir: {strain_wkdir}")
                print(f"Output Dir:  {strain_outdir}")
                
                # Inject precomputed SCF density
                # Note: You can dynamically format this string if your save folders have strain values in their name!
                scf_source = scf_save_dir_template
                self.workspace.inject_scf_density(scf_source, strain_wkdir)

                # Execute pipeline
                pipeline = StrainPipeline(
                    atoms=atoms,
                    strain_val=strain_val,
                    wkdir=strain_wkdir,
                    outdir=strain_outdir,
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
    outdir = f"./DataSets/TB2J/{PHASE}"
    wkdir = f"./DataSets/CrI3-Relax/{PHASE}"
    dbpath = os.path.join(wkdir, f"CrI3_Uniaxial_VC_{PHASE}.db")
    
    # Update this path if your scf.save files are stored per-strain
    precomputed_scf_path = "./pwscf.save" 
    
    NUMCORES = 8
    
    # Initialize and run master
    master = TB2JMaster(
        dbpath=dbpath, 
        wkdir=wkdir, 
        outdir=outdir, 
        kpts=KPTS, 
        soc=False, 
        numcores=NUMCORES
    )
    
    master.run_all(scf_save_dir_template=precomputed_scf_path)