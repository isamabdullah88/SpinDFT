import os
import shutil
from ase.db import connect

from .config import INPUT_SCF
from .wannier90 import Wannier90
from .nscf import NSCF

class TB2J:
    """
    Master pipeline controller.
    Manages database extraction, directory routing, and ensures physics
    parameters (like kmesh) remain strictly synced across all sub-modules.
    """
    def __init__(self, wkdir, outdir, kmesh=(2, 2, 1)):
        self.wkdir = os.path.abspath(wkdir)
        self.outdir = os.path.abspath(outdir)
        self.kmesh = kmesh
        os.makedirs(self.outdir, exist_ok=True)

    def run(self, dbpath, numcores=10, soc=False, scf_save_dir=None):
        print(f"Connecting to database: {dbpath}")
        print(f"Global K-Mesh enforced: {self.kmesh}")
        
        with connect(dbpath) as db:
            for row in db.select():
                atoms = db.get_atoms(row.id)
                strain_val = row.get('strain_value', 0.0) 
                
                print(f"\n{'='*46}")
                print(f"Processing Strain: {strain_val:.4f}")
                print(f"{'='*46}")

                # Set up precise temporary and output directories
                strain_wkdir = os.path.join(self.wkdir, f"Strain_Uniaxial_X_{strain_val:.4f}", "tmp")
                strain_outdir = os.path.join(self.outdir, f"_strain_uniaxial_x_{strain_val:.4f}")
                
                os.makedirs(strain_wkdir, exist_ok=True)
                os.makedirs(strain_outdir, exist_ok=True)
                
                print(f"Working Dir: {strain_wkdir}")
                print(f"Output Dir:  {strain_outdir}")

                # Copy pre-computed SCF charge density to the working directory if provided
                if scf_save_dir and os.path.exists(scf_save_dir):
                    dest_save = os.path.join(strain_wkdir, "pwscf.save")
                    if os.path.exists(dest_save):
                        shutil.rmtree(dest_save)
                    shutil.copytree(scf_save_dir, dest_save)
                    print(f"Copied pre-computed SCF charge density to {dest_save}")
                else:
                    print("WARNING: No scf_save_dir provided or found. NSCF may fail if charge density is missing!")

                # Step 1: Run explicit NSCF seamlessly (Reads the copied SCF density)
                # qe_runner = NSCF(atoms, INPUT_SCF, strain_wkdir, kmesh=self.kmesh, soc=soc)
                # qe_runner.run(numcores)

                # Step 2: Run Wannier90 & TB2J
                wannier_runner = Wannier90(INPUT_SCF, strain_wkdir, strain_outdir, kmesh=self.kmesh, soc=soc)
                result = wannier_runner.run(atoms, numcores)
                
                print(f"Strain {strain_val:.4f} finished with status: {result.get('status')}")

if __name__ == "__main__":
    outdir = "./DataSets/TB2J/FM"
    wkdir = "./DataSets/TB2J/"
    dbpath = "./DataSets/TB2J/CrI3_Uniaxial_FM.db"
    
    os.environ['OMP_NUM_THREADS'] = '1'
    # Update this to point to the pwscf.save folder from your successful SCF run
    precomputed_scf_path = "./pwscf.save" 
    
    # Change kmesh here to effortlessly update the entire pipeline
    production_kmesh = (2, 2, 1) 
    
    tb2j_runner = TB2J(wkdir, outdir, kmesh=production_kmesh)
    tb2j_runner.run(dbpath, numcores=6, soc=False, scf_save_dir=precomputed_scf_path)