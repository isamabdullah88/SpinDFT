from .hubbard import EspressoHubbard
from .CrI3 import CrI3Atom
from .config import INPUT_SCF
import os
import numpy as np

class SCF:
    # ---------------------------------------------------------------------------------------------
    def __init__(self, wkdir, nworkers, kpts, phase='FM'):
        self.wkdir = wkdir
        self.nworkers = nworkers
        self.kpts = kpts
        self.phase = phase

        self.atoms = CrI3Atom().atoms

    # ---------------------------------------------------------------------------------------------
    def strainatoms(self, straincell):
        # Apply strain
        atoms = self.atoms.copy()

        if straincell is None:
            return atoms
        
        atoms.set_cell(straincell, scale_atoms=True)
        return atoms
    
    # ---------------------------------------------------------------------------------------------
    def initmags(self, atoms):
        cridxs = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == 'Cr']

        init_mags = np.zeros(len(atoms))

        if self.phase == 'AFM':
            # Tag Cr atoms for AFM ordering (Cr=0, Cr1=1)
            atoms[cridxs[0]].tag = 0
            atoms[cridxs[1]].tag = 1

            init_mags[cridxs[0]] = 3.0 # First Cr (Always Up)
            init_mags[cridxs[1]] = -3.0 # Second Cr (Down if AFM)
        else:
            # FM ordering (Both Cr atoms up)
            init_mags[cridxs] = 3.0

        atoms.set_initial_magnetic_moments(init_mags)

        return atoms

    # ---------------------------------------------------------------------------------------------
    def run(self, args, vcrelax=False):

        if not vcrelax:
            strain, straincell, wid, slabel = args

            # Apply Strain
            atoms = self.strainatoms(straincell)
        else:
            atoms = self.strainatoms(None)
            slabel, strain = 'vc_relax', 0.0
            
        # atoms.set_tags([0] * len(atoms))

        atoms = self.initmags(atoms)
        
        wkdir = self.wkdir + f"_strain_{slabel}_{strain:.4f}"
        os.makedirs(wkdir, exist_ok=True)
        espressohub = EspressoHubbard(phase=self.phase)
        atomsout = espressohub.runQE(
            atoms, 
            INPUT_SCF, 
            kpts=self.kpts, 
            directory=wkdir
        )
        
        result = {
            'strain': strain,
            'id': f"CrI3_Biaxial_{strain:.4f}",
            'status': 'INIT'
        }

        energy = atomsout.get_potential_energy()
        moms = atomsout.get_magnetic_moments()
        forces = atomsout.get_forces()
        stress = atomsout.get_stress()
        
        result.update({
            'status': 'SUCCESS',
            'energy': energy,
            'mag_moments': moms,
            'forces': forces,
            'stress': stress,
            'atoms': atomsout
        })

        # print(f"Strain {strain:.4f}: SCF SUCCESS - Energy: {energy:.4f} eV, Mag Moments: {moms}")

        return result
        

# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    TOTAL_CORES = os.cpu_count()
    CORES_PER_JOB = 8 # Optimal for small unit cells. 
    NUM_WORKERS = max(1, TOTAL_CORES // CORES_PER_JOB)

    print(f"--- Resource Optimization ---")
    print(f"Total Cores Detected: {TOTAL_CORES}")
    print(f"Running {NUM_WORKERS} concurrent jobs with {CORES_PER_JOB} cores each.")

    strain = 0.0
    wkdir = f"./DataSets/CrI3/Strain_{strain:.4f}"
    os.makedirs(wkdir, exist_ok=True)

    scf = SCF(wkdir=wkdir)
    result = scf.run((strain, CORES_PER_JOB))
    print(result)