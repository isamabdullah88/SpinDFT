from .hubbard import EspressoHubbard
from .CrI3 import CrI3Atom
from .config import INPUT_SCF
import os
import numpy as np

class SCF:
    def __init__(self, worker_dir):
        self.worker_dir = worker_dir

        self.atoms = CrI3Atom().atoms

    def strainatoms(self, straincell):
        # Apply strain
        atoms = self.atoms.copy()
        atoms.set_cell(straincell, scale_atoms=True)
        return atoms

    def run(self, args):
        strain, straincell, wid, slabel = args

        worker_dir = self.worker_dir + f"_strain_{slabel}_{strain:.4f}"
        os.makedirs(worker_dir, exist_ok=True)

        # Apply Strain
        atoms = self.strainatoms(straincell)
        atoms.set_tags([0] * len(atoms))

        cridxs = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == 'Cr']
        
        # Tag Cr atoms for AFM ordering (Cr=0, Cr1=1)
        atoms[cridxs[0]].tag = 0
        atoms[cridxs[1]].tag = 1

        init_mags = np.zeros(len(atoms))
        init_mags[cridxs[0]] = 3.0 # First Cr (Always Up)
        init_mags[cridxs[1]] = -3.0 # Second Cr (Down if AFM)
        atoms.set_initial_magnetic_moments(init_mags)
        
        import time
        starttm = time.time()
        espressohub = EspressoHubbard()
        atomsout = espressohub.runQE(
            atoms, 
            INPUT_SCF, 
            kpts=(2, 2, 1), 
            directory=worker_dir, 
            command_prefix=f"mpirun -np 1"
        )
        print('SCF Completed in {time:.2f} seconds'.format(time=time.time() - starttm))
        
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

        print(f"Strain {strain:.4f}: SCF SUCCESS - Energy: {energy:.4f} eV, Mag Moments: {moms}")

        return result
        

if __name__ == "__main__":
    TOTAL_CORES = os.cpu_count()
    CORES_PER_JOB = 8 # Optimal for small unit cells. 
    NUM_WORKERS = max(1, TOTAL_CORES // CORES_PER_JOB)

    print(f"--- Resource Optimization ---")
    print(f"Total Cores Detected: {TOTAL_CORES}")
    print(f"Running {NUM_WORKERS} concurrent jobs with {CORES_PER_JOB} cores each.")

    strain = 0.0
    worker_dir = f"./DataSets/CrI3/Strain_{strain:.4f}"
    os.makedirs(worker_dir, exist_ok=True)

    scf = SCF(worker_dir=worker_dir)
    result = scf.run((strain, CORES_PER_JOB))
    print(result)