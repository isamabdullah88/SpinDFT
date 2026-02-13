from ase.calculators.espresso import EspressoProfile, Espresso
from .hubbard import EspressoHubbard
from .CrI3 import CrI3Atom
from .config import INPUT_SCF, PSEUDOS, PSEUDO_DIR
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

        # Unique directory for this worker/strain
        worker_dir = self.worker_dir + f"_strain_{slabel}_{strain:.4f}"
        os.makedirs(worker_dir, exist_ok=True)

        # Apply Strain (Biaxial)
        # cell = self.atoms.get_cell()
        # strain_matrix = np.array([[1+strain, 0, 0], [0, 1+strain, 0], [0, 0, 1]])
        # self.atoms.set_cell(np.dot(cell, strain_matrix), scale_atoms=True)
        atoms = self.strainatoms(straincell)

        initmags = [3.0 if atom.symbol == 'Cr' else 0.0 for atom in atoms]
        atoms.set_initial_magnetic_moments(initmags)
        
        # Use the specific MPI command
        # profile = EspressoProfile(command=f"mpirun -np {numcores} pw.x", pseudo_dir=PSEUDO_DIR)
        # print(f"Running SCF with command: mpirun -np {numcores} pw.x")
        
        # self.atoms.calc = EspressoHubbard(
        #     profile=profile,
        #     pseudopotentials=PSEUDOS,
        #     input_data=INPUT_SCF,
        #     kpts=(3, 3, 1),
        #     directory=self.worker_dir + f"_strain_{strain:.4f}"
        # )
        import time
        starttm = time.time()
        espressohub = EspressoHubbard()
        atomsout = espressohub.run_qe_manually(
            atoms, 
            INPUT_SCF, 
            kpts=(2, 2, 1), 
            directory=worker_dir, 
            command_prefix=f"mpirun -np 1"
        )
        print('SCF Completed in {time:.2f} seconds'.format(time=time.time() - starttm))

        # self.atoms.calc.write_input(self.atoms)

        # import pprint
        # pprint.pprint(INPUT_SCF)
        # print('PARSED')
        # pprint.pprint(self.atoms.calc.parameters['input_data'])
        
        result = {
            'strain': strain,
            'id': f"CrI3_Biaxial_{strain:.4f}",
            'status': 'INIT'
        }

        # try:
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
                
        # except Exception as e:
        #     result['status'] = f"SCF_FAIL: {str(e)}"
        #     return result
        
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