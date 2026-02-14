import numpy as np
from ase.io import read

def prep_strains(basecell, num_total=100, num_workers=4):
    """
    Generates strained cell matrices and packages them for ProcessPoolExecutor.
    
    Args:
        basecell: The 3x3 numpy array or ASE Cell object from your relaxed structure.
        num_total: Total number of structures to generate.
        num_workers: Number of parallel workers (to assign worker_ids).
        
    Returns:
        List of tuples: (strain_value, cell_matrix, worker_id, strain_label)
    """
    tasks = []
    # Ensure basecell is a numpy array for math operations
    basecell = np.array(basecell)
    
    n_iso = int(num_total * 1)
    n_uni = int(num_total * 0.3)
    n_shr = num_total - n_iso - n_uni
    
    # 1. Isotropic Strain (Uniform expansion/contraction)
    for i, s in enumerate(np.linspace(-0.05, 0.07, n_iso)):
        new_cell = basecell * (1 + s)
        
        task = (s, new_cell, i % num_workers, 'isotropic')
        tasks.append(task)
        
    """
    # 2. Uniaxial Strain (Stretch X, keep Y and Z fixed)
    for i, s in enumerate(np.linspace(-0.05, 0.05, n_uni)):
        new_cell = basecell.copy()
        new_cell[0] *= (1 + s) # Scale only the first lattice vector
        
        task = (s, new_cell, len(tasks) % num_workers, 'uniaxial_x')
        tasks.append(task)
        
    # 3. Shear Strain (Tilt the lattice in the XY plane)
    for i, s in enumerate(np.linspace(-0.03, 0.03, n_shr)):
        new_cell = basecell.copy()
        # Add a component of the second lattice vector to the first
        new_cell[0] += basecell[1] * s
        
        task = (s, new_cell, len(tasks) % num_workers, 'shear_xy')
        tasks.append(task)
    """
        
    return tasks

# --- Example of how to use this with your class ---
"""
def run_simulation(task_tuple):
    strain_val, new_cell, worker_id, label = task_tuple
    
    # Inside the worker process:
    # 1. Instantiate your class
    # 2. Apply the cell to a COPY of the atoms
    # my_obj = MyCrystalClass()
    # local_atoms = my_obj.atoms.copy()
    # local_atoms.set_cell(new_cell, scale_atoms=True)
    
    # 3. Run SCF...
"""

if __name__ == "__main__":
    # Example usage with a dummy cell (a=6.895)
    a = 6.895
    example_cell = [[a, 0, 0], [-a/2, a*0.866, 0], [0, 0, 20]]
    
    tasks = prepare_cell_tasks(example_cell, num_total=100, num_workers=8)
    print(f"Generated {len(tasks)} cell tasks.")
    print(f"Example Task 0 (Strain {tasks[0][0]}): \n{tasks[0][1]}")