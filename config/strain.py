import numpy as np
from .config import STRAIN_TYPE

def prep_strains(count=10):
    """
    Generates strained cell matrices and packages them for ProcessPoolExecutor.
    
    Args:
        bcell: The 3x3 numpy array or ASE Cell object from your relaxed structure.
        count: Total number of structures to generate.
        nworkers: Number of parallel workers (to assign worker_ids).
        
    Returns:
        List of tuples: (strain_value, cell_matrix, worker_id, strain_label)
    """
    tasks = []
    
    nuni = int(count * 1)
    niso = int(count * 1)
    nshr = int(count * 1)
    
    
    # Uniaxial Strain (Stretch X, keep Y and Z fixed)
    if STRAIN_TYPE == 'Uniaxial_X':
        for s in np.linspace(-0.15, 0.15, nuni):
            task = (s, 'Uniaxial_X')
            tasks.append(task)

    # Isotropic Strain (Uniform expansion/contraction)
    elif STRAIN_TYPE == 'Biaxial':
        for s in np.linspace(-0.12, 0.12, niso):
            task = (s, 'Biaxial')
            tasks.append(task)
        
    # Shear Strain (Tilt the lattice in the XY plane)
    elif STRAIN_TYPE == 'Shear_XY':
        for s in np.linspace(-0.15, 0.15, nshr):
            task = (s, 'Shear_XY')
            tasks.append(task)
        
    return tasks

if __name__ == "__main__":
    # Example usage with a dummy cell (a=6.895)
    a = 6.895
    example_cell = [[a, 0, 0], [-a/2, a*0.866, 0], [0, 0, 20]]
    
    tasks = prepare_cell_tasks(example_cell, count=100, nworkers=8)
    print(f"Generated {len(tasks)} cell tasks.")
    print(f"Example Task 0 (Strain {tasks[0][0]}): \n{tasks[0][1]}")