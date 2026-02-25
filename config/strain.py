import numpy as np
from ase.io import read

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
    # Ensure bcell is a numpy array for math operations
    # bcell = np.array(bcell)
    
    niso = int(count * 1)
    nuni = int(count * 1)
    # nshr = count - niso - nuni
    nshr = int(count * 1)
    
    # Isotropic Strain (Uniform expansion/contraction)
    # for i, s in enumerate(np.linspace(-0.08, 0.1, niso)):
    #     ncell = bcell * (1 + s)
        
    #     task = (s, ncell, i % nworkers, 'isotropic')
    #     tasks.append(task)
        
    # Uniaxial Strain (Stretch X, keep Y and Z fixed)
    # for i, s in enumerate(np.linspace(-0.15, -0.01, nuni)):
    for i, s in enumerate(np.linspace(-0.15, 0.15, nuni)):
        # ncell = bcell.copy()
        # ncell[0] *= (1 + s) # Scale only the first lattice vector
        
        task = (s, 'Uniaxial_X')
        tasks.append(task)
        
    # Shear Strain (Tilt the lattice in the XY plane)
    # for i, s in enumerate(np.linspace(-0.03, 0.03, nshr)):
    #     ncell = bcell.copy()
    #     # Add a component of the second lattice vector to the first
    #     ncell[0] += bcell[1] * s
        
    #     task = (s, ncell, len(tasks) % nworkers, 'shear_xy')
    #     tasks.append(task)

    # for i, (u, s) in enumerate(zip(np.linspace(-0.30, 0.30, nuni), np.linspace(-0.30, 0.30, nshr))):
    #     ncell = bcell.copy()
    #     ncell[0] *= (1 + u) # Scale only the first lattice vector

    #     ncell[0] += bcell[1] * s # Shear

    #     task = (s, ncell, len(tasks) % nworkers, 'uniaxial_x-shear_xy')
    #     tasks.append(task)
        
    return tasks

if __name__ == "__main__":
    # Example usage with a dummy cell (a=6.895)
    a = 6.895
    example_cell = [[a, 0, 0], [-a/2, a*0.866, 0], [0, 0, 20]]
    
    tasks = prepare_cell_tasks(example_cell, count=100, nworkers=8)
    print(f"Generated {len(tasks)} cell tasks.")
    print(f"Example Task 0 (Strain {tasks[0][0]}): \n{tasks[0][1]}")