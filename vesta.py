from ase.db import connect
from ase.io import write

# Get the last sample
with connect('cri3_dataset.db') as db:
    atoms = db.get_atoms(id=1) # Get first row

write('cri3_sample.xsf', atoms)