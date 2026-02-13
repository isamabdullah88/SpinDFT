from .scf import SCF
from .wannier90 import Wannier90

class TB2J:
    def __init__(self, worker_dir):
        self.worker_dir = worker_dir

        self.scf = SCF(worker_dir=worker_dir)
        self.wannier = Wannier90(INPUT_SCF=self.scf.INPUT_SCF, atoms=self.scf.atoms, worker_dir=worker_dir)

    def run(self, args):

        result = self.scf.run(args)
        return self.wannier.run((args[1], result))
        

