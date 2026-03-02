import os
import subprocess
import copy
import logging

from ase.io.espresso import write_espresso_in
from config import PSEUDO_DIR, PSEUDOS

logger = logging.getLogger("SpinDFT")
logprefix = "[NSCF]"

class QEShellExecutor:
    """Handles the execution of Quantum ESPRESSO shell commands."""
    def __init__(self, wkdir, prefix):
        self.wkdir = wkdir
        self.prefix = prefix

    def run_pw(self, numcores):
        cmd = f"mpirun -np {numcores} pw.x -npool 4 -ndiag 4 < {self.prefix}.pwi > {self.prefix}.pwo"
        logger.info(f"{logprefix} Executing: {cmd}")
        
        try:
            subprocess.run(cmd, shell=True, cwd=self.wkdir, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"{logprefix} ERROR during Quantum ESPRESSO execution!")
            raise RuntimeError(f"Command failed with code {e.returncode}")


class NSCFInputBuilder:
    """Manages the generation of Quantum ESPRESSO input files and k-point mapping."""
    def __init__(self, atoms, INPUT_SCF, wkdir, prefix, kmesh, soc, nbnds):
        self.atoms = atoms
        self.INPUT_SCF = INPUT_SCF
        self.wkdir = wkdir
        self.prefix = prefix
        self.kmesh = kmesh
        self.soc = soc
        self.nbnds = nbnds

    def generate_explicit_kpts(self):
        kx, ky, kz = self.kmesh
        total_kpts = kx * ky * kz
        weight = 1.0 / total_kpts
        
        kpts_str = f"K_POINTS crystal\n{total_kpts}\n"
        for k in range(kz):
            for j in range(ky):
                for i in range(kx):
                    kpts_str += f"  {i/kx:.8f}  {j/ky:.8f}  {k/kz:.8f}  {weight:.8f}\n"
        return kpts_str

    def build(self):
        # Use deepcopy to prevent mutating the global INPUT_SCF dictionary
        input_nscf = copy.deepcopy(self.INPUT_SCF)
        if 'control' not in input_nscf: input_nscf['control'] = {}
        if 'system' not in input_nscf: input_nscf['system'] = {}
        if 'electrons' not in input_nscf: input_nscf['electrons'] = {}
        
        # Ensure disk_io='high' so the generated wavefunctions are saved for pw2wannier90
        input_nscf['control'].update({
            'calculation': 'nscf',
            'outdir': './',
            'prefix': self.prefix,
            'disk_io': 'low',
            'pseudo_dir': PSEUDO_DIR
        })
        
        # Dynamic band allocation logic using the passed parameter
        target_nbnds = self.nbnds if self.nbnds is not None else (180 if self.soc else 140)
        
        # nosym and noinv MUST be forced here to map the charge density to the full grid
        input_nscf['system'].update({
            'nosym': True,
            'noinv': True,
            'nbnd': target_nbnds
        })
        
        # Force QE to converge empty bands with the exact same strict tolerance as occupied bands
        input_nscf['electrons'].update({
            'diago_full_acc': True,
            'diagonalization': 'cg'
        })
        
        nscfin = os.path.join(self.wkdir, "nscf.pwi")
        with open(nscfin, 'w') as f:
            write_espresso_in(f, self.atoms, input_data=input_nscf, pseudopotentials=PSEUDOS, kpts=None)
        
        # ------------------------------------------------------------------
        # CRITICAL FIX: Robustly inject positions, k-points, and cell params
        # ------------------------------------------------------------------
        with open(nscfin, 'r') as f:
            content = f.read()
            
        # ASE writes ATOMIC_POSITIONS, K_POINTS, and CELL_PARAMETERS at the end.
        # We find the earliest occurrence of any of these to slice the file cleanly
        # without deleting necessary data unexpectedly.
        cut_idx = len(content)
        for keyword in ["ATOMIC_POSITIONS", "K_POINTS", "CELL_PARAMETERS"]:
            idx = content.find(keyword)
            if idx != -1 and idx < cut_idx:
                cut_idx = idx
                
        content = content[:cut_idx]
            
        with open(nscfin, 'w') as f:
            # Write everything up to the cutoff (Namely: &CONTROL, &SYSTEM, &ELECTRONS, ATOMIC_SPECIES)
            f.write(content)
            
            # 1. Inject ATOMIC_POSITIONS explicitly
            f.write("ATOMIC_POSITIONS angstrom\n")
            for atom in self.atoms:
                f.write(f"  {atom.symbol:3s}  {atom.position[0]:.8f}  {atom.position[1]:.8f}  {atom.position[2]:.8f}\n")
            f.write("\n")
            
            # 2. Inject explicit K_POINTS
            f.write(self.generate_explicit_kpts())
            f.write("\n")
            
            # 3. Inject CELL_PARAMETERS explicitly to fix ibrav=0 crash
            f.write("CELL_PARAMETERS angstrom\n")
            for vec in self.atoms.cell:
                f.write(f"  {vec[0]:.8f}  {vec[1]:.8f}  {vec[2]:.8f}\n")
            f.write("\n")


class NSCF:
    """
    Main orchestrator for the Quantum ESPRESSO NSCF step.
    Bypasses ASE's calculation cache by forcefully writing inputs and using 
    subprocess. This guarantees the NSCF step executes cleanly with explicit k-points.
    """
    def __init__(self, atoms, INPUT_SCF, wkdir, kmesh=(2, 2, 1), soc=False, nbnds=None):
        self.atoms = atoms.copy()
        self.wkdir = os.path.abspath(wkdir)
        self.prefix = 'pwscf'
        
        # Initialize sub-components
        self.builder = NSCFInputBuilder(
            atoms=self.atoms, 
            INPUT_SCF=INPUT_SCF, 
            wkdir=self.wkdir, 
            prefix=self.prefix,
            kmesh=kmesh, 
            soc=soc, 
            nbnds=nbnds
        )
        self.executor = QEShellExecutor(self.wkdir, 'nscf')

    def run(self, numcores):
        logger.info(f"{logprefix} Starting NSCF in {self.wkdir}...")
        
        self.builder.build()
        
        self.executor.run_pw(numcores)
        
        logger.info(f"{logprefix} QE NSCF Pipeline completed successfully.")
        return self.atoms