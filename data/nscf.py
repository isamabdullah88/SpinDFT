import os
import subprocess
import copy
from ase.io.espresso import write_espresso_in
from .config import PSEUDO_DIR, PSEUDOS

class QEShellExecutor:
    """Handles the execution of Quantum ESPRESSO shell commands."""
    def __init__(self, wkdir, prefix):
        self.wkdir = wkdir
        self.prefix = prefix

    def run_pw(self, numcores):
        cmd = f"mpirun -np {numcores} pw.x < {self.prefix}_nscf.pwi > {self.prefix}_nscf.pwo"
        print(f"[{self.prefix}] Executing: {cmd}")
        
        try:
            subprocess.run(cmd, shell=True, cwd=self.wkdir, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[{self.prefix}] ERROR during Quantum ESPRESSO execution!")
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
            'disk_io': 'high',
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
            'diago_full_acc': True
        })
        
        nscf_in = os.path.join(self.wkdir, f"{self.prefix}_nscf.pwi")
        with open(nscf_in, 'w') as f:
            write_espresso_in(f, self.atoms, input_data=input_nscf, pseudopotentials=PSEUDOS, kpts=None)
        
        # Inject explicit K-points robustly
        with open(nscf_in, 'r') as f:
            content = f.read()
            
        kpts_idx = content.find("K_POINTS")
        if kpts_idx != -1:
            content = content[:kpts_idx]
            
        with open(nscf_in, 'w') as f:
            f.write(content)
            f.write("\n" + self.generate_explicit_kpts() + "\n")


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
        self.executor = QEShellExecutor(self.wkdir, self.prefix)

    def run(self, numcores):
        print(f"[{self.prefix}] Starting Quantum ESPRESSO Pipeline in {self.wkdir}...")
        print(f"[{self.prefix}] Step 1: Executing NSCF (Explicit K-point Mapping)...")
        
        # 1. Build the .pwi file
        self.builder.build()
        
        # 2. Execute pw.x
        self.executor.run_pw(numcores)
        
        print(f"[{self.prefix}] QE NSCF Pipeline completed successfully.")
        return self.atoms