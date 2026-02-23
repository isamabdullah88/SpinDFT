import os
import subprocess
import copy
from ase.io.espresso import write_espresso_in
from .config import PSEUDO_DIR, PSEUDOS

class NSCF:
    """
    Replaces the old NSCF class. 
    Bypasses ASE's calculation cache by forcefully writing inputs and using 
    subprocess. This guarantees the NSCF step executes cleanly with explicit k-points.
    """
    def __init__(self, atoms, INPUT_SCF, wkdir, kmesh=(2, 2, 1), soc=False):
        self.atoms = atoms.copy()
        self.soc = soc
        self.INPUT_SCF = INPUT_SCF
        self.wkdir = os.path.abspath(wkdir)
        self.prefix = 'pwscf'
        self.kmesh = kmesh

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

    def run(self, numcores):
        print(f"[{self.prefix}] Starting Quantum ESPRESSO Pipeline in {self.wkdir}...")
        
        # CRITICAL FIX: Use deepcopy to prevent mutating the global INPUT_SCF dictionary
        input_nscf = copy.deepcopy(self.INPUT_SCF)
        if 'control' not in input_nscf: input_nscf['control'] = {}
        if 'system' not in input_nscf: input_nscf['system'] = {}
        if 'electrons' not in input_nscf: input_nscf['electrons'] = {}
        
        # --- STEP 1: NSCF Calculation ---
        print(f"[{self.prefix}] Step 1: Executing NSCF (Explicit K-point Mapping)...")
        
        # Ensure disk_io='high' so the generated wavefunctions are saved for pw2wannier90
        input_nscf['control'].update({
            'calculation': 'nscf',
            'outdir': './',
            'prefix': self.prefix,
            'disk_io': 'high',
            'pseudo_dir': PSEUDO_DIR
        })
        
        # nosym and noinv MUST be forced here to map the charge density to the full grid
        # CRITICAL FIX: We calculate 140 bands (Buffer Bands) so the unconverged garbage at the top is ignored by Wannier90
        input_nscf['system'].update({
            'nosym': True,
            'noinv': True,
            'nbnd': 180 if self.soc else 140
        })
        
        # CRITICAL FIX: Force QE to converge empty bands with the exact same strict tolerance as occupied bands
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
        
        # Run NSCF
        subprocess.run(
            f"mpirun -np {numcores} pw.x < {self.prefix}_nscf.pwi > {self.prefix}_nscf.pwo", 
            shell=True, cwd=self.wkdir, check=True
        )
        
        print(f"[{self.prefix}] QE NSCF Pipeline completed successfully.")
        return self.atoms