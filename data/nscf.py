import os
import subprocess
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
        
        base_input = self.INPUT_SCF.copy()
        if 'control' not in base_input: base_input['control'] = {}
        if 'system' not in base_input: base_input['system'] = {}
        
        # --- STEP 1: NSCF Calculation ---
        print(f"[{self.prefix}] Step 1: Executing NSCF (Explicit K-point Mapping)...")
        input_nscf = base_input.copy()
        
        # Ensure disk_io='high' so the generated wavefunctions are saved for pw2wannier90
        input_nscf['control'].update({
            'calculation': 'nscf',
            'outdir': './',
            'prefix': self.prefix,
            'disk_io': 'high',
            'pseudo_dir': PSEUDO_DIR
        })
        
        # nosym and noinv MUST be forced here to map the charge density to the full grid
        input_nscf['system'].update({
            'nosym': True,
            'noinv': True,
            'nbnd': 80 if self.soc else 40
        })
        
        nscf_in = os.path.join(self.wkdir, f"{self.prefix}_nscf.pwi")
        with open(nscf_in, 'w') as f:
            write_espresso_in(f, self.atoms, input_data=input_nscf, pseudopotentials=PSEUDOS, kpts=None)
        
        # Inject explicit K-points
        with open(nscf_in, 'r') as f:
            lines = f.readlines()
        with open(nscf_in, 'w') as f:
            for line in lines:
                if not line.strip().upper().startswith('K_POINTS'):
                    f.write(line)
            f.write("\n" + self.generate_explicit_kpts() + "\n")
        
        # Run NSCF
        subprocess.run(
            f"mpirun -np {numcores} pw.x < {self.prefix}_nscf.pwi > {self.prefix}_nscf.pwo", 
            shell=True, cwd=self.wkdir, check=True
        )
        
        print(f"[{self.prefix}] QE NSCF Pipeline completed successfully.")
        return self.atoms