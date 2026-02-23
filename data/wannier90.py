import os
import shutil
import subprocess
import sys
import re

class Wannier90:
    def __init__(self, INPUT_SCF, wkdir, outdir, kmesh=(2, 2, 1), soc=False):
        self.INPUT_SCF = INPUT_SCF
        self.wkdir = os.path.abspath(wkdir)
        self.outdir = os.path.abspath(outdir)
        self.prefix = 'pwscf'
        self.kmesh = kmesh
        self.soc = soc

    def parseconfig(self, atoms, seedname):
        """
        Generates the .win file required for Wannier90, matching the exact kmesh.
        """
        nwann = 56 if self.soc else 28
        nbnds = 80 if self.soc else 40 
        kx, ky, kz = self.kmesh
        
        win_content = (
            f"num_bands = {nbnds}\n"
            f"num_wann = {nwann}\n"
            "write_hr = .true.\n"   # Tells Wannier90 to save the Hamiltonian
            "write_xyz = .true.\n"  # Tells Wannier90 to save Wannier centers for TB2J
            "dis_num_iter = 100\n"
            "num_iter = 0\n"
            "iprint = 2\n"
        )
        if self.soc: win_content += "spinors = .true.\n"

        win_content += "\nbegin projections\nCr:d\nI:p\nend projections\n\n"
        win_content += f"mp_grid = {kx} {ky} {kz}\nbegin kpoints\n"
        
        for k in range(kz):
            for j in range(ky):
                for i in range(kx):
                    win_content += f"  {i/kx:.8f}  {j/ky:.8f}  {k/kz:.8f}\n"
        
        win_content += "end kpoints\n\nbegin unit_cell_cart\n"
        for vec in atoms.cell:
            win_content += f"  {vec[0]:.6f}  {vec[1]:.6f}  {vec[2]:.6f}\n"
            
        win_content += "end unit_cell_cart\n\nbegin atoms_cart\n"
        for atom in atoms:
            win_content += f"  {atom.symbol}  {atom.position[0]:.6f}  {atom.position[1]:.6f}  {atom.position[2]:.6f}\n"
        win_content += "end atoms_cart\n"
        
        win_path = os.path.join(self.wkdir, f"{seedname}.win")
        with open(win_path, 'w') as f:
            f.write(win_content)
        return win_path

    def runcmd(self, command, serial=False):
        """
        Executes shell commands using modern subprocess.run. 
        Safely strips MPI wrapping if a serial Fortran run is required.
        """
        if serial and "mpirun" in command:
            parts = command.split()
            executable = next((p for p in parts if ".x" in p or "wannier" in p), None)
            if executable:
                command = " ".join(parts[parts.index(executable):])
        
        print(f"[{self.prefix}] Executing: {command}")
        try:
            result = subprocess.run(
                command, shell=True, cwd=self.wkdir,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, check=True
            )
            if result.stdout.strip():
                print(result.stdout.strip())
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"[{self.prefix}] ERROR in command: {command}")
            print(f"[{self.prefix}] Details: {e.stderr.strip()}")
            raise RuntimeError(f"Command failed with code {e.returncode}")

    def get_fermi_energy(self):
        """
        Automatically parses the Quantum ESPRESSO output file to find the Fermi energy in eV.
        """
        nscf_out = os.path.join(self.wkdir, f"{self.prefix}_nscf.pwo")
        scf_out = os.path.join(self.wkdir, f"{self.prefix}_scf.pwo")
        outfile = nscf_out if os.path.exists(nscf_out) else scf_out
        
        if os.path.exists(outfile):
            with open(outfile, 'r') as f:
                content = f.read()
                patterns = [
                    r'[Tt]he Fermi energy is\s+([-+.\d]+)\s+ev',
                    r'Fermi energy\s+is\s+([-+.\d]+)\s+ev',
                    r'Highest occupied, lowest unoccupied level \(ev\):\s+([-+.\d]+)',
                    r'Highest occupied level \(ev\):\s+([-+.\d]+)'
                ]
                for pattern in patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match: return match.group(1)
                
        print(f"[{self.prefix}] WARNING: Could not auto-detect Fermi energy. Defaulting to 0.0 eV.")
        return "0.0"

    def run(self, atoms, numcores):
        result = {}
        
        # Determine the sequence of Wannier runs based on SOC or Collinear
        if self.soc:
            seednames = [self.prefix]
            spins = ['none']
        else:
            seednames = [f"{self.prefix}_up", f"{self.prefix}_down"]
            spins = ['up', 'down']

        print(f"[{self.prefix}] Starting Wannier90 extraction for components: {spins}")

        for seed, spin in zip(seednames, spins):
            print(f"\n--- Processing Spin Component: {spin.upper()} ---")
            
            print(f"[{seed}] Parsing Wannier90 configuration...")
            self.parseconfig(atoms, seedname=seed)
            
            print(f"[{seed}] Step 1: wannier90.x -pp")
            self.runcmd(f"wannier90.x -pp {seed}")

            print(f"[{seed}] Step 2: pw2wannier90.x (SERIAL)...")
            pw2wan_in = os.path.join(self.wkdir, f"{seed}.pw2wan")
            pw2wan_content = (
                "&inputpp\n"
                "  outdir = './'\n"
                f"  prefix = '{self.prefix}'\n"     # Must ALWAYS point to the main QE pwscf.save folder
                f"  seedname = '{seed}'\n"          # Differentiates the up and down Wannier files
                "  write_mmn = .true.\n"
                "  write_amn = .true.\n"
            )
            if self.soc:
                pw2wan_content += "  write_spn = .true.\n"
            else:
                pw2wan_content += f"  write_spn = .false.\n  spin_component = '{spin}'\n"
            pw2wan_content += "/\n"
            
            with open(pw2wan_in, 'w') as f:
                f.write(pw2wan_content)
                
            self.runcmd(f"pw2wannier90.x < {seed}.pw2wan", serial=True)

            print(f"[{seed}] Step 3: wannier90.x minimization (SERIAL)...")
            werr_file = os.path.join(self.wkdir, f"{seed}.werr")
            if os.path.exists(werr_file):
                os.remove(werr_file)
            self.runcmd(f"wannier90.x {seed}")

        # --- TB2J Final Exchange Calculation ---
        print(f"\n[{self.prefix}] Step 4: TB2J exchange calculation...")
        python_bin_dir = os.path.dirname(sys.executable)
        wann2j_exe = os.path.join(python_bin_dir, "wann2J.py")
        if not os.path.exists(wann2j_exe):
            wann2j_exe = shutil.which("wann2J.py")
            
        if wann2j_exe is None or not os.path.exists(wann2j_exe):
            raise RuntimeError(f"wann2J.py not found in {python_bin_dir}")
            
        kx, ky, kz = self.kmesh
        efermi = self.get_fermi_energy()
        print(f"[{self.prefix}] Detected Fermi Energy: {efermi} eV")
        
        if self.soc: 
            cmd = f"{wann2j_exe} --posfile {self.prefix}.win --prefix_spinor {self.prefix} --elements Cr --kmesh {kx} {ky} {kz} --spinor --efermi {efermi}"
        else:
            # We now correctly pass the distinct UP and DOWN prefixes
            cmd = f"{wann2j_exe} --posfile {self.prefix}_up.win --prefix_up {self.prefix}_up --prefix_down {self.prefix}_down --elements Cr --kmesh {kx} {ky} {kz} --efermi {efermi}"
            
        self.runcmd(cmd)
        
        # Check output in the Multibinit subfolder
        outxml = os.path.join(self.wkdir, 'TB2J_results', 'Multibinit', 'exchange.xml')
        
        if os.path.exists(outxml):
            final_xml_path = os.path.join(self.outdir, f"exchange_{self.prefix}.xml")
            shutil.copy(outxml, final_xml_path)
            result['status'] = 'SUCCESS'
            print(f"[{self.prefix}] Pipeline complete. exchange.xml saved to {final_xml_path}")
        else:
            result['status'] = 'TB2J_FAIL'
            print(f"[{self.prefix}] Pipeline failed: exchange.xml not found at {outxml}")
                
        return result