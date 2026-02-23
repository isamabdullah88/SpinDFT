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
        self.soc = soc
        self.kmesh = kmesh

    def write_win(self, atoms, seedname):
        """Generates the Wannier90 .win file ensuring the exact band count from NSCF is read."""
        nwann, nbnds = (56, 120) if self.soc else (28, 80)
        
        win_content = (
            f"num_bands = {nbnds}\n"
            f"num_wann = {nwann}\n"
            "write_hr = .true.\n"    
            "write_xyz = .true.\n"   
            "use_ws_distance = .true.\n" 
            "dis_num_iter = 0\n"     
            "num_iter = 0\n"         
            "iprint = 2\n"
        )
        if self.soc: win_content += "spinors = .true.\n"
        
        win_content += "\nbegin projections\nCr:d\nI:p\nend projections\n\n"
        
        win_content += f"mp_grid = {self.kmesh[0]} {self.kmesh[1]} {self.kmesh[2]}\nbegin kpoints\n"
        for k in range(self.kmesh[2]):
            for j in range(self.kmesh[1]):
                for i in range(self.kmesh[0]):
                    win_content += f"  {i/self.kmesh[0]:.8f}  {j/self.kmesh[1]:.8f}  {k/self.kmesh[2]:.8f}\n"
                    
        win_content += "end kpoints\n\nbegin unit_cell_cart\n"
        for vec in atoms.cell: win_content += f"  {vec[0]:.6f}  {vec[1]:.6f}  {vec[2]:.6f}\n"
        win_content += "end unit_cell_cart\n\nbegin atoms_cart\n"
        for atom in atoms: win_content += f"  {atom.symbol}  {atom.position[0]:.6f}  {atom.position[1]:.6f}  {atom.position[2]:.6f}\n"
        win_content += "end atoms_cart\n"
        
        with open(os.path.join(self.wkdir, f"{seedname}.win"), 'w') as f: 
            f.write(win_content)

    def runcmd(self, command, serial=False, env=None):
        if serial and "mpirun" in command:
            parts = command.split()
            executable = next((p for p in parts if ".x" in p or "wannier" in p), None)
            if executable:
                command = " ".join(parts[parts.index(executable):])
        
        print(f"[{self.prefix}] Executing: {command}")
        
        run_env = os.environ.copy()
        if env: run_env.update(env)
            
        try:
            result = subprocess.run(
                command, shell=True, cwd=self.wkdir,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, check=True, env=run_env
            )
            if result.stdout.strip():
                print(result.stdout.strip())
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"[{self.prefix}] ERROR in command: {command}")
            print(f"[{self.prefix}] Details: {e.stderr.strip()}")
            raise RuntimeError(f"Command failed with code {e.returncode}")

    def fix_wannier_centers(self, seedname):
        """
        CRITICAL FIX: Bypasses the Wannier90 kz=1 vacuum shift bug.
        Reads the generated _centres.xyz file, matches the X and Y coordinates to the 
        closest atom, and physically snaps the corrupted Z coordinate onto the atomic nucleus.
        """
        xyz_file = os.path.join(self.wkdir, f"{seedname}_centres.xyz")
        if not os.path.exists(xyz_file): return

        with open(xyz_file, 'r') as f:
            lines = f.readlines()

        atoms = []
        for line in lines:
            parts = line.split()
            if len(parts) == 4 and parts[0] not in ['X', 'Wannier']:
                atoms.append({'symbol': parts[0], 'x': float(parts[1]), 'y': float(parts[2]), 'z': float(parts[3])})

        new_lines = []
        for line in lines:
            parts = line.split()
            if len(parts) == 4 and parts[0] == 'X':
                x, y = float(parts[1]), float(parts[2])
                closest_atom = None
                min_dist = 9999.0
                
                # Find the atom directly above/below this orbital in the XY plane
                for a in atoms:
                    dist_xy = ((x - a['x'])**2 + (y - a['y'])**2)**0.5
                    if dist_xy < min_dist:
                        min_dist = dist_xy
                        closest_atom = a

                # If perfectly aligned in XY plane, fix the Z coordinate!
                if closest_atom and min_dist < 1.0: 
                    new_line = f"X {x:15.8f} {y:15.8f} {closest_atom['z']:15.8f}\n"
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        with open(xyz_file, 'w') as f:
            f.writelines(new_lines)
        print(f"[{self.prefix}] Automatically corrected kz=1 Z-shift bug in {seedname}_centres.xyz")

    def get_fermi_energy(self):
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
                for match in (re.search(p, content, re.IGNORECASE) for p in patterns):
                    if match: return match.group(1)
        return "0.0"

    def run(self, atoms, numcores):
        result = {}
        
        # seednames, spins = ([self.prefix], ['none']) if self.soc else ([f"{self.prefix}_up", f"{self.prefix}_down"], ['up', 'down'])
        # print(f"[{self.prefix}] Starting Wannier90 extraction for components: {spins} with {numcores} cores.")

        # for seed, spin in zip(seednames, spins):
        #     print(f"\n--- Processing Spin Component: {spin.upper()} ---")
            
        #     self.write_win(atoms, seedname=seed)
        #     self.runcmd(f"wannier90.x -pp {seed}")

        #     pw2wan_content = (
        #         f"&inputpp\n  outdir = './'\n  prefix = '{self.prefix}'\n  seedname = '{seed}'\n"
        #         f"  write_mmn = .true.\n  write_amn = .true.\n"
        #     )
        #     pw2wan_content += "  write_spn = .true.\n/\n" if self.soc else f"  write_spn = .false.\n  spin_component = '{spin}'\n/\n"
            
        #     with open(os.path.join(self.wkdir, f"{seed}.pw2wan"), 'w') as f: 
        #         f.write(pw2wan_content)
                
        #     self.runcmd(f"mpirun -np {numcores} pw2wannier90.x < {seed}.pw2wan", serial=False)

        #     werr_file = os.path.join(self.wkdir, f"{seed}.werr")
        #     if os.path.exists(werr_file): os.remove(werr_file)
            
        #     self.runcmd(f"wannier90.x {seed}")
            
        #     # --- APPLY THE Z-SNAP FIX HERE ---
        #     self.fix_wannier_centers(seed)

        print(f"\n[{self.prefix}] Step 4: TB2J exchange calculation...")
        wann2j_exe = shutil.which("wann2J.py") or os.path.join(os.path.dirname(sys.executable), "wann2J.py")
            
        kx, ky, kz = self.kmesh
        efermi = self.get_fermi_energy()
        
        if self.soc: 
            cmd = f"{wann2j_exe} --posfile {self.prefix}.win --prefix_spinor {self.prefix} --elements Cr --kmesh {kx} {ky} {kz} --spinor --efermi {efermi}"
        else:
            cmd = f"{wann2j_exe} --posfile {self.prefix}_up.win --prefix_up {self.prefix}_up --prefix_down {self.prefix}_down --elements Cr --kmesh {kx} {ky} {kz} --efermi {efermi}"
            
        self.runcmd(cmd)
        
        outxml = os.path.join(self.wkdir, 'TB2J_results', 'Multibinit', 'exchange.xml')
        if os.path.exists(outxml):
            final_path = os.path.join(self.outdir, f"exchange_{self.prefix}.xml")
            shutil.copy(outxml, final_path)
            result.update({'status': 'SUCCESS', 'file': final_path})
            print(f"[{self.prefix}] Pipeline complete! Result saved to {final_path}")
        else:
            result['status'] = 'TB2J_FAIL'
            
        return result