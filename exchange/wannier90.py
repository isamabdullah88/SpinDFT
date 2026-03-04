import os
import logging

from config import ShellExecutor
from .fermi import FermiParser

class WannierFileManager:
    """Manages the generation of input files and post-processing fixes for Wannier90."""
    def __init__(self, wkdir, prefix, kmesh, soc, nscf_nbnds, wannier_nbnds):
        self.wkdir = wkdir
        self.prefix = prefix
        self.kmesh = kmesh
        self.soc = soc
        self.nscf_nbnds = nscf_nbnds
        self.wannier_nbnds = wannier_nbnds

        self.qeparser = FermiParser(wkdir, prefix)
        self.logprefix = "[WannierFileManager]"
        self.logger = logging.getLogger("SpinDFT")

    def write_win(self, atoms, seedname):
        nwann = 56 if self.soc else 28

        efermi = self.qeparser.efermi()
        dis_froz_min = float(efermi) - 9.0
        dis_froz_max = float(efermi) + 2.0
        dis_win_max  = float(efermi) + 10.0
        
        win_content = (
            f"num_bands = {self.nscf_nbnds}\n"
            f"num_wann = {nwann}\n"
            "\n"
            "! --- DISENTANGLEMENT ENERGY WINDOWS ---\n"
            f"dis_froz_min = {dis_froz_min:.4f}\n"
            f"dis_froz_max = {dis_froz_max:.4f}\n"
            f"dis_win_max = {dis_win_max:.4f}\n"
            "dis_num_iter = 200\n"  # MUST BE TURNED ON! (Default was 0)
            "dis_mix_ratio = 0.5\n" # Helps stabilize the math
            "\n"
            "! --- SPREAD MINIMIZATION ---\n"
            "num_iter = 200\n"      # Now we actually minimize the Wannier functions
            "write_hr = .true.\n"    
            "write_xyz = .true.\n"   
            "use_ws_distance = .true.\n" 
            "iprint = 2\n\n"
        )
        
        # 3. Tell Wannier90 to ignore the garbage bands
        # if self.nscf_nbnds > self.wannier_nbnds:
        #     win_content += f"exclude_bands = {self.wannier_nbnds + 1}-{self.nscf_nbnds}\n"

        # win_content += (
        #     "write_hr = .true.\n"    
        #     "write_xyz = .true.\n"   
        #     "use_ws_distance = .true.\n" 
        #     "dis_num_iter = 0\n"     
        #     "num_iter = 0\n"         
        #     "iprint = 2\n"
        # )

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

    def write_pw2wan(self, seedname, spin):
        pw2wan_content = (
            f"&inputpp\n  outdir = './'\n  prefix = '{self.prefix}'\n  seedname = '{seedname}'\n"
            f"  write_mmn = .true.\n  write_amn = .true.\n"
        )
        pw2wan_content += "  write_spn = .true.\n/\n" if self.soc else f"  write_spn = .false.\n  spin_component = '{spin}'\n/\n"
        
        with open(os.path.join(self.wkdir, f"{seedname}.pw2wan"), 'w') as f: 
            f.write(pw2wan_content)

    def fix_wannier_centers(self, seedname):
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
                
                for a in atoms:
                    dist_xy = ((x - a['x'])**2 + (y - a['y'])**2)**0.5
                    if dist_xy < min_dist:
                        min_dist = dist_xy
                        closest_atom = a

                if closest_atom and min_dist < 1.0: 
                    new_lines.append(f"X {x:15.8f} {y:15.8f} {closest_atom['z']:15.8f}\n")
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        with open(xyz_file, 'w') as f:
            f.writelines(new_lines)
        self.logger.info(f"{self.logprefix} Automatically corrected kz=1 Z-shift bug in {seedname}_centres.xyz")





class Wannier90:
    """
    Main orchestrator for the Wannier90 to TB2J pipeline.
    Coordinates the file manager, executor, parser, and TB2J runner.
    """
    def __init__(self, wkdir, kmesh, soc, nscf_nbnds, wannier_nbnds):
        self.wkdir = wkdir
        self.prefix = 'pwscf'
        self.kmesh = kmesh
        self.soc = soc
        self.nscf_nbnds = nscf_nbnds
        self.wannier_nbnds = wannier_nbnds
        
        # Initialize Sub-Components
        self.executor = ShellExecutor(self.wkdir, "[Wannier90]")
        self.file_manager = WannierFileManager(self.wkdir, self.prefix, self.kmesh, self.soc,
                                               self.nscf_nbnds, self.wannier_nbnds)
        # self.qe_parser = FermiParser(self.wkdir, self.prefix)
        # self.tb2j = TB2JExchange(self.wkdir, self.prefix, self.kmesh, self.soc, self.executor)

        self.logprefix = "[Wannier90]"
        self.logger = logging.getLogger("SpinDFT")

    def run(self, atoms, numcores):
        self.logger.info(f"{self.logprefix} Starting Wannier90 pipeline in {self.wkdir}...")
        seednames, spins = ([self.prefix], ['none']) if self.soc else ([f"{self.prefix}_up", f"{self.prefix}_down"], ['up', 'down'])
        self.logger.info(f"{self.logprefix} Starting Wannier90 extraction for components: {spins} with {numcores} cores.")

        for seed, spin in zip(seednames, spins):
            self.logger.info(f"{self.logprefix} --- Processing Spin Component: {spin.upper()} ---")
            
            # Step 1: Pre-processing
            self.file_manager.write_win(atoms, seedname=seed)
            self.executor.runcmd(f"wannier90.x -pp {seed}", serial=True)

            # Step 2: Extract Matrices
            self.file_manager.write_pw2wan(seed, spin)
            self.executor.runcmd(f"mpirun -np {numcores} pw2wannier90.x -npool 4 -ndiag 4 < {seed}.pw2wan", serial=False)

            # Step 3: Wannierization & Bug Fix
            # werr_file = os.path.join(self.wkdir, f"{seed}.werr")
            # if os.path.exists(werr_file): os.remove(werr_file)
            
            self.executor.runcmd(f"wannier90.x {seed}", serial=True)
            self.file_manager.fix_wannier_centers(seed)

        # # Step 4: Calculate Exchange Interactions
        # efermi = self.qe_parser.efermi()
        # return self.tb2j.calculate(efermi)