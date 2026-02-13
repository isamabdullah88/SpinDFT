
import os
import shutil

class Wannier90:
    def __init__(self, INPUT_SCF, atoms, worker_dir):
        self.INPUT_SCF = INPUT_SCF
        self.atoms = atoms
        self.worker_dir = worker_dir
        self.prefix = 'espresso'

    def parseconfig(self):

        win_content = f"""
            num_bands = 40
            num_wann = 28
            dis_num_iter = 100
            num_iter = 0
            iprint = 2
            begin projections
            Cr:d
            I:p
            end projections
            mp_grid = 4 4 1
            begin unit_cell_cart
            {self.atoms.cell[0][0]} {self.atoms.cell[0][1]} {self.atoms.cell[0][2]}
            {self.atoms.cell[1][0]} {self.atoms.cell[1][1]} {self.atoms.cell[1][2]}
            {self.atoms.cell[2][0]} {self.atoms.cell[2][1]} {self.atoms.cell[2][2]}
            end unit_cell_cart
            begin atoms_cart
        """
        for atom in self.atoms:
            win_content += f"{atom.symbol} {atom.position[0]} {atom.position[1]} {atom.position[2]}\n"

        win_content += "end atoms_cart\n"
        
        with open(os.path.join(self.worker_dir, f"{self.prefix}.win"), 'w') as f:
            f.write(win_content)

    def runcmd(self, command):
        import subprocess
        process = subprocess.Popen(command, shell=True, cwd=self.worker_dir,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"Command '{command}' failed with error: {stderr.decode()}")
        return stdout.decode()

    def run(self, args):
        numcores, result = args

        # NSCF for bandstructure
        INPUT_NSCF = self.INPUT_SCF.copy()
        INPUT_NSCF['control']['calculation'] = 'nscf'
        INPUT_NSCF['system']['nbnd'] = 40 
        INPUT_NSCF['system']['nosym'] = True
        
        self.atoms.calc.set(input_data=INPUT_NSCF)
        try:
            self.atoms.get_potential_energy()
        except Exception as e:
            result['status'] = f"NSCF_FAIL: {str(e)}"
            return result

        # Wannier90 Config
        self.parseconfig()

        # 4. Run Wannier Pipeline
        try:
            self.runcmd(f"mpirun -np {numcores} wannier90.x -pp {self.prefix}")
            
            with open(os.path.join(self.worker_dir, f"{self.prefix}.pw2wan"), 'w') as f:
                f.write(f"&inputpp outdir='./tmp' prefix='{self.prefix}' write_mmn=.true. write_amn=.true. /")
                f.write(f"&inputpp outdir='./tmp' prefix='{self.prefix}' write_mmn=.true. write_amn=.true. /")
                
            self.runcmd(f"mpirun -np {numcores} pw2wannier90.x < {self.prefix}.pw2wan")
            self.runcmd(f"mpirun -np {numcores} wannier90.x {self.prefix}")
            
            # 5. Run TB2J
            # ADDED: --np flag to make TB2J utilize the assigned cores
            self.runcmd(f"mpirun -np {numcores} wannier2J.py --posfile {self.prefix}.win --prefix {self.prefix} --elements Cr --kmesh 4 4 1 --np {numcores}")
            
            # 6. Check Results
            result_xml = os.path.join(self.worker_dir, 'TB2J_results', 'exchange.xml')
            if os.path.exists(result_xml):
                os.makedirs("./results", exist_ok=True)
                shutil.copy(result_xml, f"./results/exchange_{eps:.4f}.xml")
                result['status'] = 'SUCCESS'
            else:
                result['status'] = 'TB2J_FAIL'
                
        except Exception as e:
            result['status'] = f"W90_FAIL: {str(e)}"
        finally:
            shutil.rmtree(os.path.join(self.worker_dir, 'tmp'), ignore_errors=True)
            
        return result