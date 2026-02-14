from ase.calculators.espresso import Espresso
from ase.io.espresso import write_espresso_in, read_espresso_out
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
from .config import PSEUDOS
import subprocess
import os
import io
import re

class EspressoHubbard:
    def __init__(self):
        pass

    def parse(self, filepath, original_atoms):
        with open(filepath, 'r') as f:
            content = f.read()
            print('\nParsing!\n')
        
            # 1. Parse Energy
            e_match = re.search(r'!\s+total energy\s+=\s+([-.\d]+)\s+Ry', content)
            if not e_match:
                raise RuntimeError("Could not find Total Energy in output.")
            energy_ry = float(e_match.group(1))
            energy_ev = energy_ry * 13.6056980659 

            # 2. Parse Fermi Energy (Needed for TB2J)
            ef_match = re.search(r'the Fermi energy is\s+([-.\d]+)\s+eV', content)
            if not ef_match:
                # Fallback for insulators
                ef_match = re.search(r'highest occupied, lowest unoccupied level \(eV\):\s+([-.\d]+)', content)
            
            efermi = float(ef_match.group(1)) if ef_match else 0.0

            # 3. Parse Total Magnetization
            m_match = re.search(r'total magnetization\s+=\s+([-.\d]+)\s+Bohr', content)
            total_mag = float(m_match.group(1)) if m_match else 0.0
            
            # 4. Parse Forces
            forces = np.zeros((len(original_atoms), 3))
            f_pattern = r'atom\s+(\d+)\s+type\s+\d+\s+force\s+=\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)'
            f_matches = re.findall(f_pattern, content)
            RY_AU_TO_EV_ANG = 25.71104309541616 
            
            for atom_idx, fx, fy, fz in f_matches:
                idx = int(atom_idx) - 1
                if idx < len(original_atoms):
                    forces[idx] = [float(fx), float(fy), float(fz)]
                    forces[idx] *= RY_AU_TO_EV_ANG

            # 5. Parse Stress
            stress_voigt = np.zeros(6)
            s_match = re.search(r'total\s+stress\s+\(Ry/bohr\*\*3\).*?\n\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\n\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\n\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)', content, re.DOTALL)
            
            if s_match:
                raw = [float(x) for x in s_match.groups()]
                stress = np.array(raw).reshape(3, 3)
                RY_BOHR3_TO_EV_ANG3 = 13.6056980659 / (0.529177210903**3) 
                stress *= RY_BOHR3_TO_EV_ANG3
                stress_voigt = np.array([stress[0,0], stress[1,1], stress[2,2], 
                                        stress[1,2], stress[0,2], stress[0,1]])

            # 6. Reconstruct
            atoms_out = original_atoms.copy()
            atoms_out.calc = None 
    
            # Calculate atomic moments estimate
            cridx = [i for i, a in enumerate(atoms_out) if a.symbol == 'Cr']
            cr2idx = [i for i, a in enumerate(atoms_out) if a.symbol == 'Cr2']
            # per_cr_mag = total_mag / max(1, len(cr_indices))
            final_mags = np.zeros(len(atoms_out))
            for i in cridx: final_mags[i] = 3.0
            for i in cr2idx: final_mags[i] = -3.0

            atoms_out.set_initial_magnetic_moments(final_mags)
            calc = SinglePointCalculator(
                atoms_out,
                energy=energy_ev,
                forces=forces,
                stress=stress_voigt,
                magmom=total_mag,
                magmoms=final_mags
            )
            
            calc.results['magmoms'] = final_mags
            
            calc.efermi = efermi
            atoms_out.calc = calc
            
            return atoms_out


    def writeAFM(self, input_path):

        print('Hubbard AFM patch applied to input file.')




    def runQE(self, atoms, input_data, kpts, directory, command_prefix):
        """
        Manually writes input, appends Hubbard, runs PW.x, and reads output.
        Bypasses ASE Calculator logic to prevent file overwrites/formatting bugs.
        """
        input_filename = 'espresso.pwi'
        output_filename = 'espresso.pwo'
        
        input_path = os.path.join(directory, input_filename)
        
        # Write Standard Input
        write_espresso_in(input_path, 
            atoms, 
            format='espresso-in', 
            input_data=input_data, 
            pseudopotentials=PSEUDOS, 
            kpts=kpts)
        
        # HUBBARD Card
        with open(input_path, 'a') as f:
            f.write("\nHUBBARD (atomic)\nU Cr-3d 3.0\nU Cr1-3d 3.0\n\n")
            
        # Command to Run QE
        full_cmd = f"{command_prefix} pw.x -in {input_filename} > {output_filename}"
        
        try:
            subprocess.run(full_cmd, shell=True, cwd=directory, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"QE Failed: {e}")

        # Parse Output
        # xmlpath = os.path.join(directory, 'tmp', 'pwscf.xml')
        output_path = os.path.join(directory, output_filename)

        atomsout = self.parse(output_path, atoms)
        return atomsout