import subprocess
import platform
import os
import re
import numpy as np
from ase.io.espresso import write_espresso_in
from ase.calculators.singlepoint import SinglePointCalculator
from logging import getLogger

from .config import INPUT_SCF, PSEUDOS, KPTS

class EspressoHubbard:
    # ---------------------------------------------------------------------------------------------
    def __init__(self, phase='FM', cores_per_job=6):
        self.phase = phase
        self.cores_per_job = cores_per_job

        self.logprefix = "[EspressoHubbard] "
        self.logger = getLogger(__name__)


    # ---------------------------------------------------------------------------------------------
    def parseatoms(self, content, atoms):
        # Parse the atomic positions and cell from the QE input file content. This is necessary 
        # because the final geometry may differ from the initial structure, and we want to ensure
        # our dataset reflects the final final state. We look for the last occurrence of 
        # 'CELL_PARAMETERS' and 'ATOMIC_POSITIONS' to get the final geometry. We also handle 
        # different units (angstrom, bohr, crystal, alat) that QE might use in its output.

        atomsout = atoms.copy()
        
        try:
            # A. Parse Final Cell (if CELL_PARAMETERS is present in the output)
            cell_idx = content.rfind('CELL_PARAMETERS')
            if cell_idx != -1:
                block = content[cell_idx:].split('\n')
                unit_match = re.search(r'\((.*?)\)', block[0])
                unit = unit_match.group(1).strip().lower() if unit_match else 'angstrom'
                
                # Extract the 3x3 matrix lines
                v1 = [float(x) for x in block[1].split()[:3]]
                v2 = [float(x) for x in block[2].split()[:3]]
                v3 = [float(x) for x in block[3].split()[:3]]
                cell = np.array([v1, v2, v3])
                
                # Convert to Angstroms if needed
                if unit == 'bohr':
                    cell *= 0.529177210903
                elif unit == 'alat':
                    alat_match = re.search(r'celldm\(1\)=\s*([-.\d]+)', content)
                    if alat_match:
                        cell *= float(alat_match.group(1)) * 0.529177210903
                
                atomsout.set_cell(cell)

            # B. Parse Final Positions (if ATOMIC_POSITIONS is present)
            pos_idx = content.rfind('ATOMIC_POSITIONS')
            if pos_idx != -1:
                block = content[pos_idx:].split('\n')
                unit_match = re.search(r'\((.*?)\)', block[0])
                unit = unit_match.group(1).strip().lower() if unit_match else 'crystal'
                
                positions = []
                for line in block[1:]:
                    parts = line.split()
                    # An atom line has >=4 parts and starts with a chemical symbol
                    if len(parts) >= 4 and re.match(r'^[A-Za-z]+', parts[0]):
                        try:
                            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                        except ValueError:
                            break # End of atomic block
                    else:
                        break # End of atomic block
                        
                # Only apply if we extracted the correct number of atoms
                if len(positions) == len(atomsout):
                    positions = np.array(positions)
                    if unit == 'angstrom':
                        atomsout.set_positions(positions)
                    elif unit == 'bohr':
                        atomsout.set_positions(positions * 0.529177210903)
                    elif unit == 'crystal':
                        atomsout.set_scaled_positions(positions)
                    elif unit == 'alat':
                        alat_match = re.search(r'celldm\(1\)=\s*([-.\d]+)', content)
                        if alat_match:
                            atomsout.set_positions(positions * float(alat_match.group(1)) * 0.529177210903)
                            
        except Exception as e:
            self.logger.fatal(f"{self.logprefix}Atomic position parsing failed ({e})")

        return atomsout


    # ---------------------------------------------------------------------------------------------
    def parse(self, pwipath, pwopath, atoms):

        if not os.path.exists(pwipath):
            raise FileNotFoundError(f"Input file not found: {pwipath}")

        if not os.path.exists(pwopath):
            raise FileNotFoundError(f"Output file not found: {pwopath}")
        
        with open(pwipath, 'r') as f:
            content = f.read()

            # Parse geometry
            atomsout = self.parseatoms(content, atoms)

        with open(pwopath, 'r') as f:
            content = f.read()

            # Parse Energy
            e_match = re.search(r'!\s+total energy\s+=\s+([-.\d]+)\s+Ry', content)
            if not e_match:
                raise RuntimeError("Could not find Total Energy in output.")
            energy_ry = float(e_match.group(1))
            energy_ev = energy_ry * 13.6056980659 

            # Parse Fermi Energy (Needed for TB2J)
            ef_match = re.search(r'the Fermi energy is\s+([-.\d]+)\s+eV', content)
            if not ef_match:
                # Fallback for insulators
                ef_match = re.search(r'highest occupied, lowest unoccupied level \(eV\):\s+([-.\d]+)', content)
            
            efermi = float(ef_match.group(1)) if ef_match else 0.0

            # Total Magnetization
            m_match = re.search(r'total magnetization\s+=\s+([-.\d]+)\s+Bohr', content)
            total_mag = float(m_match.group(1)) if m_match else 0.0
            
            # Forces
            forces = np.zeros((len(atomsout), 3))
            f_pattern = r'atom\s+(\d+)\s+type\s+\d+\s+force\s+=\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)'
            f_matches = re.findall(f_pattern, content)
            RY_AU_TO_EV_ANG = 25.71104309541616 
            
            for atom_idx, fx, fy, fz in f_matches:
                idx = int(atom_idx) - 1
                if idx < len(atomsout):
                    forces[idx] = [float(fx), float(fy), float(fz)]
                    forces[idx] *= RY_AU_TO_EV_ANG

            # Stress
            stress_voigt = np.zeros(6)
            s_match = re.search(r'total\s+stress\s+\(Ry/bohr\*\*3\).*?\n\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\n\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\n\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)', content, re.DOTALL)
            
            if s_match:
                raw = [float(x) for x in s_match.groups()]
                stress = np.array(raw).reshape(3, 3)
                RY_BOHR3_TO_EV_ANG3 = 13.6056980659 / (0.529177210903**3) 
                stress *= RY_BOHR3_TO_EV_ANG3
                stress_voigt = np.array([stress[0,0], stress[1,1], stress[2,2], 
                                        stress[1,2], stress[0,2], stress[0,1]])

            # Reconstruct
            # atomsout = atoms.copy()
            atomsout.calc = None 
    
            # Calculate atomic moments estimate
            cridx = [i for i, a in enumerate(atomsout) if a.symbol == 'Cr']
            cr2idx = [i for i, a in enumerate(atomsout) if a.symbol == 'Cr2']
            # per_cr_mag = total_mag / max(1, len(cr_indices))
            final_mags = np.zeros(len(atomsout))
            for i in cridx: final_mags[i] = 3.0
            for i in cr2idx: final_mags[i] = -3.0

            atomsout.set_initial_magnetic_moments(final_mags)
            calc = SinglePointCalculator(
                atomsout,
                energy=energy_ev,
                forces=forces,
                stress=stress_voigt,
                magmom=total_mag,
                magmoms=final_mags
            )
            
            calc.results['magmoms'] = final_mags
            
            calc.efermi = efermi
            atomsout.calc = calc
            
            return atomsout


    # ---------------------------------------------------------------------------------------------
    def runQE(self, atoms, input_data, kpts, directory):
        """
        Manually writes input, appends Hubbard, runs PW.x, and reads output.
        Bypasses ASE Calculator logic to prevent file overwrites/formatting bugs.
        """
        inname = 'espresso.pwi'
        outname = 'espresso.pwo'
        
        inputpath = os.path.join(directory, inname)
        
        # Write Standard Input
        write_espresso_in(inputpath, 
            atoms, 
            format='espresso-in', 
            input_data=input_data, 
            pseudopotentials=PSEUDOS, 
            kpts=kpts)
        
        # HUBBARD Card
        with open(inputpath, 'a') as f:
            if self.phase == 'AFM':
                f.write("\nHUBBARD (atomic)\nU Cr-3d 3.0\nU Cr1-3d 3.0\n\n")
            else:
                f.write("\nHUBBARD (atomic)\nU Cr-3d 3.0\n\n")
            
        
        if platform.system() == "Darwin":
            cmd = f"mpirun -np {self.cores_per_job} pw.x -in {inname} > {outname}"
        else:
            cmd = f"mpirun --bind-to core -np {self.cores_per_job} pw.x -in {inname} > {outname}"
        
        try:
            subprocess.run(cmd, shell=True, cwd=directory, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"QE Failed: {e}")

        # Parse Output
        pwipath = os.path.join(directory, inname)
        pwopath = os.path.join(directory, outname)

        atomsout = self.parse(pwipath, pwopath, atoms)
        
        return atomsout
    

# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    from ase.db import connect
    from .crI3 import CrI3

    phase = 'FM'
    stntype = 'Shear_XY'
    wkdir = f"./DataSets/HPC/Kpts-10x10-{stntype}/{phase}"
    dbpath = os.path.join(wkdir, f"Kpts-10x10-{stntype}-{phase}.db")

    strains = np.linspace(-0.15, 0.15, 21)
    
    hubbardcalc = EspressoHubbard(phase=phase)
    with connect(dbpath) as db:
        for i, strain in enumerate(strains):
            print(f'Writing to db: {i} of {len(strains)}')
            pwopath = os.path.join(wkdir, f"Strain_{stntype}_{strain:.4f}", "espresso.pwo")
            pwipath = os.path.join(wkdir, f"Strain_{stntype}_{strain:.4f}", "espresso.pwi")

            if (not os.path.exists(pwopath)) or (not os.path.exists(pwipath)):
                print(f"Input/Output file not found for strain {strain:.4f} at {pwopath}/{pwipath}. Skipping...")
                continue


            crI3 = CrI3()
            atoms = crI3.strain_atoms(stntype=stntype, stnvalue=strain)

            atomsout = hubbardcalc.parse(pwipath, pwopath, atoms)

            energy = atomsout.get_potential_energy()
            moms = atomsout.get_magnetic_moments()
            forces = atomsout.get_forces()
            stress = atomsout.get_stress()

            result = {
                'strain': strain,
                'id': f"CrI3_{stntype}_{strain:.4f}",
                'status': 'SUCCESS',
                'energy': energy,
                'mag_moments': moms,
                'forces': forces,
                'stress': stress,
                'atoms': atomsout
            }

            db.write(
                result['atoms'],
                key_value_pairs={
                    'strain_value': result['strain'],
                    'dataid': result['id'],
                    'pipeline_status': result['status']
                },
                data={
                    'mag_moments': result['mag_moments'],
                    'forces': result['forces'],
                    'stress': result['stress'],
                    'scf_parameters': INPUT_SCF,
                    'kpoints': KPTS
                }
            )