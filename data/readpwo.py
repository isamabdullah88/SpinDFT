
import os
import re
import numpy as np
from ase.io.espresso import read_espresso_out

directory = "./DataSets/CrI3/_strain_0.0000"
output_filename = "espresso.pwo"
output_path = os.path.join(directory, output_filename)

# 4. READ & SANITIZE
with open(output_path, 'r') as f:
    content = f.read()
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
    forces = np.zeros((8, 3))
    f_pattern = r'atom\s+(\d+)\s+type\s+\d+\s+force\s+=\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)'
    f_matches = re.findall(f_pattern, content)
    RY_AU_TO_EV_ANG = 25.71104309541616 
    
    for atom_idx, fx, fy, fz in f_matches:
        idx = int(atom_idx) - 1
        if idx < 8:
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


    # # 6. Reconstruct
    # atoms_out = original_atoms.copy()
    # calc = SinglePointCalculator(
    #     atoms_out, 
    #     energy=energy_ev, 
    #     forces=forces, 
    #     stress=stress_voigt,
    #     magmom=total_mag
    # )
    # # Attach efermi as a custom attribute
    # calc.efermi = efermi
    # atoms_out.calc = calc
    
    # # Hack for initial magmoms array (Total distributed over Cr)
    # cr_indices = [i for i, a in enumerate(atoms_out) if a.symbol == 'Cr']
    # per_cr_mag = total_mag / max(1, len(cr_indices))
    # final_mags = np.zeros(len(atoms_out))
    # for i in cr_indices: final_mags[i] = per_cr_mag
    # atoms_out.set_initial_magnetic_moments(final_mags)
    