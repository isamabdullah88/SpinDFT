import os
import re
import logging

class FermiParser:
    """Parses Quantum ESPRESSO output files to extract physics properties."""
    def __init__(self, wkdir, prefix):
        self.wkdir = wkdir
        self.prefix = prefix
        self.logprefix = "[FermiParser]"
        self.logger = logging.getLogger("SpinDFT")

    def efermi(self):
        """
        Aggressively hunts for the Fermi Energy or Highest Occupied Level.
        Checks the local wkdir for NSCF logs, then looks one directory UP 
        for the SCF 'espresso.out' file.
        """

        pdir = os.path.dirname(os.path.abspath(self.wkdir))

        # Search order: 1) NSCF logs in current wkdir, 2) SCF log in parent directory
        logfiles = [
            os.path.join(self.wkdir, "nscf.pwo"),
            os.path.join(pdir, "espresso.pwo"),
        ]

        # All the weird ways Quantum ESPRESSO spells Fermi energy
        rpatterns = [
            r'the Fermi energy is\s+([\d\.\-]+)\s+ev',
            r'Fermi energy is\s+([\d\.\-]+)\s+eV',
            r'highest occupied, lowest unoccupied level \(ev\):\s+([\d\.\-]+)\s+[\d\.\-]+',
            r'highest occupied level \(ev\):\s+([\d\.\-]+)'
        ]

        for filepath in logfiles:
            self.logger.info(f"{self.logprefix} Checking {filepath} for Fermi energy...")
            if not os.path.exists(filepath):
                continue
                
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                    
                    for pattern in rpatterns:
                        match = re.search(pattern, content, re.IGNORECASE)
                        if match:
                            efermi = float(match.group(1))
                            self.logger.info(f"{self.logprefix} 🎯 Found Fermi Energy: {efermi} eV in {os.path.basename(filepath)}")
                            return efermi
            except Exception as e:
                self.logger.error(f"{self.logprefix} Error reading {os.path.basename(filepath)}: {e}")
                
        # Estimate Fermi energy if not found in logs (critical fallback)
        self.logger.warning(f"{self.logprefix} 🚨 CRITICAL WARNING: Could not find Fermi energy in any log file!")
        self.logger.warning(f"{self.logprefix} 🚨 Disentanglement windows might be wrong! Defaulting to -1.9857 eV.")
        
        return -1.9857