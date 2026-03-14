import os
import shutil
from glob import glob
import logging

class WorkspaceManager:
    """Manages the creation of directories and injection of charge densities."""
    def __init__(self, wkdir):
        self.bwkdir = os.path.abspath(wkdir)
        self.wkdir = ""
        self.twkdir = ""
        self.pwscfdir = ""

        if not os.path.exists(self.bwkdir):
            os.makedirs(self.bwkdir, exist_ok=True)

        self.logprefix = "[WorkspaceManager]"

        self.logger = logging.getLogger("SpinDFT")

    def setwkdir(self, strain, stntype):
        """Creates temporary and output directories for a specific strain."""
        self.wkdir = os.path.join(self.bwkdir, f"Strain_{stntype}_{strain:.4f}")
        self.tmpdir = os.path.join(self.wkdir, "tmp")
        self.pwscfdir = os.path.join(self.tmpdir, "pwscf.save")

    def cleanscf(self):
        """Cleans up .hdf5 wavefunction files from the pwscf.save directory."""
        self.logger.info(f"{self.logprefix} Initiating SCF Cleanup for {self.wkdir}...")
        
        dcount = 0
        if self.pwscfdir and os.path.exists(self.pwscfdir):
            for f in glob(os.path.join(self.pwscfdir, "*.hdf5")):
                if os.path.basename(f) == "charge-density.hdf5":
                    continue
                
                os.remove(f)
                dcount += 1

        self.logger.info(f"{self.logprefix} Cleaned up {dcount} HDF5 files in {self.pwscfdir}")

        dcount = 0
        if self.pwscfdir and os.path.exists(self.pwscfdir):
            for f in glob(os.path.join(self.pwscfdir, "*.dat")):
                if os.path.basename(f) == "charge-density.dat":
                    continue
                
                os.remove(f)
                dcount += 1

        self.logger.info(f"{self.logprefix} Cleaned up {dcount} DAT files in {self.pwscfdir}")

    def cleanwannier(self):
        """Cleans up pwscf.save folder and other heavy files."""
        self.logger.info(f"{self.logprefix} Initiating Wannier Cleanup for {self.wkdir}")
        
        if os.path.exists(self.pwscfdir):
            shutil.rmtree(self.pwscfdir)
            self.logger.info(f"{self.logprefix} Deleted QE Save Directory: {self.pwscfdir}")
            
        extensions = ['*.mmn', '*.amn', '*.chk', '*.eig', '*.nnkp', '*wfc*', '*.dat', '*.hdf5']
        dcount = 0
        
        for ext in extensions:
            for filepath in glob(os.path.join(self.tmpdir, ext)):
                try:
                    os.remove(filepath)
                    dcount += 1
                except OSError as e:
                    self.logger.warning(f"{self.logprefix} Could not delete {filepath} - {e}")
                    
        self.logger.info(f"{self.logprefix} Deleted {dcount} heavy Wannier90/QE matrix files.")
        self.logger.info(f"{self.logprefix} --- Cleanup Complete ---\n")
