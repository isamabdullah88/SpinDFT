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

    def setwkdir(self, strain):
        """Creates temporary and output directories for a specific strain."""
        self.wkdir = os.path.join(self.bwkdir, f"Strain_Uniaxial_X_{strain:.4f}")
        self.tmpdir = os.path.join(self.wkdir, "tmp")
        self.pwscfdir = os.path.join(self.tmpdir, "pwscf.save")

    def cleanscf(self):
        """Cleans up .hdf5 wavefunction files from the pwscf.save directory."""
        self.logger.info(f"{self.logprefix} Initiating SCF Cleanup for {self.wkdir}...")
        
        if self.pwscfdir and os.path.exists(self.pwscfdir):
            for f in glob(os.path.join(self.pwscfdir, "*.hdf5")):
                if os.path.basename(f) == "charge-density.hdf5":
                    continue
                
                os.remove(f)

            self.logger.info(f"{self.logprefix} Cleaned up HDF5 files in {self.pwscfdir}")

        if self.pwscfdir and os.path.exists(self.pwscfdir):
            for f in glob(os.path.join(self.pwscfdir, "*.dat")):
                if os.path.basename(f) == "charge-density.dat":
                    continue
                
                os.remove(f)
            
        self.logger.info(f"{self.logprefix} Cleaned up DAT files in {self.pwscfdir}")
        self.logger.info(f"{self.logprefix} --- SCF Cleanup Complete ---\n")

    def cleanwannier(self):
        """Cleans up pwscf.save folder and other heavy files."""
        self.logger.info(f"{self.logprefix} Initiating Wannier Cleanup for {self.wkdir}")
        
        if os.path.exists(self.pwscfdir):
            shutil.rmtree(self.pwscfdir)
            self.logger.info(f"{self.logprefix} Deleted QE Save Directory: {self.pwscfdir}")
            
        extensions = ['*.mmn', '*.amn', '*.chk', '*.eig', '*.nnkp', '*wfc*']
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
