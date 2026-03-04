
import logging
import shutil
import os
import sys

from config import ShellExecutor
from .fermi import FermiParser


class TB2JExchange:
    """Configures and runs the TB2J exchange calculation."""
    def __init__(self, wkdir, kmesh, soc):
        self.wkdir = wkdir
        # self.outdir = outdir
        self.prefix = 'pwscf'
        self.kmesh = kmesh
        self.soc = soc
        self.executor = ShellExecutor(self.wkdir, "[TB2J]")
        self.qe_parser = FermiParser(self.wkdir, self.prefix)

        self.logprefix = "[TB2J]"
        self.logger = logging.getLogger("SpinDFT")

    def run(self):

        # Step 4: Calculate Exchange Interactions
        efermi = self.qe_parser.efermi()

        self.logger.info(f"{self.logprefix} TB2J exchange calculation...")

        wann2jexe = shutil.which("wann2J.py") or os.path.join(os.path.dirname(sys.executable), "wann2J.py")
        if not wann2jexe or not os.path.exists(wann2jexe):
            raise RuntimeError("wann2J.py executable not found in path.")
            
        kx, ky, kz = self.kmesh
        self.logger.info(f"{self.logprefix} Detected Fermi Energy: {efermi} eV")
        
        if self.soc: 
            cmd = f"{wann2jexe} --posfile {self.prefix}.win --prefix_spinor {self.prefix} --elements Cr --kmesh {kx} {ky} {kz} --spinor --efermi {efermi}"
        else:
            cmd = f"{wann2jexe} --posfile {self.prefix}_up.win --prefix_up {self.prefix}_up --prefix_down {self.prefix}_down --elements Cr --kmesh {kx} {ky} {kz} --efermi {efermi}"
            
        self.executor.runcmd(cmd)
        
        outpath = os.path.join(self.wkdir, 'TB2J_results', 'Multibinit', 'exchange.xml')
        if os.path.exists(outpath):
            self.logger.info(f"{self.logprefix} Pipeline complete! Result saved to {outpath}")
            return {'status': 'SUCCESS', 'file': outpath}
        else:
            self.logger.warning(f"{self.logprefix} Pipeline failed: exchange.xml not found at {outpath}")
            return {'status': 'TB2J_FAIL'}