import subprocess
import os
import logging


class ShellExecutor:
    """Handles the execution of shell commands, environment injection, and MPI parsing.
       Silently logs all standard output to a file instead of dumping to the console.
    """
    def __init__(self, wkdir, logprefix):
        self.wkdir = wkdir

        self.logprefix = f"{logprefix} -> [ShellExecutor]"
        self.logger = logging.getLogger("SpinDFT")

    def runcmd(self, command, serial=False, env=None):
        if serial and "mpirun" in command:
            parts = command.split()
            executable = next((p for p in parts if ".x" in p or "wannier" in p), None)
            if executable:
                command = " ".join(parts[parts.index(executable):])
        
        self.logger.info(f"{self.logprefix} Executing: {command} (Output redirected to log)")
        
        runenv = os.environ.copy()
        if env: 
            runenv.update(env)
            
        try:
            result = subprocess.run(
                command, shell=True, cwd=self.wkdir,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, check=True, env=runenv
            )
            
            self.logger.debug(f"{'='*50}\n")
            self.logger.debug(f"COMMAND: {command}")
            self.logger.debug(f"{'='*50}\n")
            if result.stdout.strip():
                self.logger.debug(result.stdout.strip())
            if result.stderr.strip():
                self.logger.warning("--- STDERR / WARNINGS ---")
                self.logger.warning(result.stderr.strip())
            self.logger.debug("\n")
                
            return result.stdout
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"{self.logprefix} ERROR in command: {command}")
            
            self.logger.error(f"{'!'*50}")
            self.logger.error(f"CRASH IN COMMAND: {command}")
            self.logger.error(f"EXIT CODE: {e.returncode}")
            self.logger.error(f"STDERR:\n{e.stderr.strip()}")
            self.logger.error(f"{'!'*50}\n\n")
                
            raise RuntimeError(f"Command failed with code {e.returncode}. See log for details.")
