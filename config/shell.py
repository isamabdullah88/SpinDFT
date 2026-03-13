import subprocess
import os
import logging


class ShellExecutor:
    """Handles the execution of shell commands, environment injection, and MPI parsing.
       Silently logs all standard output to a file instead of dumping to the console.
    """
    def __init__(self, wkdir, logprefix):
        self.wkdir = wkdir

        self.logprefix = logprefix
        self.logger = logging.getLogger("SpinDFT")

        self.lfile = os.path.join(self.wkdir, f"{self.logprefix}_output.log")

    def runcmd(self, command, serial=False, env=None):
        if serial and "mpirun" in command:
            parts = command.split()
            executable = next((p for p in parts if ".x" in p or "wannier" in p), None)
            if executable:
                command = " ".join(parts[parts.index(executable):])
        
        self.logger.info(f"{self.logprefix} -> [ShellExecutor] Executing: {command} (Output redirected to log)")
        
        runenv = os.environ.copy()
        if env: 
            runenv.update(env)
            
        try:
            result = subprocess.run(
                command, shell=True, cwd=self.wkdir,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, check=True, env=runenv
            )
            
            with open(self.lfile, 'a') as f:
                f.write(f"{'='*50}\n")
                f.write(f"COMMAND: {command}\n")
                f.write(f"{'='*50}\n")
                if result.stdout.strip():
                    f.write(result.stdout.strip() + "\n")
                if result.stderr.strip():
                    f.write("--- STDERR / WARNINGS ---\n")
                    f.write(result.stderr.strip() + "\n")
                f.write("\n")
                
        except subprocess.CalledProcessError as e:
            
            with open(self.lfile, 'a') as f:
                f.write(f"{'!'*50}\n")
                f.write(f"CRASH IN COMMAND: {command}\n")
                f.write(f"EXIT CODE: {e.returncode}\n")
                if e.stderr.strip():
                    f.write(f"STDERR:\n{e.stderr.strip()}\n")
                f.write(f"{'!'*50}\n\n")
                
            self.logger.info(f"{self.logprefix} -> [ShellExecutor] ---------------------------------------------------")
            self.logger.info(f"{self.logprefix} -> [ShellExecutor] ERROR in command: {command}")
            self.logger.error(f"{self.logprefix} -> [ShellExecutor] Command failed with code {e.returncode}. See log for details.")
            self.logger.info(f"{self.logprefix} -> [ShellExecutor] ---------------------------------------------------")
                
            raise RuntimeError(f"Command failed with code {e.returncode}. See log for details.")
