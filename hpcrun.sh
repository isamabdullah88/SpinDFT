#!/bin/bash

# --- SLURM Directives ---
#SBATCH --job-name=CrI3_RelaxedFM       # A clear name for your thesis dataset job
#SBATCH --nodes=1                      # Request exactly 1 physical node
#SBATCH --ntasks=24                    # Request all 24 cores on that node
#SBATCH --time=05-00:00:00                # Max wall time (adjust based on total configs: e.g., 100 configs * 4 mins = ~7 hours)
#SBATCH --partition=cpuonly            # (Change this if your cluster uses a different default partition)
#SBATCH --output=RelaxedFM-%j.out       # Standard output log (%j will be replaced by the unique Job ID)
#SBATCH --error=RelaxedFM-%j.err        # Standard error log
#SBATCH --exclusive                    # CRITICAL: Prevents VS Code or other users from sharing this node

# --- 1. Clean and Load System Environment ---
echo "Starting job on $HOSTNAME"
module purge
module load qe75
module load gcc openmpi mkl

# --- 2. Load Your Custom Python Environment ---
# (Fallback to home directory paths as discussed during the /scratch debugging)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qe-env

# --- 3. Execute the Python Workflow ---
# Force OpenMP threads to 1 to prevent the cache thrashing we fixed earlier
export OMP_NUM_THREADS=1

echo "Beginning dataset generation..."

# Run your Python script
# (Ensure your Python script is set to loop sequentially and uses: mpirun -np 24 pw.x -npool 8)
python run.py --WKDIR "/scratch/isam.Balghari/DataSets/Relaxed" --DBPATH "/scratch/isam.Balghari/DataSets/Relaxed/Relaxed.db" --N_CALCULATIONS 21 --CORES_PER_JOB 24 --PRERELAXED_DIR None

echo "All configurations complete!"