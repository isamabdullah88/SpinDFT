#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting Ultimate RunPod Setup: QE 7.5 + Wannier90 + TB2J Pipeline (Bare-Metal Python)..."

# 1. Update package list and install system dependencies
echo "Installing compilers, tmux, python3, and high-perf math libraries..."
apt-get update -y
apt-get install -y build-essential gfortran wget git tmux htop \
                   libopenmpi-dev openmpi-bin \
                   libopenblas-dev libfftw3-dev \
                   python3 python3-pip python3-dev

# 2. Set up the local Scratch Directory to prevent Network I/O bottlenecks
echo "Creating local scratch directory for fast NVMe I/O..."
mkdir -p /root/qe_scratch

# 3. Download Quantum ESPRESSO v7.5
echo "Downloading QE 7.5 source code..."
cd $HOME
wget https://gitlab.com/QEF/q-e/-/archive/qe-7.5/q-e-qe-7.5.tar.gz
tar -xzf q-e-qe-7.5.tar.gz
rm q-e-qe-7.5.tar.gz
cd q-e-qe-7.5

# 4. The Wannier90 Git Submodule Fix
# (Note: If you intend to compile w90, you WILL need to uncomment these lines 
# otherwise 'make w90' will fail due to the missing repository)
# echo "Fetching Wannier90 source code to fix the submodule trap..."
# rm -rf external/wannier90
# git clone --depth 1 --branch v3.1.0 https://github.com/wannier-developers/wannier90.git external/wannier90

# 5. Configure the build environment
echo "Configuring build with MPI and optimized math libraries..."
./configure --enable-parallel CFLAGS="-O3 -march=native" FFLAGS="-O3 -march=native"

# 6. Compile pw.x, wannier90.x, and pw2wannier90.x
NUM_CORES=$(nproc)
echo "Compiling pw.x using $NUM_CORES cores..."
make pw -j$NUM_CORES

echo "Compiling wannier90.x using $NUM_CORES cores..."
make w90 -j$NUM_CORES

echo "Compiling Post-Processing (pw2wannier90.x) using $NUM_CORES cores..."
make pp -j$NUM_CORES

# 7. Add QE to PATH and optimize the Linux Scheduler (OpenMP limiters)
echo "Writing environment variables to ~/.bashrc..."
cat << 'EOF' >> $HOME/.bashrc

# Quantum ESPRESSO & Wannier90 Path
export PATH=$HOME/q-e-qe-7.5/bin:$PATH

# OpenMPI Root Bypass
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# OpenMP Thread Limiters (Prevents CPU choking on MPI runs)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
EOF

# Load the new bashrc immediately for this session
export PATH=$HOME/q-e-qe-7.5/bin:$PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# 8. Install Python Libraries System-Wide
echo "Installing ASE, Numpy, TB2J and dependencies globally..."
# Using --break-system-packages to allow global installation on modern Ubuntu
pip3 install --upgrade pip --break-system-packages
pip3 install numpy ase TB2J six python-dateutil pandas --break-system-packages

# 9. Verify the installation
echo "======================================"
echo "Installation Complete!"
echo "Testing Executables:"
which pw.x
which wannier90.x
which pw2wannier90.x
echo "Testing MPI setup:"
mpirun --version
echo "Testing Python Packages:"
python3 -c "import ase, numpy; print(f'ASE version: {ase.__version__} | Numpy version: {numpy.__version__}')"
TB2J_run --help | head -n 5
echo "======================================"
echo "To begin your workflow, run: source ~/.bashrc"