#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting High-Performance Quantum ESPRESSO 7.5 Source Build..."

# 1. Update package list and install compilers, tmux, & high-perf math libraries
echo "Installing GCC, GFortran, OpenMPI, OpenBLAS, FFTW3, and tmux..."
apt-get update -y
apt-get install -y build-essential gfortran wget git tmux htop \
                   libopenmpi-dev openmpi-bin \
                   libopenblas-dev libfftw3-dev

# 2. Download Quantum ESPRESSO v7.5 Source Code
echo "Downloading QE 7.5 source code..."
cd $HOME
wget https://gitlab.com/QEF/q-e/-/archive/qe-7.5/q-e-qe-7.5.tar.gz
tar -xzf q-e-qe-7.5.tar.gz
rm q-e-qe-7.5.tar.gz
cd q-e-qe-7.5

# 3. Configure the build environment
echo "Configuring build with MPI and optimized math libraries..."
./configure --enable-parallel

# 4. Compile the PWscf package using ALL available CPU cores
NUM_CORES=$(nproc)
echo "Compiling pw.x using $NUM_CORES cores. This will take a few minutes..."
make pw -j$NUM_CORES

# 5. Add the compiled binary to the system PATH
echo "Adding QE 7.5 to system PATH..."
echo "export PATH=\$HOME/q-e-qe-7.5/bin:\$PATH" >> $HOME/.bashrc
export PATH=$HOME/q-e-qe-7.5/bin:$PATH

export OMP_NUM_THREADS=1
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# 6. Verify the installation
echo "======================================"
echo "Bare-Metal Compilation Complete for v7.5!"
echo "Testing PWscf (pw.x) executable path:"
which pw.x
echo "Testing MPI setup:"
mpirun --version
echo "======================================"
echo "To begin, run: source ~/.bashrc"