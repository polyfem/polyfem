#!/bin/bash
#
# Submit job as (build defaults to Release):
#
#   sbatch compile.sh
#   sbatch --export=BUILD='Debug',ALL compile.sh 
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=8GB

# Load modules
module purge

module load gcc/9.1.0
module load cmake/intel/3.16.3
module load mesa/intel/17.0.2
module swap python/intel python3/intel/3.6.3
module load boost/intel/1.71.0
module load mpfr/gnu/3.1.5
module load zlib/intel/1.2.8
module load mpc/gnu/1.0.3
module load blast+/2.7.1

export CC=${GCC_ROOT}/bin/gcc
export CXX=${GCC_ROOT}/bin/g++

export PARDISO_LIC_PATH="${HOME}/.pardiso"
export PARDISO_INSTALL_PREFIX="${HOME}/.local"
export OMP_NUM_THREADS=8

export CMAKE_INCLUDE_PATH=$(env | grep _INC= | cut -d= -f2 | xargs | sed -e 's/ /:/g')
export CMAKE_LIBRARY_PATH=$(env | grep _LIB= | cut -d= -f2 | xargs | sed -e 's/ /:/g')

# Run job
mkdir build
cd build
cmake ..
make -j 8