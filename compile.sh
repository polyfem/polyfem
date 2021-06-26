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
#SBATCH --time=2:00:00
#SBATCH --mem=16GB

# Load modules
module purge

module load gcc/10.2.0
module load cmake/3.18.4
module load intel/19.1.2
module load boost/intel/1.74.0

export CC=${GCC_ROOT}/bin/gcc
export CXX=${GCC_ROOT}/bin/g++

export PARDISO_LIC_PATH="${HOME}/.pardiso"
export PARDISO_INSTALL_PREFIX="${HOME}/.local"
export OMP_NUM_THREADS=8

export CMAKE_INCLUDE_PATH=$(env | grep _INC= | cut -d= -f2 | xargs | sed -e 's/ /:/g')
export CMAKE_LIBRARY_PATH=$(env | grep _LIB= | cut -d= -f2 | xargs | sed -e 's/ /:/g')

# Run job
cd "${SLURM_SUBMIT_DIR}"
mkdir build
cd build

echo ${BUILD}

if [ -z "${BUILD}" ]; then
	BUILD=Release
fi

mkdir ${BUILD}
pushd ${BUILD}
cmake -DCMAKE_BUILD_TYPE=${BUILD} ../..
make -j8
popd
