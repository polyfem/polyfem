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
##SBATCH --reservation=panozzo
#SBATCH --partition=c18_25

# Load modules
module purge

module load mercurial/intel/4.0.1
module load gcc/6.3.0
module load cmake/intel/3.7.1
module load eigen/3.3.1
module load mesa/intel/17.0.2
module swap python/intel python3/intel/3.6.3

module load mpfr/gnu/3.1.5
module load zlib/intel/1.2.8
module load suitesparse/intel/4.5.4
module load lapack/gnu/3.7.0
module load gmp/gnu/6.1.2
module load mpc/gnu/1.0.3
module load cuda/8.0.44
module load tbb/intel/2017u3
module load blast+/2.7.1

export CC=${GCC_ROOT}/bin/gcc
export CXX=${GCC_ROOT}/bin/g++

export PARDISO_LIC_PATH="${HOME}/.pardiso"
export PARDISO_INSTALL_PREFIX="${HOME}/.local"
export OMP_NUM_THREADS=8

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
