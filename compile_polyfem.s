#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=AdjointPolyfemBuild
#SBATCH --mail-type=END
#SBATCH --mail-user=ag4571@nyu.edu
#SBATCH --output=adjoint_polyfem_build_%j.out

module purge
module load cmake/3.28.0 python/intel/3.8.6 gcc/10.2.0 onetbb/intel/2021.1.1

BUILDDIR = RUNDIR=$SCRATCH/adjoint-polyfem/buildRelease
mkdir -p $BUILDDIR
cd $BUILDDIR

cmake -DCMAKE_BUILD_TYPE=Release $HOME/adjoint-polyfem/
make -j32
