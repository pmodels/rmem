#!/bin/bash -l
#SBATCH --partition=debug
##SBATCH --partition=standard
#SBATCH --time=00:05:00
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --account=project_465000098

echo "loading modules"
#-------------------------------------------------------------------------------
module load LUMI/22.12
module load gcc
module load libfabric/1.15.2.0
module list
#-------------------------------------------------------------------------------
# get a unique tag
TAG=`date '+%Y-%m-%d-%H%M'`-`uuidgen -t | head -c 4`
#-------------------------------------------------------------------------------
DBS_HOME=${HOME}/lib-PMI-4.1.1
HOME_DIR=${HOME}/rmem
SCRATCH_DIR=/scratch/project_465000098/tgillis/rmem_${TAG}_${SLURM_JOBID}

echo "--------------------------------------------------"
echo "running in ${SCRATCH_DIR}"
echo "--------------------------------------------------"

GIT_COMMIT=$(git rev-parse --short HEAD)

#-------------------------------------------------------------------------------
mkdir -p ${SCRATCH_DIR}
mkdir -p ${SCRATCH_DIR}/build
cd ${SCRATCH_DIR}
cp -r ${HOME_DIR}/src .
cp -r ${HOME_DIR}/make_arch .
cp -r ${HOME_DIR}/Makefile .

rpm -qi libfabric

make clean
OPTS="-DNO_WRITE_DATA" make fast
${DBS_HOME}/bin/mpiexec -n 2 -ppn 1 --bind-to core -l ./rmem
#-------------------------------------------------------------------------------
