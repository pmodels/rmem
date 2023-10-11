#!/bin/bash -l
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH -A CSC371_crusher
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1

echo "loading modules"
#-------------------------------------------------------------------------------
module load PrgEnv-gnu-amd
module load libfabric/1.15.2.0
module load rocm
#module load LUMI/22.12
#module load gcc
#module load PrgEnv-gnu
#module load craype-accel-amd-gfx90a
#module load rocm
module list
#-------------------------------------------------------------------------------
# get a unique tag
TAG=`date '+%Y-%m-%d-%H%M'`-`uuidgen -t | head -c 4`
#-------------------------------------------------------------------------------
DBS_DIR=${HOME}/.local/hydra
MPI_DIR=${DBS_DIR}
HOME_DIR=/ccs/proj/csc371/yguo/crusher/rmem/rmem-private
SCRATCH_DIR=/ccs/proj/csc371/yguo/crusher/scratch/rmem_${TAG}_${SLURM_JOBID}

# HIP custom 
export HIPCC_COMPILE_FLAGS_APPEND="--offload-arch=gfx90a $(CC --cray-print-opts=cflags)"
export HIPCC_LINK_FLAGS_APPEND=$(CC --cray-print-opts=libs)
#-------------------------------------------------------------------------------
# asan with cuda, from https://github.com/google/sanitizers/issues/629
# asan with pthread, from https://github.com/google/sanitizers/issues/1171
# export ASAN_OPTIONS=protect_shadow_gap=0:use_sigaltstack=0
# export TSAN_OPTIONS=second_deadlock_stack=1
#-------------------------------------------------------------------------------
export FI_HMEM_CUDA_USE_GDRCOPY=1
#export FI_CXI_OPTIMIZED_MRS=0
#export FI_LOG_LEVEL=Debug
#export FI_CXI_RDZV_THRESHOLD=4096
#export HYDRA_TOPO_DEBUG=1
echo "--------------------------------------------------"
echo "running in ${SCRATCH_DIR}"
echo "FI_CXI_OPTIMIZED_MRS = ${FI_CXI_OPTIMIZED_MRS}"
echo "FI_CXI_RDZV_EAGER_SIZE = ${FI_CXI_RDZV_EAGER_SIZE}"
echo "FI_CXI_RDZV_THRESHOLD = ${FI_CXI_RDZV_THRESHOLD}"
echo "FI_CXI_RDZV_GET_MIN = ${FI_CXI_RDZV_GET_MIN}"
echo "--------------------------------------------------"

#-------------------------------------------------------------------------------
mkdir -p ${SCRATCH_DIR}
mkdir -p ${SCRATCH_DIR}/build
cd ${SCRATCH_DIR}
cp -r ${HOME_DIR}/src .
cp -r ${HOME_DIR}/make_arch .
cp -r ${HOME_DIR}/Makefile .

#-------------------------------------------------------------------------------
# need to put 2 here to let some space of the thread!
MPI_OPT="-n 2 -ppn 1 -l --bind-to core:2"
#-------------------------------------------------------------------------------
export PMI_DIR=${DBS_DIR}
# for device in 0 1; do
for device in 1; do
    export USE_HIP=${device}
    make clean ARCH_FILE=make_arch/crusher.mak
    #make info debug
    # make info debug ARCH_FILE=make_arch/crusher.mak
    # make info verbose ARCH_FILE=make_arch/crusher.mak
    make info fast ARCH_FILE=make_arch/crusher.mak
    ldd rmem
    #test delivery
    declare -a test=(
        "-r am -d am -c delivery"
        "-r am -d am -c counter"
        "-r am -d tag -c fence"
        #"-r am -d tag -c delivery"
    )
    for RMEM_OPT in "${test[@]}"; do
        echo "==> ${MPI_OPT} with ${RMEM_OPT} - HIP? ${USE_HIP}"
        FI_PROVIDER="cxi" ${MPI_DIR}/bin/mpiexec ${MPI_OPT} ./rmem ${RMEM_OPT}
        # FI_PROVIDER="cxi" srun -n2 --ntasks-per-node=1 --gpus-per-node=1 --gpu-bind=closest ./rmem ${RMEM_OPT}
    done
done

#echo "--------------------------------------------------"
#echo "running in ${SCRATCH_DIR}"
#echo "--------------------------------------------------"
#
#GIT_COMMIT=$(git rev-parse --short HEAD)
#
##-------------------------------------------------------------------------------
#mkdir -p ${SCRATCH_DIR}
#mkdir -p ${SCRATCH_DIR}/build
#cd ${SCRATCH_DIR}
#cp -r ${HOME_DIR}/src .
#cp -r ${HOME_DIR}/make_arch .
#cp -r ${HOME_DIR}/Makefile .
#
#rpm -qi libfabric
#
#make clean
#OPTS="-DNO_WRITE_DATA" make fast
#${DBS_HOME}/bin/mpiexec -n 2 -ppn 1 --bind-to core -l ./rmem
##-------------------------------------------------------------------------------
