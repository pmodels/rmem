#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=0:10:00
#SBATCH --ntasks=4
#SBATCH --nodes=2
#SBATCH --account=p200210
#SBATCH --qos=short

echo "loading modules"
#-------------------------------------------------------------------------------
module load env/staging/2022.1
module load GCC
module load UCX
module load Automake Autoconf
module load libtool
module load Perl
module load CUDA
module list

#-------------------------------------------------------------------------------
# get a unique tag
TAG=`date '+%Y-%m-%d-%H%M'`-`uuidgen -t | head -c 4`
# get the dir list
MPI_DIR=${HOME}/lib-OFI-1.18.1-CUDA-11.7.0
HOME_DIR=${HOME}/rmem
SCRATCH_DIR=/project/scratch/p200210/tgillis/benchme_${TAG}_${SLURM_JOBID}

echo "--------------------------------------------------"
echo "running in ${SCRATCH_DIR}"
echo "--------------------------------------------------"

#-------------------------------------------------------------------------------
mkdir -p ${SCRATCH_DIR}
mkdir -p ${SCRATCH_DIR}/build
cd ${SCRATCH_DIR}
cp -r ${HOME_DIR}/src .
cp -r ${HOME_DIR}/make_arch .
cp -r ${HOME_DIR}/Makefile .

#-------------------------------------------------------------------------------
export FI_VERBS_DEVICE_NAME=mlx5_0
#export FI_VERBS_RX_SIZE=2048
#export FI_VERBS_TX_SIZE=2048
#export FI_OFI_RXM_RX_SIZE=2048
#export FI_OFI_RXM_TX_SIZE=2048
#export FI_LOG_LEVEL=Debug
#export FI_VERBS_PREFER_XRC=1
#-------------------------------------------------------------------------------
MPI_OPT="-n 2 -ppn 1 -l --bind-to core"
#-------------------------------------------------------------------------------
export USE_CUDA=0
make info
make clean
#OPTS="-ldl -rdynamic" make fast
make fast
ldd rmem

#test delivery
declare -a test=(
    "-r am -d am -c delivery"
    "-r am -d am -c cq_data"
    "-r tag -d tag -c cq_data"
)
for RMEM_OPT in "${test[@]}"; do
    echo "==> ${MPI_OPT} with ${RMEM_OPT}"
    FI_PROVIDER="verbs;ofi_rxm" ${MPI_DIR}/bin/mpiexec ${MPI_OPT} ./rmem ${RMEM_OPT}
done

##-------------------------------------------------------------------------------
## test rtr and dtc
#declare -a test=(
#    "-r am -d am -c delivery"
#    "-r tag -d tag -c delivery")
#for RMEM_OPT in "${test[@]}"; do
#    echo "==> ${MPI_OPT} with ${RMEM_OPT}"
#    FI_PROVIDER="psm3" ${MPI_DIR}/bin/mpiexec ${MPI_OPT} ./rmem ${RMEM_OPT}
#done
#
##test delivery
#declare -a test=(
#    "-r am -d am -c delivery"
#    "-r am -d am -c counter"
#    "-r am -d am -c cq_data")
#for RMEM_OPT in "${test[@]}"; do
#    echo "==> ${MPI_OPT} with ${RMEM_OPT}"
#    FI_PROVIDER="psm3" ${MPI_DIR}/bin/mpiexec ${MPI_OPT} ./rmem ${RMEM_OPT}
#done
#

##-------------------------------------------------------------------------------
#FI_PROVIDER="psm3" ${MPI_DIR}/bin/mpiexec -ppn 1 -l --bind-to core \
#    -n 1 perf record --call-graph dwarf -o perf.0 ./rmem : \
#    -n 1 perf record --call-graph dwarf -o perf.1 ./rmem
#
#
#for id in 0 1
#do
#    perf script -i perf.${id} > out.${id}.perf
#    ${HOME}/FlameGraph/stackcollapse-perf.pl out.${id}.perf > out.${id}.folded
#    ${HOME}/FlameGraph/flamegraph.pl out.${id}.folded > ${id}.svg
#done
#
