#!/bin/bash
#SBATCH --job-name=trigr
#SBATCH --partition=batch
##SBATCH --partition=gpu
#SBATCH --account=examples
#SBATCH --nodes=2
#SBATCH --tasks-per-node=2
#SBATCH --mem=240G
#SBATCH --time=0:01:00

#-------------------------------------------------------------------------------
module purge
module load GCC/11.3.0
module list

#-------------------------------------------------------------------------------
# get a unique tag
TAG=`date '+%Y-%m-%d-%H%M'`-`uuidgen -t | head -c 4`
# get the dir list
DBS_DIR=${HOME}/lib-OFI-1.18.0-dbg
HOME_DIR=${HOME}/rmem
SCRATCH_DIR=/gpfs/scratch/betatest/tgillis/benchme_${TAG}_${SLURM_JOBID}

echo "--------------------------------------------------"
echo "running in ${SCRATCH_DIR}"
echo "--------------------------------------------------"

#-------------------------------------------------------------------------------
cd ${SCRATCH_DIR}
cp -r ${HOME_DIR}/m4 .
cp -r ${HOME_DIR}/src .
cp -r ${HOME_DIR}/Makefile.am .
cp -r ${HOME_DIR}/configure.ac .
cp -r ${HOME_DIR}/autogen.sh .

#-------------------------------------------------------------------------------
./autogen.sh
CC=gcc CXX=g++ ./configure --enable-fast \
    --with-ofi=${DBS_DIR} \
    --with-pmi=${DBS_DIR} 

make clean
make -j 8

#-------------------------------------------------------------------------------
FI_PROVIDER="psm3" ${DBS_DIR}/bin/mpiexec -ppn 1 -l --bind-to core \
    -n 1 perf record --call-graph dwarf -o perf.0 ./rmem : \
    -n 1 perf record --call-graph dwarf -o perf.1 ./rmem


for id in 0 1
do
    perf script -i perf.${id} > out.${id}.perf
    ${HOME}/FlameGraph/stackcollapse-perf.pl out.${id}.perf > out.${id}.folded
    ${HOME}/FlameGraph/flamegraph.pl out.${id}.folded > ${id}.svg
done

